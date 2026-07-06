import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RAFT_RPO import RAFTRetrievalBank, RPOActionScorer


class Model(nn.Module):
    """
    RAFT retrieval with utility-grounded Retrieval Preference Optimization.

    Actions:
    0. no retrieval: use the host linear forecast
    1. RAFT fused retrieval forecast
    2..G+1. one forecast per RAFT periodic retrieval branch
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.features = getattr(configs, 'features', 'M')

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)

        self.retrieval_bank = RAFTRetrievalBank(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=getattr(configs, 'n_period', 3),
            topm=getattr(configs, 'topm', 20),
            temperature=getattr(configs, 'raft_temperature', 0.1),
            max_bank_size=getattr(configs, 'raft_max_bank_size', 4096),
            exclusion_radius=(
                getattr(configs, 'raft_exclusion_radius', 0)
                or self.seq_len + self.pred_len
            ),
        )
        self.period_num = self.retrieval_bank.period_num
        self.retrieval_pred = nn.ModuleList([
            nn.Linear(self.pred_len // period, self.pred_len)
            for period in self.period_num
        ])
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)

        self.rpo_controller = RPOActionScorer(
            channels=self.channels,
            period_num=self.period_num,
            hidden_size=getattr(configs, 'rpo_hidden_size', 64),
            no_retrieval_bias=getattr(configs, 'rpo_no_retrieval_bias', 1.0),
            softmax_temperature=getattr(configs, 'rpo_softmax_temperature', 1.0),
        )
        self.rpo_loss_weight = getattr(configs, 'rpo_loss_weight', 0.1)
        self.rpo_pairwise_loss_weight = getattr(configs, 'rpo_pairwise_loss_weight', 0.2)
        self.rpo_retrieval_loss_weight = getattr(configs, 'rpo_retrieval_loss_weight', 0.2)
        self.rpo_entropy_weight = getattr(configs, 'rpo_entropy_weight', 0.0)
        self.rpo_gain_margin = getattr(configs, 'rpo_gain_margin', 0.0)
        self.host_anchor_loss_weight = getattr(configs, 'host_anchor_loss_weight', 0.5)
        self.rpo_hard_eval = getattr(configs, 'rpo_hard_eval', False)

        self.latest_aux_loss = None
        self.latest_aux_details = {}
        self.latest_diagnostics = {}

    def prepare_retrieval_bank(self, train_data, device=None):
        device = device or next(self.parameters()).device
        print('Building RAFT-RPO retrieval bank from train split...')
        self.retrieval_bank.build(train_data, device=device)

    def prepare_dataset(self, train_data, valid_data=None, test_data=None):
        self.prepare_retrieval_bank(train_data, device=next(self.parameters()).device)

    def host_forecast(self, x_enc):
        x_offset = x_enc[:, -1:, :].detach()
        x_norm = x_enc - x_offset
        baseline_residual = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)
        return baseline_residual + x_offset, baseline_residual, x_offset

    def _period_candidates(self, period_retrieval, x_offset):
        period_outputs = []
        for i, period in enumerate(self.period_num):
            retrieved = period_retrieval[i]
            bsz, pred_len, channels = retrieved.shape
            compressed = retrieved.reshape(bsz, pred_len // period, period, channels)
            compressed = compressed[:, :, 0, :]
            out = self.retrieval_pred[i](compressed.permute(0, 2, 1)).permute(0, 2, 1)
            period_outputs.append(out)
        period_residuals = torch.stack(period_outputs, dim=1)
        period_forecasts = period_residuals + x_offset.unsqueeze(1)
        return period_residuals, period_forecasts

    def _comparison_tensors(self, target_y, baseline, all_actions, final):
        target = target_y[:, -self.pred_len:, :]
        if self.features == 'MS':
            return (
                target[:, :, -1:],
                baseline[:, :, -1:],
                all_actions[:, :, :, -1:],
                final[:, :, -1:],
            )
        return target, baseline, all_actions, final

    @staticmethod
    def _gather_action(actions, index):
        bsz = actions.size(0)
        view_shape = [bsz] + [1] * (actions.ndim - 2)
        gather_index = index.view(*view_shape).expand(-1, *actions.shape[2:])
        return actions.gather(1, gather_index.unsqueeze(1)).squeeze(1)

    @staticmethod
    def _balanced_action_weight(action_index, action_count, dtype, device):
        counts = torch.bincount(action_index, minlength=action_count).to(device=device, dtype=dtype)
        weights = counts.sum() / counts.clamp_min(1.0)
        return weights / weights.mean().clamp_min(1e-6)

    def forecast_with_retrieval(self, x_enc, batch_index=None, mode='train', target_y=None):
        baseline, baseline_residual, x_offset = self.host_forecast(x_enc)
        period_retrieval, retrieval_info = self.retrieval_bank.retrieve(
            x_enc, batch_index=batch_index, mode=mode
        )
        period_residuals, period_forecasts = self._period_candidates(period_retrieval, x_offset)
        retrieval_sum = period_residuals.sum(dim=1)
        raft_residual = self.linear_pred(
            torch.cat([baseline_residual, retrieval_sum], dim=1).permute(0, 2, 1)
        ).permute(0, 2, 1)
        raft_forecast = raft_residual + x_offset

        candidates = torch.cat([raft_forecast.unsqueeze(1), period_forecasts], dim=1)
        preference_score, action_probabilities, action_logits, action_index = self.rpo_controller(
            x_enc, baseline, candidates, retrieval_info
        )
        all_actions = torch.cat([baseline.unsqueeze(1), candidates], dim=1)
        expected_final = torch.einsum('ba,bapc->bpc', action_probabilities, all_actions)
        selected_forecast = self._gather_action(all_actions, action_index)
        if self.rpo_hard_eval and not self.training:
            final = selected_forecast
        else:
            final = expected_final

        retrieval_probability = action_probabilities[:, 1:].sum(dim=1, keepdim=True).clamp_min(1e-6)
        conditional_retrieval = torch.einsum(
            'bk,bkpc->bpc',
            action_probabilities[:, 1:] / retrieval_probability,
            candidates,
        )

        self.latest_aux_loss = None
        self.latest_aux_details = {}
        oracle_index = None
        oracle_gain = None
        policy_regret = None
        baseline_err = None
        retrieval_err = None
        final_err = None
        oracle_err = None
        action_err = None
        action_gain = None
        best_retrieval_gain = None
        oracle_forecast = None

        if target_y is not None:
            target, baseline_cmp, all_actions_cmp, final_cmp = self._comparison_tensors(
                target_y, baseline, all_actions, final
            )
            action_err = ((all_actions_cmp - target.unsqueeze(1)) ** 2).mean(dim=(2, 3))
            baseline_err = action_err[:, 0]
            retrieval_err = action_err[:, 1]
            final_err = F.mse_loss(final_cmp, target, reduction='none').mean(dim=(1, 2))
            action_gain = baseline_err.unsqueeze(1) - action_err

            with torch.no_grad():
                action_score = action_gain.detach().clone()
                action_score[:, 1:] = action_score[:, 1:] - self.rpo_gain_margin
                action_score[:, 0] = 0.0
                oracle_index = torch.argmax(action_score, dim=1)
                oracle_err = action_err.detach().gather(1, oracle_index.view(-1, 1)).view(-1)
                oracle_gain = baseline_err.detach() - oracle_err
                policy_regret = final_err.detach() - oracle_err
                best_retrieval_gain = action_gain.detach()[:, 1:].max(dim=1).values
                oracle_forecast = self._gather_action(all_actions.detach(), oracle_index)

            if self.training:
                action_weight = self._balanced_action_weight(
                    oracle_index.detach(),
                    all_actions.size(1),
                    action_logits.dtype,
                    action_logits.device,
                )
                policy_loss = F.cross_entropy(
                    action_logits,
                    oracle_index.detach(),
                    weight=action_weight,
                )

                retrieval_preferred = (
                    action_gain.detach()[:, 1:] > self.rpo_gain_margin
                ).to(action_logits.dtype)
                pair_logits = action_logits[:, 1:] - action_logits[:, :1]
                pos_rate = retrieval_preferred.mean().clamp(1e-3, 1.0 - 1e-3)
                pair_weight = torch.where(
                    retrieval_preferred > 0,
                    0.5 / pos_rate,
                    0.5 / (1.0 - pos_rate),
                )
                pairwise_loss = F.binary_cross_entropy_with_logits(
                    pair_logits,
                    retrieval_preferred,
                    weight=pair_weight,
                )
                always_retrieval_loss = action_err[:, 1].mean()
                host_anchor_loss = F.mse_loss(baseline_cmp, target)
                entropy = -(
                    action_probabilities * (action_probabilities + 1e-8).log()
                ).sum(dim=1).mean()
                self.latest_aux_loss = (
                    self.host_anchor_loss_weight * host_anchor_loss
                    + self.rpo_loss_weight * (
                        policy_loss + self.rpo_pairwise_loss_weight * pairwise_loss
                    )
                    + self.rpo_retrieval_loss_weight * always_retrieval_loss
                    - self.rpo_entropy_weight * entropy
                )
                self.latest_aux_details = {
                    'rpo_policy_loss': policy_loss.detach(),
                    'rpo_pairwise_loss': pairwise_loss.detach(),
                    'rpo_retrieval_loss': always_retrieval_loss.detach(),
                    'host_anchor_loss': host_anchor_loss.detach(),
                    'rpo_entropy': entropy.detach(),
                    'baseline_err': baseline_err.mean().detach(),
                    'retrieval_err': retrieval_err.mean().detach(),
                    'final_err': final_err.mean().detach(),
                }

        action_names = torch.tensor(
            [0, 1] + self.period_num,
            dtype=x_enc.dtype,
            device=x_enc.device,
        ).view(1, -1).repeat(x_enc.size(0), 1)
        self.latest_diagnostics = {
            'baseline': baseline.detach(),
            'raw_retrieval_forecast': raft_forecast.detach(),
            'retrieval_forecast': conditional_retrieval.detach(),
            'retrieval_enhanced': final.detach(),
            'rpo_selected_forecast': selected_forecast.detach(),
            'preference_score': preference_score.detach(),
            'rpo_accept_probability': preference_score.detach().view(-1, 1),
            'action_probabilities': action_probabilities.detach(),
            'rpo_action_probabilities': action_probabilities.detach(),
            'action_index': action_index.detach(),
            'no_retrieval_probability': action_probabilities[:, :1].detach(),
            'fusion_weight': preference_score.detach(),
            'top_similarity': retrieval_info['top_similarity'].detach(),
            'primary_top_similarity': retrieval_info['primary_top_similarity'].detach(),
            'period_similarity': retrieval_info['period_similarity'].detach(),
            'top_indices': retrieval_info['top_indices'].detach(),
            'primary_top_indices': retrieval_info['primary_top_indices'].detach(),
            'retrieval_weights': retrieval_info['weights'].detach(),
            'rpo_action_names': action_names.detach(),
        }
        if oracle_index is not None:
            self.latest_diagnostics.update({
                'oracle_action_index': oracle_index.detach(),
                'oracle_gain': oracle_gain.detach(),
                'policy_regret': policy_regret.detach(),
                'baseline_err': baseline_err.detach(),
                'retrieval_err': retrieval_err.detach(),
                'final_err': final_err.detach(),
                'oracle_err': oracle_err.detach(),
                'rpo_candidate_mse': action_err.detach(),
                'rpo_candidate_gain': action_gain.detach(),
                'rpo_best_retrieval_gain': best_retrieval_gain.detach(),
                'rpo_oracle_forecast': oracle_forecast.detach(),
            })
        return final

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                batch_index=None, mode='train', target_y=None):
        if self.task_name != 'long_term_forecast':
            raise NotImplementedError('RAFT_RPO_MLP currently supports long_term_forecast only.')
        return self.forecast_with_retrieval(
            x_enc, batch_index=batch_index, mode=mode, target_y=target_y
        )[:, -self.pred_len:, :]
