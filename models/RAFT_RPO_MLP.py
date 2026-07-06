import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RAFT_RPO import RAFTRetrievalBank, RPOCandidateScorer


class Model(nn.Module):
    """
    RAFT retrieval with forecast-utility-aware Retrieval Preference Optimization.

    RAFT remains the recall/reference module:
    - the original RAFT soft top-m aggregation is kept as the always-retrieve
      reference forecast;
    - RPO only reranks the individual RAFT top-m candidates and predicts whether
      that reranked forecast should replace the reference forecast.
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

        reference_temperature = getattr(
            configs,
            'rpo_reference_temperature',
            getattr(configs, 'raft_temperature', 0.1),
        )
        if reference_temperature is None:
            reference_temperature = getattr(configs, 'raft_temperature', 0.1)
        policy_temperature = getattr(
            configs,
            'rpo_policy_temperature',
            None,
        )
        if policy_temperature is None:
            policy_temperature = getattr(configs, 'rpo_softmax_temperature', 1.0)
        self.rpo_controller = RPOCandidateScorer(
            hidden_size=getattr(configs, 'rpo_hidden_size', 64),
            reference_temperature=reference_temperature,
            policy_temperature=policy_temperature,
            score_alpha=getattr(configs, 'rpo_score_alpha', 1.0),
            gate_temperature=getattr(configs, 'rpo_gate_temperature', 0.05),
            gate_epsilon=getattr(configs, 'rpo_gate_epsilon', 0.0),
        )

        self.rpo_loss_weight = getattr(configs, 'rpo_loss_weight', 0.1)
        self.rpo_pairwise_loss_weight = getattr(configs, 'rpo_pairwise_loss_weight', 1.0)
        self.rpo_gate_loss_weight = getattr(configs, 'rpo_gate_loss_weight', 0.5)
        self.rpo_utility_loss_weight = getattr(configs, 'rpo_utility_loss_weight', 0.5)
        self.rpo_top1_loss_weight = getattr(configs, 'rpo_top1_loss_weight', 0.0)
        self.rpo_retrieval_loss_weight = getattr(configs, 'rpo_retrieval_loss_weight', 0.2)
        self.rpo_entropy_weight = getattr(configs, 'rpo_entropy_weight', 0.0)
        self.rpo_beta = getattr(configs, 'rpo_beta', 1.0)
        self.rpo_gain_margin = getattr(configs, 'rpo_gain_margin', 0.0)
        self.rpo_pair_margin = getattr(configs, 'rpo_pair_margin', self.rpo_gain_margin)
        if self.rpo_pair_margin is None:
            self.rpo_pair_margin = self.rpo_gain_margin
        self.rpo_gate_epsilon = getattr(configs, 'rpo_gate_epsilon', 0.0)
        self.host_anchor_loss_weight = getattr(configs, 'host_anchor_loss_weight', 0.5)
        self.rpo_hard_eval = getattr(configs, 'rpo_hard_eval', False)
        self.rpo_utility_reference = getattr(configs, 'rpo_utility_reference', 'raft')
        if self.rpo_utility_reference not in {'raft', 'baseline'}:
            self.rpo_utility_reference = 'raft'

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

    def _candidate_period_rank(self, retrieval_info):
        top_similarity = retrieval_info['top_similarity']
        bsz, g_num, topm = top_similarity.shape
        device = top_similarity.device
        period = torch.tensor(self.period_num, device=device, dtype=top_similarity.dtype)
        period = period.view(1, g_num, 1).expand(bsz, g_num, topm)
        rank = torch.arange(topm, device=device, dtype=top_similarity.dtype)
        rank = rank.view(1, 1, topm).expand(bsz, g_num, topm)
        return period.reshape(bsz, -1), rank.reshape(bsz, -1)

    def _individual_candidate_forecasts(
        self,
        candidate_y_mg,
        period_residuals,
        retrieval_sum,
        baseline_residual,
        x_offset,
    ):
        bsz, _, topm, _, _ = candidate_y_mg.shape
        forecasts = []
        residuals = []
        for i, period in enumerate(self.period_num):
            candidates = candidate_y_mg[:, i]
            flat = candidates.reshape(bsz * topm, self.pred_len, self.channels)
            compressed = flat.reshape(bsz * topm, self.pred_len // period, period, self.channels)
            compressed = compressed[:, :, 0, :]
            candidate_residual = self.retrieval_pred[i](
                compressed.permute(0, 2, 1)
            ).permute(0, 2, 1).reshape(bsz, topm, self.pred_len, self.channels)

            context_residual = (
                retrieval_sum.unsqueeze(1)
                - period_residuals[:, i:i + 1]
                + candidate_residual
            )
            baseline_context = baseline_residual.unsqueeze(1).expand(-1, topm, -1, -1)
            fused_input = torch.cat([baseline_context, context_residual], dim=2)
            fused_input = fused_input.reshape(bsz * topm, 2 * self.pred_len, self.channels)
            fused_residual = self.linear_pred(
                fused_input.permute(0, 2, 1)
            ).permute(0, 2, 1).reshape(bsz, topm, self.pred_len, self.channels)

            residuals.append(candidate_residual)
            forecasts.append(fused_residual + x_offset.unsqueeze(1))
        return torch.cat(forecasts, dim=1), torch.cat(residuals, dim=1)

    def _select_feature(self, value):
        if self.features == 'MS':
            return value[..., -1:]
        return value

    @staticmethod
    def _gather_candidate(candidates, index):
        bsz = candidates.size(0)
        gather_index = index.view(bsz, 1, 1, 1).expand(-1, 1, *candidates.shape[2:])
        return candidates.gather(1, gather_index).squeeze(1)

    def _reference_forecast(self, baseline, raft_forecast):
        if self.rpo_utility_reference == 'baseline':
            return baseline
        return raft_forecast

    def _binary_preference_loss(self, gate_logit, target):
        target = target.to(gate_logit.dtype)
        pos_rate = target.mean().clamp(1e-3, 1.0 - 1e-3)
        weight = torch.where(target > 0, 0.5 / pos_rate, 0.5 / (1.0 - pos_rate))
        return F.binary_cross_entropy_with_logits(gate_logit, target, weight=weight)

    def _candidate_preference_pair_mask(self, candidate_mae):
        better_than = candidate_mae.unsqueeze(2) + self.rpo_pair_margin
        worse_than = candidate_mae.unsqueeze(1)
        return better_than < worse_than

    def _dpo_pair_loss(self, candidate_mae, policy_log_probability, reference_log_probability):
        pair_mask = self._candidate_preference_pair_mask(candidate_mae)
        if not pair_mask.any():
            return policy_log_probability.sum() * 0.0, pair_mask

        policy_delta = policy_log_probability.unsqueeze(2) - policy_log_probability.unsqueeze(1)
        reference_delta = reference_log_probability.unsqueeze(2) - reference_log_probability.unsqueeze(1)
        logits = self.rpo_beta * (policy_delta - reference_delta)
        return -F.logsigmoid(logits[pair_mask]).mean(), pair_mask

    def forecast_with_retrieval(self, x_enc, batch_index=None, mode='train', target_y=None):
        baseline, baseline_residual, x_offset = self.host_forecast(x_enc)
        period_retrieval, retrieval_info = self.retrieval_bank.retrieve(
            x_enc, batch_index=batch_index, mode=mode
        )
        period_residuals, _ = self._period_candidates(period_retrieval, x_offset)
        retrieval_sum = period_residuals.sum(dim=1)
        raft_residual = self.linear_pred(
            torch.cat([baseline_residual, retrieval_sum], dim=1).permute(0, 2, 1)
        ).permute(0, 2, 1)
        raft_forecast = raft_residual + x_offset
        reference_forecast = self._reference_forecast(baseline, raft_forecast)

        candidate_forecasts, candidate_residuals = self._individual_candidate_forecasts(
            retrieval_info['candidate_y_mg'],
            period_residuals,
            retrieval_sum,
            baseline_residual,
            x_offset,
        )
        candidate_similarity = retrieval_info['top_similarity'].reshape(x_enc.size(0), -1)
        candidate_period, candidate_rank = self._candidate_period_rank(retrieval_info)

        rpo = self.rpo_controller(
            x_enc,
            reference_forecast,
            candidate_forecasts,
            candidate_similarity,
            candidate_period,
            candidate_rank,
        )
        reranked_forecast = torch.einsum(
            'bk,bkpc->bpc',
            rpo['policy_probability'],
            candidate_forecasts,
        )
        accept_probability = rpo['accept_probability']
        soft_final = (
            accept_probability * reranked_forecast
            + (1.0 - accept_probability) * reference_forecast
        )
        selected_candidate = self._gather_candidate(candidate_forecasts, rpo['candidate_index'])
        hard_accept = (rpo['predicted_utility'] > self.rpo_gate_epsilon).view(-1, 1, 1)
        selected_forecast = torch.where(hard_accept, selected_candidate, reference_forecast)
        if self.rpo_hard_eval and not self.training:
            final = selected_forecast
        else:
            final = soft_final

        action_probabilities = torch.cat([
            1.0 - accept_probability.view(-1, 1),
            accept_probability.view(-1, 1) * rpo['policy_probability'],
        ], dim=1)
        action_index = torch.where(
            hard_accept.view(-1),
            rpo['candidate_index'] + 1,
            torch.zeros_like(rpo['candidate_index']),
        )

        self.latest_aux_loss = None
        self.latest_aux_details = {}

        oracle_action_index = None
        oracle_candidate_index = None
        oracle_forecast = None
        oracle_gain_mse = None
        oracle_gain_mae = None
        policy_regret = None
        pair_mask = None
        candidate_mse = None
        candidate_mae = None
        candidate_gain_mse = None
        candidate_gain_mae = None
        reference_mse = None
        reference_mae = None
        baseline_mse = None
        baseline_mae = None
        raft_mse = None
        raft_mae = None
        final_mse = None
        final_mae = None
        reranked_mse = None
        reranked_mae = None
        oracle_mse = None
        oracle_mae = None
        best_candidate_mse = None
        best_candidate_mae = None
        diagnostic_pair_mask = None

        if target_y is not None:
            target = self._select_feature(target_y[:, -self.pred_len:, :])
            baseline_cmp = self._select_feature(baseline)
            raft_cmp = self._select_feature(raft_forecast)
            reference_cmp = self._select_feature(reference_forecast)
            candidate_cmp = self._select_feature(candidate_forecasts)
            final_cmp = self._select_feature(final)
            reranked_cmp = self._select_feature(reranked_forecast)

            candidate_mse = ((candidate_cmp - target.unsqueeze(1)) ** 2).mean(dim=(2, 3))
            candidate_mae = (candidate_cmp - target.unsqueeze(1)).abs().mean(dim=(2, 3))
            diagnostic_pair_mask = self._candidate_preference_pair_mask(candidate_mae.detach())
            reference_mse = ((reference_cmp - target) ** 2).mean(dim=(1, 2))
            reference_mae = (reference_cmp - target).abs().mean(dim=(1, 2))
            baseline_mse = ((baseline_cmp - target) ** 2).mean(dim=(1, 2))
            baseline_mae = (baseline_cmp - target).abs().mean(dim=(1, 2))
            raft_mse = ((raft_cmp - target) ** 2).mean(dim=(1, 2))
            raft_mae = (raft_cmp - target).abs().mean(dim=(1, 2))
            final_mse = ((final_cmp - target) ** 2).mean(dim=(1, 2))
            final_mae = (final_cmp - target).abs().mean(dim=(1, 2))
            reranked_mse = ((reranked_cmp - target) ** 2).mean(dim=(1, 2))
            reranked_mae = (reranked_cmp - target).abs().mean(dim=(1, 2))

            candidate_gain_mse = reference_mse.unsqueeze(1) - candidate_mse
            candidate_gain_mae = reference_mae.unsqueeze(1) - candidate_mae
            best_candidate_mae, oracle_candidate_index = candidate_mae.min(dim=1)
            best_candidate_mse = candidate_mse.gather(1, oracle_candidate_index.view(-1, 1)).view(-1)
            oracle_gain_mae = reference_mae - best_candidate_mae
            oracle_gain_mse = reference_mse - best_candidate_mse
            oracle_use = oracle_gain_mae > self.rpo_gain_margin
            oracle_action_index = torch.where(
                oracle_use,
                oracle_candidate_index + 1,
                torch.zeros_like(oracle_candidate_index),
            )
            oracle_candidate_forecast = self._gather_candidate(
                candidate_forecasts.detach(),
                oracle_candidate_index,
            )
            oracle_forecast = torch.where(
                oracle_use.view(-1, 1, 1),
                oracle_candidate_forecast,
                reference_forecast.detach(),
            )
            oracle_cmp = self._select_feature(oracle_forecast)
            oracle_mse = ((oracle_cmp - target) ** 2).mean(dim=(1, 2))
            oracle_mae = (oracle_cmp - target).abs().mean(dim=(1, 2))
            policy_regret = final_mse.detach() - oracle_mse.detach()

            if self.training:
                pair_loss, pair_mask = self._dpo_pair_loss(
                    candidate_mae.detach(),
                    rpo['policy_log_probability'],
                    rpo['reference_log_probability'],
                )
                gate_target = oracle_use.detach().to(rpo['gate_logit'].dtype)
                gate_loss = self._binary_preference_loss(rpo['gate_logit'], gate_target)
                utility_loss = F.smooth_l1_loss(
                    rpo['predicted_utility'],
                    oracle_gain_mae.detach(),
                )
                top1_loss = F.cross_entropy(
                    rpo['policy_logits'],
                    oracle_candidate_index.detach(),
                )
                host_anchor_loss = F.mse_loss(baseline_cmp, target)
                raft_branch_loss = F.mse_loss(raft_cmp, target)
                policy_entropy = -(
                    rpo['policy_probability'] * rpo['policy_log_probability']
                ).sum(dim=1).mean()

                self.latest_aux_loss = (
                    self.host_anchor_loss_weight * host_anchor_loss
                    + self.rpo_retrieval_loss_weight * raft_branch_loss
                    + self.rpo_loss_weight * (
                        self.rpo_pairwise_loss_weight * pair_loss
                        + self.rpo_gate_loss_weight * gate_loss
                        + self.rpo_utility_loss_weight * utility_loss
                        + self.rpo_top1_loss_weight * top1_loss
                    )
                    - self.rpo_entropy_weight * policy_entropy
                )
                pair_count = (
                    pair_mask.float().sum(dim=(1, 2)).mean().detach()
                    if pair_mask is not None else torch.zeros((), device=x_enc.device)
                )
                self.latest_aux_details = {
                    'rpo_pair_loss': pair_loss.detach(),
                    'rpo_gate_loss': gate_loss.detach(),
                    'rpo_utility_loss': utility_loss.detach(),
                    'rpo_top1_loss': top1_loss.detach(),
                    'host_anchor_loss': host_anchor_loss.detach(),
                    'raft_branch_loss': raft_branch_loss.detach(),
                    'rpo_policy_entropy': policy_entropy.detach(),
                    'rpo_pair_count': pair_count,
                    'reference_mae': reference_mae.mean().detach(),
                    'oracle_gain_mae': oracle_gain_mae.mean().detach(),
                    'final_mae': final_mae.mean().detach(),
                }

        diag_baseline = self._select_feature(baseline).detach()
        diag_raft = self._select_feature(raft_forecast).detach()
        diag_reference = self._select_feature(reference_forecast).detach()
        diag_reranked = self._select_feature(reranked_forecast).detach()
        diag_final = self._select_feature(final).detach()
        diag_selected = self._select_feature(selected_forecast).detach()
        reference_is_raft = 1.0 if self.rpo_utility_reference == 'raft' else 0.0
        reference_flag = torch.full(
            (x_enc.size(0), 1),
            reference_is_raft,
            device=x_enc.device,
            dtype=x_enc.dtype,
        )
        self.latest_diagnostics = {
            'baseline': diag_baseline,
            'raw_retrieval_forecast': diag_raft,
            'rpo_reference_forecast': diag_reference,
            'retrieval_forecast': diag_reranked,
            'rpo_reranked_forecast': diag_reranked,
            'retrieval_enhanced': diag_final,
            'rpo_selected_forecast': diag_selected,
            'preference_score': rpo['predicted_utility'].detach().view(-1, 1),
            'rpo_predicted_utility': rpo['predicted_utility'].detach().view(-1, 1),
            'rpo_accept_probability': accept_probability.detach().view(-1, 1),
            'action_probabilities': action_probabilities.detach(),
            'rpo_action_probabilities': action_probabilities.detach(),
            'action_index': action_index.detach(),
            'no_retrieval_probability': (1.0 - accept_probability.view(-1, 1)).detach(),
            'fusion_weight': accept_probability.detach().view(-1, 1),
            'top_similarity': retrieval_info['top_similarity'].detach(),
            'primary_top_similarity': retrieval_info['primary_top_similarity'].detach(),
            'period_similarity': retrieval_info['period_similarity'].detach(),
            'top_indices': retrieval_info['top_indices'].detach(),
            'primary_top_indices': retrieval_info['primary_top_indices'].detach(),
            'retrieval_weights': retrieval_info['weights'].detach(),
            'rpo_candidate_similarity': candidate_similarity.detach(),
            'rpo_candidate_period': candidate_period.detach(),
            'rpo_candidate_rank': candidate_rank.detach(),
            'rpo_candidate_indices': retrieval_info['top_indices'].reshape(x_enc.size(0), -1).detach(),
            'rpo_candidate_scores': rpo['scores'].detach(),
            'rpo_policy_probabilities': rpo['policy_probability'].detach(),
            'rpo_reference_probabilities': rpo['reference_probability'].detach(),
            'rpo_reference_is_raft': reference_flag.detach(),
            'rpo_candidate_residual_mean_abs': candidate_residuals.abs().mean(dim=(2, 3)).detach(),
        }
        if oracle_action_index is not None:
            rpo_action_names = torch.cat([
                torch.zeros(x_enc.size(0), 1, device=x_enc.device, dtype=x_enc.dtype),
                candidate_period.to(dtype=x_enc.dtype),
            ], dim=1)
            pair_count_per_sample = (
                diagnostic_pair_mask.float().sum(dim=(1, 2))
                if diagnostic_pair_mask is not None
                else torch.zeros(x_enc.size(0), device=x_enc.device, dtype=x_enc.dtype)
            )
            self.latest_diagnostics.update({
                'rpo_action_names': rpo_action_names.detach(),
                'oracle_action_index': oracle_action_index.detach(),
                'oracle_candidate_index': oracle_candidate_index.detach(),
                'oracle_gain': oracle_gain_mse.detach(),
                'oracle_gain_mae': oracle_gain_mae.detach(),
                'policy_regret': policy_regret.detach(),
                'baseline_err': baseline_mse.detach(),
                'baseline_mae_err': baseline_mae.detach(),
                'reference_err': reference_mse.detach(),
                'reference_mae_err': reference_mae.detach(),
                'retrieval_err': raft_mse.detach(),
                'retrieval_mae_err': raft_mae.detach(),
                'reranked_err': reranked_mse.detach(),
                'reranked_mae_err': reranked_mae.detach(),
                'final_err': final_mse.detach(),
                'final_mae_err': final_mae.detach(),
                'oracle_err': oracle_mse.detach(),
                'oracle_mae_err': oracle_mae.detach(),
                'rpo_candidate_mse': candidate_mse.detach(),
                'rpo_candidate_mae': candidate_mae.detach(),
                'rpo_candidate_gain': candidate_gain_mse.detach(),
                'rpo_candidate_gain_mae': candidate_gain_mae.detach(),
                'rpo_best_candidate_mse': best_candidate_mse.detach(),
                'rpo_best_candidate_mae': best_candidate_mae.detach(),
                'rpo_best_candidate_gain_mae': oracle_gain_mae.detach(),
                'rpo_pair_count': pair_count_per_sample.detach(),
                'rpo_oracle_forecast': self._select_feature(oracle_forecast).detach(),
            })
        return final

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                batch_index=None, mode='train', target_y=None):
        if self.task_name != 'long_term_forecast':
            raise NotImplementedError('RAFT_RPO_MLP currently supports long_term_forecast only.')
        return self.forecast_with_retrieval(
            x_enc, batch_index=batch_index, mode=mode, target_y=target_y
        )[:, -self.pred_len:, :]
