import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.PhaseRPO_RFRL import PhaseAwareRetrievalBank, RetrievalPreferenceController, AdaptiveFusion


class Model(nn.Module):
    """
    Default Phase-RPO-RFRL backbone with a two-layer MLP host.
    This is a backbone-style version rather than a plug-and-play wrapper.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.features = getattr(configs, 'features', 'M')

        self.hidden_dim = getattr(configs, 'mlp_hidden_dim', configs.d_model)
        self.mlp_dropout = getattr(configs, 'mlp_dropout', configs.dropout)
        self.use_host_norm = getattr(configs, 'mlp_use_layernorm', True)

        self.input_norm = nn.LayerNorm(self.seq_len) if self.use_host_norm else nn.Identity()
        self.host_backbone = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.mlp_dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.baseline_head = nn.Linear(self.hidden_dim, self.pred_len)
        self.shortcut_head = nn.Linear(self.seq_len, self.pred_len)

        self.rpo_loss_weight = getattr(configs, 'rpo_loss_weight', 0.1)
        self.rfrl_loss_weight = getattr(configs, 'rfrl_loss_weight', 0.05)
        self.retrieval_cost = getattr(configs, 'retrieval_cost', 0.01)
        self.use_phase_rpo_rfrl = getattr(configs, 'use_phase_rpo_rfrl', True)
        self.latest_aux_loss = None
        self.latest_diagnostics = {}

        top_k = getattr(configs, 'phase_top_k', 8)
        max_freqs = getattr(configs, 'phase_max_freqs', 16)
        temperature = getattr(configs, 'phase_temperature', 0.07)
        max_bank_size = getattr(configs, 'phase_max_bank_size', 4096)
        exclusion_radius = getattr(configs, 'phase_exclusion_radius', self.seq_len + self.pred_len)
        self.phase_retrieval = PhaseAwareRetrievalBank(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            top_k=top_k,
            max_freqs=max_freqs,
            temperature=temperature,
            phase_weight=getattr(configs, 'phase_similarity_weight', 0.55),
            amplitude_weight=getattr(configs, 'amplitude_similarity_weight', 0.25),
            time_weight=getattr(configs, 'time_similarity_weight', 0.20),
            max_bank_size=max_bank_size,
            exclusion_radius=exclusion_radius,
        )
        self.host_to_retrieval = nn.Sequential(
            nn.Linear(self.hidden_dim, self.pred_len),
            nn.GELU(),
        )
        self.retrieval_adapter = nn.Sequential(
            nn.Linear(self.pred_len, self.pred_len),
            nn.GELU(),
            nn.Dropout(self.mlp_dropout),
            nn.Linear(self.pred_len, self.pred_len),
        )
        self.rfrl_controller = RetrievalPreferenceController(
            channels=self.channels,
            hidden_size=getattr(configs, 'rfrl_hidden_size', 64),
        )
        self.adaptive_fusion = AdaptiveFusion(channels=self.channels)

    def prepare_retrieval_bank(self, train_data, device=None):
        if not self.use_phase_rpo_rfrl:
            return
        device = device or next(self.parameters()).device
        print('Building Phase-RPO-RFRL retrieval bank from train split...')
        self.phase_retrieval.build(train_data, device=device)

    def encode_host(self, x_enc):
        host_input = x_enc.permute(0, 2, 1)
        host_input = self.input_norm(host_input)
        host_state = self.host_backbone(host_input)
        return host_input, host_state

    def host_forecast(self, x_enc, return_state=False):
        host_input, host_state = self.encode_host(x_enc)
        baseline = self.baseline_head(host_state) + self.shortcut_head(host_input)
        baseline = baseline.permute(0, 2, 1)
        if return_state:
            return baseline, host_state
        return baseline

    def forecast(self, x_enc):
        return self.host_forecast(x_enc)

    def forecast_with_retrieval(self, x_enc, batch_index=None, mode='train', target_y=None):
        baseline, host_state = self.host_forecast(x_enc, return_state=True)
        if not self.use_phase_rpo_rfrl:
            self.latest_aux_loss = None
            self.latest_diagnostics = {'baseline': baseline.detach()}
            return baseline

        raw_retrieval, retrieval_info = self.phase_retrieval.retrieve(x_enc, batch_index=batch_index, mode=mode)
        residual = raw_retrieval - x_enc[:, -1:, :]
        host_guided_bias = self.host_to_retrieval(host_state)
        retrieval_input = residual.permute(0, 2, 1) + host_guided_bias
        retrieval_forecast = self.retrieval_adapter(retrieval_input).permute(0, 2, 1)
        retrieval_forecast = retrieval_forecast + x_enc[:, -1:, :]

        preference_score, guidance = self.rfrl_controller(x_enc, baseline, retrieval_forecast, retrieval_info)
        retrieval_enhanced = baseline + preference_score * (retrieval_forecast - baseline)
        final, fusion_weight = self.adaptive_fusion(baseline, retrieval_enhanced, guidance)

        self.latest_aux_loss = None
        if self.training and target_y is not None:
            target = target_y[:, -self.pred_len:, :]
            if self.features == 'MS':
                target = target[:, :, -1:]
                baseline_cmp = baseline[:, :, -1:]
                retrieval_cmp = retrieval_enhanced[:, :, -1:]
                final_cmp = final[:, :, -1:]
            else:
                baseline_cmp = baseline
                retrieval_cmp = retrieval_enhanced
                final_cmp = final

            baseline_err = F.mse_loss(baseline_cmp.detach(), target, reduction='none').mean(dim=(1, 2))
            retrieval_err = F.mse_loss(retrieval_cmp.detach(), target, reduction='none').mean(dim=(1, 2))
            final_err = F.mse_loss(final_cmp.detach(), target, reduction='none').mean(dim=(1, 2))
            utility = torch.sigmoid((baseline_err - retrieval_err) / (baseline_err.detach().mean() + 1e-6))
            preference_loss = F.binary_cross_entropy(
                preference_score.view(-1).clamp(1e-5, 1.0 - 1e-5),
                utility.detach(),
            )
            policy_target = torch.sigmoid((baseline_err - final_err) / (baseline_err.detach().mean() + 1e-6))
            policy_loss = F.mse_loss(guidance.mean(dim=(1, 2)), policy_target.detach())
            cost_loss = fusion_weight.mean()
            self.latest_aux_loss = (
                self.rpo_loss_weight * preference_loss
                + self.rfrl_loss_weight * (policy_loss + self.retrieval_cost * cost_loss)
            )

        self.latest_diagnostics = {
            'baseline': baseline.detach(),
            'retrieval_forecast': retrieval_forecast.detach(),
            'retrieval_enhanced': retrieval_enhanced.detach(),
            'preference_score': preference_score.detach(),
            'fusion_weight': fusion_weight.detach(),
            'top_similarity': retrieval_info['top_similarity'].detach(),
            'top_indices': retrieval_info['top_indices'],
            'host_state_norm': host_state.detach().norm(dim=-1).mean(dim=-1),
        }
        return final

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                batch_index=None, mode='train', target_y=None):
        if self.task_name != 'long_term_forecast':
            raise NotImplementedError('PhaseRPO_RFRL_MLP currently supports long_term_forecast only.')
        return self.forecast_with_retrieval(
            x_enc, batch_index=batch_index, mode=mode, target_y=target_y
        )[:, -self.pred_len:, :]
