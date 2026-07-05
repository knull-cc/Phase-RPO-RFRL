import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.PhaseRPO_RFRL import PhaseAwareRetrievalBank, RetrievalPreferenceController, AdaptiveFusion
from models.DLinear import Model as DLinearBaseline


class Model(DLinearBaseline):
    """
    DLinear host with the Phase-RPO-RFRL retrieval-control plugin.
    Use --model PhaseRPO_RFRL_DLinear for this method.
    Use --model DLinear for the pure baseline.
    """

    def __init__(self, configs, individual=False):
        super(Model, self).__init__(configs)
        self.features = getattr(configs, 'features', 'M')

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
        self.retrieval_adapter = nn.Sequential(
            nn.Linear(self.pred_len, self.pred_len),
            nn.GELU(),
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

    def forecast(self, x_enc):
        return super(Model, self).forecast(x_enc)

    def forecast_with_retrieval(self, x_enc, batch_index=None, mode='train', target_y=None):
        baseline = self.forecast(x_enc)
        if not self.use_phase_rpo_rfrl:
            self.latest_aux_loss = None
            self.latest_diagnostics = {'baseline': baseline.detach()}
            return baseline

        raw_retrieval, retrieval_info = self.phase_retrieval.retrieve(x_enc, batch_index=batch_index, mode=mode)
        residual = raw_retrieval - x_enc[:, -1:, :]
        retrieval_forecast = self.retrieval_adapter(residual.permute(0, 2, 1)).permute(0, 2, 1)
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
        }
        return final

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                batch_index=None, mode='train', target_y=None):
        return self.forecast_with_retrieval(
            x_enc, batch_index=batch_index, mode=mode, target_y=target_y
        )[:, -self.pred_len:, :]
