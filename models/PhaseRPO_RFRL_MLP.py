import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.PhaseRPO_RFRL import PhaseAwareRetrievalBank, RetrievalPreferenceController, AdaptiveFusion


class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def normalize(self, x):
        mean = x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        return (x - mean) / stdev, mean, stdev

    def denormalize(self, x, mean, stdev):
        return x * stdev + mean


class SpectralResidualMLPHost(nn.Module):
    """
    Frequency-aware temporal MLP host for Phase-RPO-RFRL.
    It uses RevIN and an input-derived spectral context only.
    """

    def __init__(self, seq_len, pred_len, hidden_dim, dropout=0.1,
                 spectral_bins=16, use_revin=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.spectral_bins = min(max(int(spectral_bins), 0), max(seq_len // 2, 0))
        self.use_revin = use_revin
        self.revin = RevIN()

        self.input_proj = nn.Linear(seq_len, hidden_dim)
        if self.spectral_bins > 0:
            self.spectral_proj = nn.Sequential(
                nn.Linear(2 * self.spectral_bins, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.spectral_proj = None

        self.temporal_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )
        self.shortcut_head = nn.Linear(seq_len, pred_len)

    def _spectral_context(self, x_cf):
        if self.spectral_proj is None:
            return torch.zeros(
                x_cf.size(0), x_cf.size(1), self.hidden_dim,
                dtype=x_cf.dtype, device=x_cf.device,
            )

        x_centered = x_cf - x_cf.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(x_centered, dim=-1)[..., 1:self.spectral_bins + 1]
        amplitude = spectrum.abs()
        phase_cos = spectrum.real / (amplitude + 1e-6)
        phase_sin = spectrum.imag / (amplitude + 1e-6)
        amp_scale = torch.log1p(amplitude)
        amp_scale = amp_scale / (amp_scale.mean(dim=-1, keepdim=True) + 1e-6)
        spectral_feature = torch.cat([phase_cos * amp_scale, phase_sin * amp_scale], dim=-1)
        return self.spectral_proj(spectral_feature)

    def forward(self, x, return_state=False):
        if self.use_revin:
            x_norm, mean, stdev = self.revin.normalize(x)
        else:
            x_norm = x
            mean = torch.zeros_like(x[:, :1, :])
            stdev = torch.ones_like(x[:, :1, :])

        x_cf = x_norm.permute(0, 2, 1)
        projected = self.input_proj(x_cf)
        state = projected + self._spectral_context(x_cf)
        hidden = state + self.temporal_mlp(state)
        out_norm = self.output_proj(hidden) + self.shortcut_head(x_cf)
        out = out_norm.permute(0, 2, 1)

        if self.use_revin:
            out = self.revin.denormalize(out, mean, stdev)

        stats = {'mean': mean, 'stdev': stdev}
        if return_state:
            return out, hidden, stats
        return out


class Model(nn.Module):
    """
    Phase-RPO-RFRL backbone with a frequency-aware residual MLP host.
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
        self.use_revin = getattr(configs, 'mlp_use_revin', True)
        self.host_spectral_bins = getattr(configs, 'mlp_spectral_bins',
                                          getattr(configs, 'phase_max_freqs', 16))

        self.host = SpectralResidualMLPHost(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            hidden_dim=self.hidden_dim,
            dropout=self.mlp_dropout,
            spectral_bins=self.host_spectral_bins,
            use_revin=self.use_revin,
        )

        self.rpo_loss_weight = getattr(configs, 'rpo_loss_weight', 0.1)
        self.rfrl_loss_weight = getattr(configs, 'rfrl_loss_weight', 0.05)
        self.retrieval_cost = getattr(configs, 'retrieval_cost', 0.01)
        self.retrieval_residual_scale = nn.Parameter(
            torch.tensor(float(getattr(configs, 'retrieval_residual_init', 0.1)))
        )
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
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.mlp_dropout),
            nn.Linear(self.hidden_dim, self.pred_len),
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

        nn.init.zeros_(self.retrieval_adapter[-1].weight)
        nn.init.zeros_(self.retrieval_adapter[-1].bias)

    def prepare_retrieval_bank(self, train_data, device=None):
        if not self.use_phase_rpo_rfrl:
            return
        device = device or next(self.parameters()).device
        print('Building Phase-RPO-RFRL retrieval bank from train split...')
        self.phase_retrieval.build(train_data, device=device)

    def host_forecast(self, x_enc, return_state=False):
        baseline, host_state, norm_stats = self.host(x_enc, return_state=True)
        if return_state:
            return baseline, host_state, norm_stats
        return baseline

    def forecast(self, x_enc):
        return self.host_forecast(x_enc)

    def forecast_with_retrieval(self, x_enc, batch_index=None, mode='train', target_y=None):
        baseline, host_state, norm_stats = self.host_forecast(x_enc, return_state=True)
        if not self.use_phase_rpo_rfrl:
            self.latest_aux_loss = None
            self.latest_diagnostics = {'baseline': baseline.detach()}
            return baseline

        phase_delta, retrieval_info = self.phase_retrieval.retrieve(
            x_enc, batch_index=batch_index, mode=mode
        )
        stdev = norm_stats['stdev'].clamp_min(1e-5)
        phase_delta_norm = phase_delta / stdev
        host_guided_bias = self.host_to_retrieval(host_state)
        retrieval_input = phase_delta_norm.permute(0, 2, 1) + host_guided_bias
        retrieval_correction_norm = self.retrieval_adapter(retrieval_input).permute(0, 2, 1)
        retrieval_correction = self.retrieval_residual_scale * retrieval_correction_norm * stdev

        raw_retrieval_forecast = x_enc[:, -1:, :] + phase_delta
        retrieval_forecast = baseline + retrieval_correction

        preference_score, guidance = self.rfrl_controller(x_enc, baseline, retrieval_forecast, retrieval_info)
        retrieval_enhanced = baseline + preference_score * retrieval_correction
        final, fusion_weight = self.adaptive_fusion(baseline, retrieval_enhanced, guidance)

        self.latest_aux_loss = None
        if self.training and target_y is not None:
            target = target_y[:, -self.pred_len:, :]
            if self.features == 'MS':
                target = target[:, :, -1:]
                baseline_cmp = baseline[:, :, -1:]
                retrieval_cmp = retrieval_forecast[:, :, -1:]
                final_cmp = final[:, :, -1:]
            else:
                baseline_cmp = baseline
                retrieval_cmp = retrieval_forecast
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
            'raw_retrieval_forecast': raw_retrieval_forecast.detach(),
            'retrieval_forecast': retrieval_forecast.detach(),
            'retrieval_correction': retrieval_correction.detach(),
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
