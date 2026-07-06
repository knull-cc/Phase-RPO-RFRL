import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.PhaseRPO_RFRL import TemporalPhaseRetrievalBank, RetrievalActionController, AdaptiveFusion


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


def _parse_alpha_bins(value):
    if isinstance(value, str):
        values = [item.strip() for item in value.split(',') if item.strip()]
        bins = [float(item) for item in values]
    elif isinstance(value, (list, tuple)):
        bins = [float(item) for item in value]
    else:
        bins = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]

    bins = sorted({min(max(item, 0.0), 1.0) for item in bins})
    if 0.0 not in bins:
        bins = [0.0] + bins
    return bins or [0.0, 0.02, 0.05, 0.10]


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
        self.rfrl_regret_loss_weight = getattr(configs, 'rfrl_regret_loss_weight', 0.5)
        self.host_anchor_loss_weight = getattr(configs, 'host_anchor_loss_weight', 0.5)
        self.retrieval_adapter_loss_weight = getattr(configs, 'retrieval_adapter_loss_weight', 0.02)
        self.retrieval_correction_reg_weight = getattr(configs, 'retrieval_correction_reg_weight', 0.001)
        self.retrieval_correction_clip = getattr(configs, 'retrieval_correction_clip', 2.0)
        self.retrieval_cost = getattr(configs, 'retrieval_cost', 0.01)
        self.rpo_gain_margin = getattr(configs, 'rpo_gain_margin', 0.0)
        self.rfrl_gain_margin = getattr(configs, 'rfrl_gain_margin', 0.0)
        self.retrieval_residual_scale = nn.Parameter(
            torch.tensor(float(getattr(configs, 'retrieval_residual_init', 0.1)))
        )
        self.use_phase_rpo_rfrl = getattr(configs, 'use_phase_rpo_rfrl', True)
        self.latest_aux_loss = None
        self.latest_aux_details = {}
        self.latest_diagnostics = {}

        legacy_top_k = getattr(configs, 'phase_top_k', 32)
        retrieval_top_k = getattr(configs, 'retrieval_top_k', None)
        retrieval_top_k = legacy_top_k if retrieval_top_k is None else retrieval_top_k
        retrieval_top_m = getattr(configs, 'retrieval_top_m', 8)
        max_freqs = getattr(configs, 'phase_max_freqs', 24)
        legacy_temperature = getattr(configs, 'phase_temperature', 0.10)
        temperature = getattr(configs, 'retrieval_temperature', None)
        temperature = legacy_temperature if temperature is None else temperature
        max_bank_size = getattr(configs, 'phase_max_bank_size', 4096)
        exclusion_radius = getattr(configs, 'phase_exclusion_radius', self.seq_len + self.pred_len)
        self.retrieval_bank = TemporalPhaseRetrievalBank(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            primary_top_k=retrieval_top_k,
            rerank_top_m=retrieval_top_m,
            max_freqs=max_freqs,
            time_key_len=getattr(configs, 'retrieval_time_key_len', 96),
            temperature=temperature,
            retrieval_mode=getattr(configs, 'retrieval_mode', 'time_phase_rerank'),
            phase_weight=getattr(configs, 'phase_similarity_weight', 0.20),
            amplitude_weight=getattr(configs, 'amplitude_similarity_weight', 0.10),
            time_weight=getattr(configs, 'time_similarity_weight', 1.00),
            max_bank_size=max_bank_size,
            exclusion_radius=exclusion_radius,
        )
        self.phase_retrieval = self.retrieval_bank
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
        self.rfrl_controller = RetrievalActionController(
            channels=self.channels,
            hidden_size=getattr(configs, 'rfrl_hidden_size', 64),
            alpha_bins=_parse_alpha_bins(getattr(configs, 'rfrl_alpha_bins', '0,0.01,0.02,0.05,0.1,0.2,0.4')),
            no_retrieval_bias=getattr(configs, 'rfrl_no_retrieval_bias', 2.0),
        )
        self.adaptive_fusion = AdaptiveFusion()

        nn.init.zeros_(self.retrieval_adapter[-1].weight)
        nn.init.zeros_(self.retrieval_adapter[-1].bias)

    def prepare_retrieval_bank(self, train_data, device=None):
        if not self.use_phase_rpo_rfrl:
            return
        device = device or next(self.parameters()).device
        print('Building time-primary Phase-RPO-RFRL retrieval bank from train split...')
        self.retrieval_bank.build(train_data, device=device)

    def host_forecast(self, x_enc, return_state=False):
        baseline, host_state, norm_stats = self.host(x_enc, return_state=True)
        if return_state:
            return baseline, host_state, norm_stats
        return baseline

    def forecast(self, x_enc):
        return self.host_forecast(x_enc)

    def _comparison_tensors(self, target_y, baseline, retrieval_forecast,
                            retrieval_correction, final):
        target = target_y[:, -self.pred_len:, :]
        if self.features == 'MS':
            return (
                target[:, :, -1:],
                baseline[:, :, -1:],
                retrieval_forecast[:, :, -1:],
                retrieval_correction[:, :, -1:],
                final[:, :, -1:],
            )
        return target, baseline, retrieval_forecast, retrieval_correction, final

    def forecast_with_retrieval(self, x_enc, batch_index=None, mode='train', target_y=None):
        baseline, host_state, norm_stats = self.host_forecast(x_enc, return_state=True)
        if not self.use_phase_rpo_rfrl:
            self.latest_aux_loss = None
            self.latest_aux_details = {}
            self.latest_diagnostics = {'baseline': baseline.detach()}
            return baseline

        retrieved_delta, retrieval_info = self.retrieval_bank.retrieve(
            x_enc, batch_index=batch_index, mode=mode
        )
        stdev = norm_stats['stdev'].clamp_min(1e-5)
        retrieved_delta_norm = retrieved_delta / stdev
        host_guided_bias = self.host_to_retrieval(host_state)
        retrieval_input = retrieved_delta_norm.permute(0, 2, 1) + host_guided_bias
        retrieval_correction_norm = self.retrieval_adapter(retrieval_input).permute(0, 2, 1)
        if self.retrieval_correction_clip > 0:
            retrieval_correction_norm = self.retrieval_correction_clip * torch.tanh(
                retrieval_correction_norm / self.retrieval_correction_clip
            )
        retrieval_scale = self.retrieval_residual_scale.clamp(0.0, 1.0)
        retrieval_correction = retrieval_scale * retrieval_correction_norm * stdev

        raw_retrieval_forecast = x_enc[:, -1:, :] + retrieved_delta
        retrieval_forecast = baseline + retrieval_correction

        preference_score, action_alpha, action_probabilities, action_logits, action_index = self.rfrl_controller(
            x_enc, baseline, retrieval_forecast, retrieval_info
        )
        retrieval_enhanced = retrieval_forecast
        final, fusion_weight = self.adaptive_fusion(baseline, retrieval_enhanced, action_alpha)

        self.latest_aux_loss = None
        self.latest_aux_details = {}
        oracle_alpha = None
        oracle_index = None
        oracle_gain = None
        policy_regret = None
        baseline_err = None
        retrieval_err = None
        final_err = None
        oracle_err = None
        if target_y is not None:
            target, baseline_cmp, retrieval_cmp, correction_cmp, final_cmp = self._comparison_tensors(
                target_y, baseline, retrieval_forecast, retrieval_correction, final
            )

            with torch.no_grad():
                alpha_bins = self.rfrl_controller.alpha_bins.to(
                    device=x_enc.device, dtype=baseline_cmp.dtype
                )
                alpha_candidates = (
                    baseline_cmp.detach().unsqueeze(0)
                    + alpha_bins.view(-1, 1, 1, 1) * correction_cmp.detach().unsqueeze(0)
                )
                alpha_err = ((alpha_candidates - target.detach().unsqueeze(0)) ** 2).mean(dim=(2, 3))
                baseline_err = F.mse_loss(
                    baseline_cmp.detach(), target.detach(), reduction='none'
                ).mean(dim=(1, 2))

                action_gain = baseline_err.unsqueeze(0) - alpha_err
                action_score = action_gain.clone()
                nonzero_actions = alpha_bins.view(-1, 1) > 0
                if self.rfrl_gain_margin > 0:
                    action_score = action_score - self.rfrl_gain_margin * nonzero_actions.to(action_score.dtype)
                action_score[0] = 0.0
                oracle_index = torch.argmax(action_score, dim=0)
                oracle_err = alpha_err.gather(0, oracle_index.view(1, -1)).view(-1)
                oracle_alpha = alpha_bins[oracle_index].view(-1, 1, 1)
                oracle_gain = baseline_err - oracle_err
                retrieval_err = F.mse_loss(
                    retrieval_cmp.detach(), target.detach(), reduction='none'
                ).mean(dim=(1, 2))
                final_err = F.mse_loss(
                    final_cmp.detach(), target.detach(), reduction='none'
                ).mean(dim=(1, 2))
                expected_policy_err = torch.sum(
                    action_probabilities.detach() * alpha_err.transpose(0, 1),
                    dim=1,
                )
                policy_regret = final_err - oracle_err
                scale = baseline_err.mean().clamp_min(1e-6)
                rpo_target = (oracle_gain > self.rpo_gain_margin).to(preference_score.dtype)

            if self.training:
                rpo_target_detached = rpo_target.detach()
                pos_rate = rpo_target_detached.mean().clamp(1e-3, 1.0 - 1e-3)
                rpo_weight = torch.where(
                    rpo_target_detached > 0,
                    0.5 / pos_rate,
                    0.5 / (1.0 - pos_rate),
                )
                preference_loss = F.binary_cross_entropy(
                    preference_score.view(-1).clamp(1e-5, 1.0 - 1e-5),
                    rpo_target_detached,
                    weight=rpo_weight,
                )
                action_counts = torch.bincount(
                    oracle_index.detach(),
                    minlength=self.rfrl_controller.alpha_bins.numel(),
                ).to(action_logits.dtype)
                action_weight = action_counts.sum() / action_counts.clamp_min(1.0)
                action_weight = action_weight / action_weight.mean().clamp_min(1e-6)
                policy_loss = F.cross_entropy(
                    action_logits,
                    oracle_index.detach(),
                    weight=action_weight.to(action_logits.device),
                )
                expected_policy_err = torch.sum(
                    action_probabilities * alpha_err.transpose(0, 1).detach(),
                    dim=1,
                )
                regret_loss = ((expected_policy_err - oracle_err.detach()) / scale.detach()).mean()

                adapter_target = target.detach() - baseline_cmp.detach()
                adapter_effect = oracle_alpha.detach() * correction_cmp
                adapter_weight = (oracle_alpha.detach().view(-1) > 0).to(correction_cmp.dtype)
                adapter_per_sample = F.smooth_l1_loss(
                    adapter_effect, adapter_target, reduction='none'
                ).mean(dim=(1, 2))
                if adapter_weight.sum() > 0:
                    adapter_loss = (adapter_per_sample * adapter_weight).sum() / adapter_weight.sum().clamp_min(1.0)
                else:
                    adapter_loss = correction_cmp.sum() * 0.0

                correction_reg_loss = (retrieval_correction / stdev).pow(2).mean()
                alpha_bins_for_cost = self.rfrl_controller.alpha_bins.to(
                    device=x_enc.device, dtype=action_probabilities.dtype
                )
                cost_loss = torch.sum(action_probabilities * alpha_bins_for_cost.view(1, -1), dim=1).mean()
                host_anchor_loss = F.mse_loss(baseline_cmp, target)
                self.latest_aux_loss = (
                    self.host_anchor_loss_weight * host_anchor_loss
                    +
                    self.rpo_loss_weight * preference_loss
                    + self.rfrl_loss_weight * (
                        policy_loss
                        + self.rfrl_regret_loss_weight * regret_loss
                        + self.retrieval_cost * cost_loss
                    )
                    + self.retrieval_adapter_loss_weight * (
                        adapter_loss + self.retrieval_correction_reg_weight * correction_reg_loss
                    )
                )
                self.latest_aux_details = {
                    'preference_loss': preference_loss.detach(),
                    'policy_loss': policy_loss.detach(),
                    'regret_loss': regret_loss.detach(),
                    'adapter_loss': adapter_loss.detach(),
                    'correction_reg_loss': correction_reg_loss.detach(),
                    'retrieval_cost_loss': cost_loss.detach(),
                    'host_anchor_loss': host_anchor_loss.detach(),
                    'baseline_err': baseline_err.mean().detach(),
                    'retrieval_err': retrieval_err.mean().detach(),
                    'final_err': final_err.mean().detach(),
                }

        self.latest_diagnostics = {
            'baseline': baseline.detach(),
            'raw_retrieval_forecast': raw_retrieval_forecast.detach(),
            'retrieval_forecast': retrieval_forecast.detach(),
            'retrieval_correction': retrieval_correction.detach(),
            'retrieval_enhanced': retrieval_enhanced.detach(),
            'preference_score': preference_score.detach(),
            'action_alpha': action_alpha.detach(),
            'action_index': action_index.detach(),
            'action_probabilities': action_probabilities.detach(),
            'no_retrieval_probability': action_probabilities[:, :1].detach(),
            'action_alpha_bins': self.rfrl_controller.alpha_bins.to(
                device=x_enc.device, dtype=x_enc.dtype
            ).view(1, -1).repeat(x_enc.size(0), 1).detach(),
            'fusion_weight': fusion_weight.detach(),
            'top_similarity': retrieval_info['top_similarity'].detach(),
            'primary_top_similarity': retrieval_info['primary_top_similarity'].detach(),
            'time_similarity': retrieval_info['time_similarity'].detach(),
            'phase_similarity': retrieval_info['phase_similarity'].detach(),
            'amplitude_similarity': retrieval_info['amplitude_similarity'].detach(),
            'top_indices': retrieval_info['top_indices'],
            'primary_top_indices': retrieval_info['primary_top_indices'],
            'host_state_norm': host_state.detach().norm(dim=-1).mean(dim=-1),
            'retrieval_residual_scale': retrieval_scale.detach().view(1, 1).repeat(x_enc.size(0), 1),
            'retrieval_correction_norm': retrieval_correction_norm.detach().abs().mean(dim=(1, 2)),
        }
        if oracle_alpha is not None:
            self.latest_diagnostics['oracle_alpha'] = oracle_alpha.detach()
            self.latest_diagnostics['oracle_action_index'] = oracle_index.detach()
            self.latest_diagnostics['oracle_gain'] = oracle_gain.detach()
            self.latest_diagnostics['policy_regret'] = policy_regret.detach()
            self.latest_diagnostics['baseline_err'] = baseline_err.detach()
            self.latest_diagnostics['retrieval_err'] = retrieval_err.detach()
            self.latest_diagnostics['oracle_err'] = oracle_err.detach()
            self.latest_diagnostics['final_err'] = final_err.detach()
        return final

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                batch_index=None, mode='train', target_y=None):
        if self.task_name != 'long_term_forecast':
            raise NotImplementedError('PhaseRPO_RFRL_MLP currently supports long_term_forecast only.')
        return self.forecast_with_retrieval(
            x_enc, batch_index=batch_index, mode=mode, target_y=target_y
        )[:, -self.pred_len:, :]
