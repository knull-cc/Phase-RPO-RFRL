import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPhaseRetrievalBank(nn.Module):
    """
    Retrieval bank for the next Phase-RPO-RFRL prototype.

    The primary retrieval signal is time-domain similarity. Phase and amplitude
    are used as a second-stage reranker over the temporal top-K candidates.
    This keeps phase as the method prior without making weak phase-only keys
    carry the whole retrieval problem.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        primary_top_k=32,
        rerank_top_m=8,
        max_freqs=24,
        time_key_len=96,
        temperature=0.10,
        retrieval_mode='time_phase_rerank',
        phase_weight=0.20,
        amplitude_weight=0.10,
        time_weight=1.00,
        max_bank_size=4096,
        exclusion_radius=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.primary_top_k = max(1, int(primary_top_k))
        self.rerank_top_m = max(1, int(rerank_top_m))
        self.max_freqs = max(0, int(max_freqs))
        self.time_key_len = max(1, int(time_key_len))
        self.temperature = temperature
        self.retrieval_mode = retrieval_mode
        self.phase_weight = phase_weight
        self.amplitude_weight = amplitude_weight
        self.time_weight = time_weight
        self.max_bank_size = max_bank_size
        self.exclusion_radius = exclusion_radius or (seq_len + pred_len)

        self.register_buffer('memory_x', torch.empty(0), persistent=False)
        self.register_buffer('memory_delta', torch.empty(0), persistent=False)
        self.register_buffer('memory_phase', torch.empty(0), persistent=False)
        self.register_buffer('memory_amplitude', torch.empty(0), persistent=False)
        self.register_buffer('memory_time', torch.empty(0), persistent=False)
        self.register_buffer('memory_indices', torch.empty(0, dtype=torch.long), persistent=False)

    @property
    def is_ready(self):
        return self.memory_x.numel() > 0

    def build(self, train_dataset, device):
        xs, deltas, indices = [], [], []
        n_items = len(train_dataset)
        if self.max_bank_size and n_items > self.max_bank_size:
            keep = torch.linspace(0, n_items - 1, steps=self.max_bank_size).long().unique().tolist()
        else:
            keep = range(n_items)

        for item_idx in keep:
            sample = train_dataset[item_idx]
            if len(sample) == 5:
                index, seq_x, seq_y, _, _ = sample
            else:
                index = item_idx
                seq_x, seq_y, _, _ = sample
            seq_x = torch.as_tensor(seq_x, dtype=torch.float32)
            seq_y = torch.as_tensor(seq_y[-self.pred_len:], dtype=torch.float32)
            xs.append(seq_x)
            deltas.append(seq_y - seq_x[-1:, :])
            indices.append(int(index))

        if not xs:
            return

        self.memory_x = torch.stack(xs, dim=0).to(device)
        self.memory_delta = torch.stack(deltas, dim=0).to(device)
        self.memory_indices = torch.tensor(indices, dtype=torch.long, device=device)
        self.memory_phase, self.memory_amplitude, self.memory_time = self._encode(self.memory_x)

    def _frequency_bins(self, spectrum_size, device):
        total = max(spectrum_size - 1, 0)
        limit = min(self.max_freqs, total)
        if limit <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if limit >= total:
            return torch.arange(1, total + 1, dtype=torch.long, device=device)

        low_count = min(total, max(2, limit // 2))
        low_bins = torch.arange(1, low_count + 1, dtype=torch.long, device=device)
        log_bins = torch.logspace(
            0.0,
            math.log10(float(total)),
            steps=limit,
            device=device,
        ).round().long()
        bins = torch.cat([low_bins, log_bins]).clamp(1, total).unique(sorted=True)
        if bins.numel() > limit:
            select = torch.linspace(0, bins.numel() - 1, steps=limit, device=device).round().long()
            bins = bins[select].unique(sorted=True)
        return bins

    def _time_key(self, x_norm):
        x_cf = x_norm.permute(0, 2, 1)
        key_len = min(self.time_key_len, x_cf.size(-1))
        pooled = F.adaptive_avg_pool1d(x_cf, key_len)
        if x_cf.size(-1) > 1:
            diff = x_cf[..., 1:] - x_cf[..., :-1]
            diff = F.adaptive_avg_pool1d(diff, key_len)
        else:
            diff = torch.zeros_like(pooled)
        return torch.cat([pooled, diff], dim=-1).flatten(start_dim=1)

    def _encode(self, x):
        x_centered = x - x.mean(dim=1, keepdim=True)
        x_std = x_centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        x_norm = x_centered / x_std

        time_key = F.normalize(self._time_key(x_norm), dim=1)
        spectrum = torch.fft.rfft(x_norm, dim=1)
        bins = self._frequency_bins(spectrum.size(1), x.device)
        if bins.numel() == 0:
            phase = torch.zeros(x.size(0), 2 * x.size(2), device=x.device, dtype=x.dtype)
            amplitude = torch.zeros(x.size(0), x.size(2), device=x.device, dtype=x.dtype)
        else:
            spectrum = spectrum.index_select(1, bins)
            amplitude_raw = spectrum.abs()
            phase_cos = spectrum.real / (amplitude_raw + 1e-6)
            phase_sin = spectrum.imag / (amplitude_raw + 1e-6)
            amp_scale = torch.log1p(amplitude_raw)
            amp_scale = amp_scale / (amp_scale.mean(dim=1, keepdim=True) + 1e-6)
            phase = torch.cat([phase_cos * amp_scale, phase_sin * amp_scale], dim=1)
            phase = phase.flatten(start_dim=1)
            amplitude = torch.log1p(amplitude_raw).flatten(start_dim=1)

        phase = F.normalize(phase, dim=1)
        amplitude = F.normalize(amplitude, dim=1)
        return phase, amplitude, time_key

    def _empty_diagnostics(self, x):
        zeros = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        return {
            'top_indices': None,
            'primary_top_indices': None,
            'top_similarity': zeros,
            'primary_top_similarity': zeros,
            'weights': None,
            'phase_similarity': zeros.view(-1),
            'time_similarity': zeros.view(-1),
            'amplitude_similarity': zeros.view(-1),
            'rerank_similarity': zeros.view(-1),
            'primary_similarity': zeros.view(-1),
        }

    def _mask_training_neighbors(self, score, raw_score, batch_index, mode):
        if mode != 'train' or batch_index is None:
            return score

        index = batch_index.to(score.device).long().view(-1, 1)
        distance = (self.memory_indices.view(1, -1) - index).abs()
        score = score.masked_fill(distance <= self.exclusion_radius, float('-inf'))
        empty_rows = torch.isinf(score).all(dim=1)
        if empty_rows.any():
            score = score.clone()
            score[empty_rows] = raw_score[empty_rows]
        return score

    @staticmethod
    def _gather(score, indices):
        return score.gather(1, indices)

    def retrieve(self, x, batch_index=None, mode='train'):
        if not self.is_ready:
            fallback = torch.zeros(x.size(0), self.pred_len, x.size(2), device=x.device, dtype=x.dtype)
            return fallback, self._empty_diagnostics(x)

        query_phase, query_amplitude, query_time = self._encode(x)
        phase_sim = torch.matmul(query_phase, self.memory_phase.transpose(0, 1))
        amplitude_sim = torch.matmul(query_amplitude, self.memory_amplitude.transpose(0, 1))
        time_sim = torch.matmul(query_time, self.memory_time.transpose(0, 1))

        phase_score = (
            self.time_weight * time_sim
            + self.phase_weight * phase_sim
            + self.amplitude_weight * amplitude_sim
        )
        retrieval_mode = (self.retrieval_mode or 'time_phase_rerank').lower()

        if retrieval_mode == 'phase':
            score = self._mask_training_neighbors(phase_score.clone(), phase_score, batch_index, mode)
            m = min(self.rerank_top_m, score.size(1))
            top_similarity, top_indices = torch.topk(score, k=m, dim=1)
            primary_top_similarity = top_similarity
            primary_top_indices = top_indices
            final_time = self._gather(time_sim, top_indices)
            final_phase = self._gather(phase_sim, top_indices)
            final_amplitude = self._gather(amplitude_sim, top_indices)
        else:
            primary_score = self._mask_training_neighbors(time_sim.clone(), time_sim, batch_index, mode)
            k = min(self.primary_top_k, primary_score.size(1))
            primary_top_similarity, primary_top_indices = torch.topk(primary_score, k=k, dim=1)

            candidate_time = self._gather(time_sim, primary_top_indices)
            candidate_phase = self._gather(phase_sim, primary_top_indices)
            candidate_amplitude = self._gather(amplitude_sim, primary_top_indices)
            if retrieval_mode == 'time':
                rerank_score = candidate_time
            else:
                rerank_score = (
                    self.time_weight * candidate_time
                    + self.phase_weight * candidate_phase
                    + self.amplitude_weight * candidate_amplitude
                )

            m = min(self.rerank_top_m, rerank_score.size(1))
            top_similarity, final_pos = torch.topk(rerank_score, k=m, dim=1)
            top_indices = primary_top_indices.gather(1, final_pos)
            final_time = candidate_time.gather(1, final_pos)
            final_phase = candidate_phase.gather(1, final_pos)
            final_amplitude = candidate_amplitude.gather(1, final_pos)

        weights = F.softmax(top_similarity / max(self.temperature, 1e-6), dim=1)
        candidates = self.memory_delta[top_indices]
        retrieved_delta = torch.einsum('bk,bkpc->bpc', weights, candidates)
        diagnostics = {
            'top_indices': top_indices,
            'primary_top_indices': primary_top_indices,
            'top_similarity': top_similarity,
            'primary_top_similarity': primary_top_similarity,
            'weights': weights,
            'phase_similarity': final_phase.mean(dim=1),
            'time_similarity': final_time.mean(dim=1),
            'amplitude_similarity': final_amplitude.mean(dim=1),
            'rerank_similarity': top_similarity.mean(dim=1),
            'primary_similarity': primary_top_similarity.mean(dim=1),
        }
        return retrieved_delta, diagnostics


class RetrievalActionController(nn.Module):
    """
    RPO + RFRL controller.

    RPO predicts whether the retrieval correction is likely useful. RFRL consumes
    that score as a detached risk feature and selects a continuous action alpha
    from a small discrete action support.
    """

    def __init__(
        self,
        channels,
        hidden_size=64,
        alpha_bins=(0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40),
        no_retrieval_bias=2.0,
    ):
        super().__init__()
        self.channels = channels
        alpha_bins = torch.as_tensor(alpha_bins, dtype=torch.float32).clamp(0.0, 1.0)
        if alpha_bins.numel() == 0:
            alpha_bins = torch.tensor([0.0, 0.10, 0.20], dtype=torch.float32)
        alpha_bins = torch.unique(alpha_bins, sorted=True)
        if not torch.isclose(alpha_bins[0], torch.tensor(0.0, dtype=alpha_bins.dtype)):
            alpha_bins = torch.cat([torch.zeros(1, dtype=alpha_bins.dtype), alpha_bins]).unique(sorted=True)
        self.register_buffer('alpha_bins', alpha_bins, persistent=False)

        self.feature_net = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.preference_head = nn.Linear(hidden_size, 1)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, alpha_bins.numel()),
        )
        with torch.no_grad():
            final_layer = self.policy_net[-1]
            final_layer.bias.zero_()
            final_layer.bias[0] = float(no_retrieval_bias)

    @staticmethod
    def _diag_mean(diagnostics, name, x):
        value = diagnostics.get(name)
        if value is None:
            return torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        if value.ndim == 1:
            return value.to(device=x.device, dtype=x.dtype)
        return value.reshape(value.size(0), -1).mean(dim=1).to(device=x.device, dtype=x.dtype)

    def _features(self, x, baseline, retrieval_forecast, diagnostics):
        delta = retrieval_forecast - baseline
        top_similarity = diagnostics.get('top_similarity')
        weights = diagnostics.get('weights')
        if top_similarity is None or weights is None:
            sim_mean = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            sim_gap = sim_mean
            entropy = sim_mean
        else:
            top_similarity = top_similarity.to(device=x.device, dtype=x.dtype)
            weights = weights.to(device=x.device, dtype=x.dtype)
            sim_mean = top_similarity.mean(dim=1)
            sim_gap = top_similarity[:, 0] - top_similarity[:, -1]
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=1)

        return torch.stack([
            x.mean(dim=(1, 2)),
            x.std(dim=(1, 2), unbiased=False),
            delta.abs().mean(dim=(1, 2)),
            sim_mean,
            sim_gap,
            entropy,
            self._diag_mean(diagnostics, 'time_similarity', x),
            self._diag_mean(diagnostics, 'phase_similarity', x),
            self._diag_mean(diagnostics, 'amplitude_similarity', x),
        ], dim=1)

    def forward(self, x, baseline, retrieval_forecast, diagnostics):
        features = self._features(x, baseline, retrieval_forecast, diagnostics)
        hidden = self.feature_net(features)
        preference_score = torch.sigmoid(self.preference_head(hidden)).view(-1, 1, 1)
        policy_context = torch.cat([hidden, preference_score.detach().view(-1, 1)], dim=1)
        action_logits = self.policy_net(policy_context)
        action_probabilities = F.softmax(action_logits, dim=1)
        alpha_bins = self.alpha_bins.to(device=x.device, dtype=x.dtype)
        action_alpha = torch.sum(action_probabilities * alpha_bins.view(1, -1), dim=1)
        action_index = torch.argmax(action_probabilities, dim=1)
        return (
            preference_score,
            action_alpha.view(-1, 1, 1),
            action_probabilities,
            action_logits,
            action_index,
        )


class AdaptiveFusion(nn.Module):
    def forward(self, baseline, retrieval_forecast, action_alpha):
        fusion_weight = action_alpha.clamp(0.0, 1.0)
        final = baseline + fusion_weight * (retrieval_forecast - baseline)
        return final, fusion_weight


PhaseAwareRetrievalBank = TemporalPhaseRetrievalBank
