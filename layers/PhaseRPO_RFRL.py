import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseAwareRetrievalBank(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        top_k=8,
        max_freqs=16,
        temperature=0.07,
        phase_weight=0.55,
        amplitude_weight=0.25,
        time_weight=0.20,
        max_bank_size=4096,
        exclusion_radius=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.top_k = top_k
        self.max_freqs = max_freqs
        self.temperature = temperature
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

        self.memory_x = torch.stack(xs, dim=0).to(device)
        self.memory_delta = torch.stack(deltas, dim=0).to(device)
        self.memory_indices = torch.tensor(indices, dtype=torch.long, device=device)
        self.memory_phase, self.memory_amplitude, self.memory_time = self._encode(self.memory_x)

    def _encode(self, x):
        x_centered = x - x.mean(dim=1, keepdim=True)
        spectrum = torch.fft.rfft(x_centered, dim=1)
        limit = min(self.max_freqs, max(spectrum.size(1) - 1, 0))
        if limit == 0:
            phase = torch.zeros(x.size(0), 2 * x.size(2), device=x.device, dtype=x.dtype)
            amplitude = torch.zeros(x.size(0), x.size(2), device=x.device, dtype=x.dtype)
        else:
            spectrum = spectrum[:, 1:limit + 1, :]
            amplitude_raw = spectrum.abs()
            phase_cos = spectrum.real / (amplitude_raw + 1e-6)
            phase_sin = spectrum.imag / (amplitude_raw + 1e-6)
            amp_weight = amplitude_raw / (amplitude_raw.mean(dim=1, keepdim=True) + 1e-6)
            phase = torch.cat([phase_cos * amp_weight, phase_sin * amp_weight], dim=1).flatten(start_dim=1)
            amplitude = amplitude_raw.flatten(start_dim=1)

        time_key = x_centered.flatten(start_dim=1)
        phase = F.normalize(phase, dim=1)
        amplitude = F.normalize(amplitude, dim=1)
        time_key = F.normalize(time_key, dim=1)
        return phase, amplitude, time_key

    def retrieve(self, x, batch_index=None, mode='train'):
        if not self.is_ready:
            fallback = torch.zeros(x.size(0), self.pred_len, x.size(2), device=x.device, dtype=x.dtype)
            diagnostics = {
                'top_indices': None,
                'top_similarity': torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype),
                'weights': None,
                'phase_similarity': torch.zeros(x.size(0), device=x.device, dtype=x.dtype),
            }
            return fallback, diagnostics

        query_phase, query_amplitude, query_time = self._encode(x)
        phase_sim = torch.matmul(query_phase, self.memory_phase.transpose(0, 1))
        amplitude_sim = torch.matmul(query_amplitude, self.memory_amplitude.transpose(0, 1))
        time_sim = torch.matmul(query_time, self.memory_time.transpose(0, 1))
        sim = (
            self.phase_weight * phase_sim
            + self.amplitude_weight * amplitude_sim
            + self.time_weight * time_sim
        )
        raw_sim = sim

        if mode == 'train' and batch_index is not None:
            index = batch_index.to(x.device).long().view(-1, 1)
            distance = (self.memory_indices.view(1, -1) - index).abs()
            sim = sim.masked_fill(distance <= self.exclusion_radius, float('-inf'))
            empty_rows = torch.isinf(sim).all(dim=1)
            if empty_rows.any():
                sim = sim.clone()
                sim[empty_rows] = raw_sim[empty_rows]

        k = min(self.top_k, sim.size(1))
        top_similarity, top_indices = torch.topk(sim, k=k, dim=1)
        weights = F.softmax(top_similarity / self.temperature, dim=1)
        candidates = self.memory_delta[top_indices]
        retrieved_delta = torch.einsum('bk,bkpc->bpc', weights, candidates)
        diagnostics = {
            'top_indices': top_indices,
            'top_similarity': top_similarity,
            'weights': weights,
            'phase_similarity': phase_sim.gather(1, top_indices).mean(dim=1),
        }
        return retrieved_delta, diagnostics


class RetrievalPreferenceController(nn.Module):
    def __init__(self, channels, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.preference_head = nn.Linear(hidden_size, 1)
        self.policy_head = nn.Linear(hidden_size, 1)
        self.channel_gate = nn.Linear(hidden_size, channels)

    def forward(self, x, baseline, retrieval_forecast, diagnostics):
        delta = retrieval_forecast - baseline
        top_similarity = diagnostics['top_similarity']
        weights = diagnostics['weights']
        if top_similarity is None or weights is None:
            sim_mean = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            sim_gap = sim_mean
            entropy = sim_mean
        else:
            sim_mean = top_similarity.mean(dim=1)
            sim_gap = top_similarity[:, 0] - top_similarity[:, -1]
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=1)

        features = torch.stack([
            x.mean(dim=(1, 2)),
            x.std(dim=(1, 2)),
            delta.abs().mean(dim=(1, 2)),
            sim_mean,
            sim_gap,
            entropy,
        ], dim=1)
        hidden = self.net(features)
        preference_score = torch.sigmoid(self.preference_head(hidden)).view(-1, 1, 1)
        policy_strength = torch.sigmoid(self.policy_head(hidden)).view(-1, 1, 1)
        channel_gate = torch.sigmoid(self.channel_gate(hidden)).unsqueeze(1)
        guidance = policy_strength * channel_gate
        return preference_score, guidance


class AdaptiveFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.residual_gate = nn.Parameter(torch.zeros(1, 1, channels))

    def forward(self, baseline, retrieval_forecast, guidance):
        gate = torch.clamp(guidance * torch.sigmoid(self.residual_gate), 0.0, 1.0)
        final = baseline + gate * (retrieval_forecast - baseline)
        return final, gate
