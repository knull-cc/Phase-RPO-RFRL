import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _raft_periods(n_period, pred_len):
    base_periods = [16, 8, 4, 2, 1]
    periods = sorted(base_periods[-max(1, int(n_period)):], reverse=True)
    periods = [period for period in periods if pred_len % period == 0]
    return periods or [1]


class RAFTRetrievalBank(nn.Module):
    """
    RAFT-style multi-granularity retrieval bank.

    It stores train split lookback/future pairs, decomposes them with the same
    periodic averaging used by RAFT, and returns top-m soft retrieved future
    residuals for each period.  The individual top-m candidates are preserved
    in diagnostics so RPO can learn forecast-utility-aware reranking instead of
    scoring only the already-aggregated RAFT branch.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        topm=20,
        temperature=0.1,
        max_bank_size=4096,
        exclusion_radius=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.period_num = _raft_periods(n_period, pred_len)
        self.n_period = len(self.period_num)
        self.topm = max(1, int(topm))
        self.temperature = float(temperature)
        self.max_bank_size = int(max_bank_size) if max_bank_size else 0
        self.exclusion_radius = exclusion_radius or (seq_len + pred_len)

        self.register_buffer('memory_x_mg', torch.empty(0), persistent=False)
        self.register_buffer('memory_y_mg', torch.empty(0), persistent=False)
        self.register_buffer('memory_indices', torch.empty(0, dtype=torch.long), persistent=False)

    @property
    def is_ready(self):
        return self.memory_x_mg.numel() > 0 and self.memory_y_mg.numel() > 0

    def _periodic_mean(self, data, period):
        length = data.size(1)
        remainder = length % period
        if remainder:
            pad_len = period - remainder
            pad = data[:, -1:, :].repeat(1, pad_len, 1)
            data = torch.cat([data, pad], dim=1)
        grouped = data.reshape(data.size(0), data.size(1) // period, period, data.size(2))
        mean = grouped.mean(dim=2, keepdim=True).repeat(1, 1, period, 1)
        return mean.reshape(data.size(0), data.size(1), data.size(2))[:, :length, :]

    def decompose_mg(self, data, remove_offset=True):
        pieces = []
        offsets = []
        for period in self.period_num:
            cur = self._periodic_mean(data, period)
            if remove_offset:
                offset = cur[:, -1:, :]
                cur = cur - offset
                offsets.append(offset)
            pieces.append(cur)
        stacked = torch.stack(pieces, dim=0)
        if not remove_offset:
            return stacked, None
        return stacked, torch.stack(offsets, dim=0)

    def build(self, train_dataset, device):
        xs, ys, indices = [], [], []
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
            ys.append(seq_y)
            indices.append(int(index))

        if not xs:
            return

        train_x = torch.stack(xs, dim=0).to(device)
        train_y = torch.stack(ys, dim=0).to(device)
        self.memory_x_mg, _ = self.decompose_mg(train_x, remove_offset=True)
        self.memory_y_mg, _ = self.decompose_mg(train_y, remove_offset=True)
        self.memory_indices = torch.tensor(indices, dtype=torch.long, device=device)

    def _periodic_batch_corr(self, data_all, key, in_bsz=512):
        _, _, features = key.shape
        _, train_len, _ = data_all.shape
        bx = key - key.mean(dim=2, keepdim=True)
        bx = F.normalize(bx, dim=2)

        sims = []
        iters = math.ceil(train_len / in_bsz)
        for i in range(iters):
            start = i * in_bsz
            end = min((i + 1) * in_bsz, train_len)
            ax = data_all[:, start:end].to(key.device)
            ax = ax - ax.mean(dim=2, keepdim=True)
            ax = F.normalize(ax, dim=2)
            sims.append(torch.bmm(bx, ax.transpose(-1, -2)))
        return torch.cat(sims, dim=2)

    def _mask_training_neighbors(self, score, raw_score, batch_index, mode):
        if mode != 'train' or batch_index is None or self.memory_indices.numel() == 0:
            return score
        index = batch_index.to(score.device).long().view(1, -1, 1)
        memory_index = self.memory_indices.view(1, 1, -1)
        mask = (memory_index - index).abs() <= self.exclusion_radius
        score = score.masked_fill(mask, float('-inf'))
        empty_rows = torch.isinf(score).all(dim=2, keepdim=True)
        if empty_rows.any():
            score = torch.where(empty_rows, raw_score, score)
        return score

    def _empty_diagnostics(self, x):
        bsz = x.size(0)
        zeros = torch.zeros(bsz, self.n_period, 1, device=x.device, dtype=x.dtype)
        return {
            'top_indices': torch.zeros(bsz, self.n_period, 1, device=x.device, dtype=torch.long),
            'primary_top_indices': torch.zeros(bsz, self.n_period, 1, device=x.device, dtype=torch.long),
            'top_similarity': zeros,
            'primary_top_similarity': zeros,
            'weights': torch.ones_like(zeros),
            'period_similarity': zeros.squeeze(-1),
            'candidate_y_mg': torch.zeros(
                bsz, self.n_period, 1, self.pred_len, self.channels,
                device=x.device, dtype=x.dtype,
            ),
        }

    def retrieve(self, x, batch_index=None, mode='train'):
        if not self.is_ready:
            fallback = torch.zeros(
                self.n_period, x.size(0), self.pred_len, x.size(2),
                device=x.device, dtype=x.dtype,
            )
            return fallback, self._empty_diagnostics(x)

        x_mg, _ = self.decompose_mg(x, remove_offset=True)
        sim = self._periodic_batch_corr(
            self.memory_x_mg.flatten(start_dim=2),
            x_mg.flatten(start_dim=2),
        )
        raw_sim = sim
        sim = self._mask_training_neighbors(sim.clone(), raw_sim, batch_index, mode)

        g_num, bsz, train_len = sim.shape
        topm = min(self.topm, train_len)
        flat_sim = sim.reshape(g_num * bsz, train_len)
        top_similarity, top_indices = torch.topk(flat_sim, topm, dim=1)
        top_similarity = top_similarity.reshape(g_num, bsz, topm)
        top_indices = top_indices.reshape(g_num, bsz, topm)

        weights = F.softmax(top_similarity / max(self.temperature, 1e-6), dim=2)
        retrieved = []
        candidate_y_mg = []
        for period_id in range(g_num):
            indices = top_indices[period_id].reshape(-1)
            candidates = self.memory_y_mg[period_id].index_select(0, indices)
            candidates = candidates.reshape(bsz, topm, self.pred_len, self.channels)
            candidate_y_mg.append(candidates)
            cur = torch.einsum('bm,bmpc->bpc', weights[period_id], candidates)
            retrieved.append(cur)

        period_retrieval = torch.stack(retrieved, dim=0)
        diagnostics = {
            'top_indices': top_indices.permute(1, 0, 2).contiguous(),
            'primary_top_indices': top_indices.permute(1, 0, 2).contiguous(),
            'top_similarity': top_similarity.permute(1, 0, 2).contiguous(),
            'primary_top_similarity': top_similarity.permute(1, 0, 2).contiguous(),
            'weights': weights.permute(1, 0, 2).contiguous(),
            'period_similarity': top_similarity.mean(dim=2).transpose(0, 1).contiguous(),
            'candidate_y_mg': torch.stack(candidate_y_mg, dim=1).contiguous(),
        }
        return period_retrieval, diagnostics


class RPOCandidateScorer(nn.Module):
    """
    Forecast-utility-aware scorer over the RAFT top-m retrieval candidates.

    RAFT similarity remains the reference policy.  RPO only learns an additive
    score inside the recalled candidate set:

        pi_ref    = softmax(sim_RAFT / tau_ref)
        pi_theta  = softmax((sim_RAFT + alpha * s_theta) / tau)

    A separate utility head predicts whether the reranked candidate mixture
    should be used or whether the model should fall back to the RAFT reference
    forecast.
    """

    def __init__(
        self,
        hidden_size=64,
        reference_temperature=0.1,
        policy_temperature=1.0,
        score_alpha=1.0,
        gate_temperature=0.05,
        gate_epsilon=0.0,
    ):
        super().__init__()
        self.reference_temperature = max(float(reference_temperature), 1e-6)
        self.policy_temperature = max(float(policy_temperature), 1e-6)
        self.score_alpha = float(score_alpha)
        self.gate_temperature = max(float(gate_temperature), 1e-6)
        self.gate_epsilon = float(gate_epsilon)

        self.scorer = nn.Sequential(
            nn.Linear(12, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.utility_head = nn.Sequential(
            nn.Linear(10, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def _candidate_features(
        self,
        x,
        reference,
        candidates,
        similarity,
        reference_probability,
        candidate_period,
        candidate_rank,
    ):
        bsz, candidate_count = candidates.shape[:2]
        x_mean = x.mean(dim=(1, 2), keepdim=True).view(bsz, 1).repeat(1, candidate_count)
        x_std = x.std(dim=(1, 2), unbiased=False, keepdim=True).view(bsz, 1).repeat(1, candidate_count)
        if x.size(1) > 1:
            slope = (x[:, -1, :] - x[:, 0, :]).abs().mean(dim=1, keepdim=True).repeat(1, candidate_count)
        else:
            slope = torch.zeros_like(x_mean)

        reference_std = reference.std(dim=(1, 2), unbiased=False, keepdim=True).view(bsz, 1)
        reference_std = reference_std.repeat(1, candidate_count)
        candidate_std = candidates.std(dim=(2, 3), unbiased=False)

        delta = candidates - reference.unsqueeze(1)
        delta_abs = delta.abs().mean(dim=(2, 3))
        delta_std = delta.std(dim=(2, 3), unbiased=False)
        delta_mean = delta.mean(dim=(2, 3))

        max_period = candidate_period.max().clamp_min(1.0)
        period_code = candidate_period / max_period
        max_rank = candidate_rank.max().clamp_min(1.0)
        rank_code = candidate_rank / max_rank

        return torch.stack([
            x_mean,
            x_std,
            slope,
            reference_std,
            candidate_std,
            delta_abs,
            delta_std,
            delta_mean,
            similarity,
            reference_probability,
            rank_code,
            period_code,
        ], dim=2)

    def forward(self, x, reference, candidates, similarity, candidate_period, candidate_rank):
        similarity = similarity.to(dtype=x.dtype, device=x.device)
        candidate_period = candidate_period.to(dtype=x.dtype, device=x.device)
        candidate_rank = candidate_rank.to(dtype=x.dtype, device=x.device)

        reference_log_probability = F.log_softmax(
            similarity / self.reference_temperature,
            dim=1,
        )
        reference_probability = reference_log_probability.exp()

        features = self._candidate_features(
            x,
            reference,
            candidates,
            similarity,
            reference_probability,
            candidate_period,
            candidate_rank,
        )
        scores = self.scorer(features).squeeze(-1)
        policy_logits = (similarity + self.score_alpha * scores) / self.policy_temperature
        policy_log_probability = F.log_softmax(policy_logits, dim=1)
        policy_probability = policy_log_probability.exp()

        delta = candidates - reference.unsqueeze(1)
        delta_abs = delta.abs().mean(dim=(2, 3))
        policy_entropy = -(
            policy_probability * (policy_log_probability)
        ).sum(dim=1)
        reference_entropy = -(
            reference_probability * reference_log_probability
        ).sum(dim=1)

        utility_features = torch.stack([
            scores.max(dim=1).values,
            scores.mean(dim=1),
            scores.std(dim=1, unbiased=False),
            similarity.max(dim=1).values,
            similarity.mean(dim=1),
            reference_probability.max(dim=1).values,
            policy_entropy,
            reference_entropy,
            delta_abs.max(dim=1).values,
            delta_abs.mean(dim=1),
        ], dim=1)
        predicted_utility = self.utility_head(utility_features).squeeze(-1)
        gate_logit = (predicted_utility - self.gate_epsilon) / self.gate_temperature
        accept_probability = torch.sigmoid(gate_logit).view(-1, 1, 1)
        action_index = torch.argmax(policy_probability, dim=1)

        return {
            'scores': scores,
            'policy_logits': policy_logits,
            'policy_probability': policy_probability,
            'policy_log_probability': policy_log_probability,
            'reference_probability': reference_probability,
            'reference_log_probability': reference_log_probability,
            'predicted_utility': predicted_utility,
            'gate_logit': gate_logit,
            'accept_probability': accept_probability,
            'candidate_index': action_index,
        }
