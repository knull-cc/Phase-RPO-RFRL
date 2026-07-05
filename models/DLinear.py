import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.PhaseRPO_RFRL import PhaseAwareRetrievalBank, RetrievalPreferenceController, AdaptiveFusion


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.features = getattr(configs, 'features', 'M')
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.task_name == 'classification':
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

        self.use_phase_rpo_rfrl = getattr(configs, 'use_phase_rpo_rfrl', True)
        self.rpo_loss_weight = getattr(configs, 'rpo_loss_weight', 0.1)
        self.rfrl_loss_weight = getattr(configs, 'rfrl_loss_weight', 0.05)
        self.retrieval_cost = getattr(configs, 'retrieval_cost', 0.01)
        self.latest_aux_loss = None
        self.latest_diagnostics = {}

        if self.use_phase_rpo_rfrl and self.task_name in ['long_term_forecast', 'short_term_forecast']:
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
        if not self.use_phase_rpo_rfrl or not hasattr(self, 'phase_retrieval'):
            return
        device = device or next(self.parameters()).device
        print('Building Phase-RPO-RFRL retrieval bank from train split...')
        self.phase_retrieval.build(train_data, device=device)

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def forecast_with_retrieval(self, x_enc, batch_index=None, mode='train', target_y=None):
        baseline = self.forecast(x_enc)
        if not self.use_phase_rpo_rfrl or not hasattr(self, 'phase_retrieval'):
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

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                batch_index=None, mode='train', target_y=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast_with_retrieval(
                x_enc, batch_index=batch_index, mode=mode, target_y=target_y)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
