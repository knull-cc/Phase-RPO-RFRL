TSF-Lib

一个精简的长序列时间序列预测（long-term forecasting）基础框架。

这个仓库的目标不是做一个复杂的大而全框架，而是提供一个干净、统一、容易修改的 base。你可以把新模型直接放进 models/ 运行，也可以让 AI 参考 example/ 中的示例文件，在这个 base 上快速实现和测试新的 idea。

核心目的

* 提供一个最小可用的 TSF 训练/验证/测试框架。
* 保留统一的数据加载、参数入口、训练流程和指标计算。
* 支持把新模型文件直接放入 models/ 后运行。
* 提供 example/ 作为参考，让 AI 或开发者可以基于已有示例改出新方法。
* 尽量减少无关工程复杂度，方便快速实验。

特性

* 统一入口：通过 run.py 运行所有长序列预测实验。
* 模型即插即用：在 models/ 下新增 X.py，文件中包含 class Model，即可通过 --model X 调用。
* 自动模型加载：无需手动维护模型注册表。
* 精简任务范围：只保留 long-term forecasting，便于集中修改和实验。
* 当前默认模型：`PhaseRPO_RFRL_MLP`，用于实现和测试 Phase-RPO-RFRL。
* 提供 example：example/ 中的代码用于指导 AI 或开发者如何基于当前 base 修改模型。

Phase-RPO-RFRL

当前实现只保留 `PhaseRPO_RFRL_MLP`。它不使用周期查询机制；host 的先验来自输入窗口自身的频域摘要，检索插件采用时域相似度主召回，并把相位/幅值作为候选重排信号。
当前主线是 risk-aware retrieval action policy：检索不是固定后处理，而是一个带 no-retrieval 动作的样本级策略决策。

当前流程为：

RevIN Residual MLP Host -> time-domain retrieval top-K -> phase/amplitude rerank top-M -> residual adapter -> counterfactual utility estimation -> RPO risk score -> RFRL action policy -> Adaptive Fusion

模型会在训练开始前用 train split 构建时间安全的 retrieval bank。检索分支返回候选 future 相对候选历史末端的 residual/delta，而不是把候选绝对 future 当作完整预测。
RPO 不直接乘到预测上，而是学习当前检索修正是否有收益的 risk/preference score；RFRL 使用该 score 作为控制特征，在 `0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4` 等动作强度中学习选择。`0` 是显式 no-retrieval 动作，不是普通 gate 的边界值。
最终形式为：

final = baseline + action_alpha * retrieval_correction

训练时会用真实未来构造 oracle action label：系统枚举每个候选 `action_alpha` 的 counterfactual forecast，选择误差最低且超过 gain margin 的动作作为 RFRL 监督信号。adapter 的辅助训练也只作用在 oracle 认为 retrieval 有用的样本上，学习 `oracle_alpha * retrieval_correction` 的有效残差，避免坏检索样本强行训练 correction。
为保证 host model 仍是第一预测路径，训练中额外保留 host baseline anchor loss；retrieval correction 在归一化空间中使用 tanh clip 和正则约束，避免 adapter 产生大尺度有害残差。RPO 使用 class-balanced preference loss，避免 oracle-abstain 样本被多数 accept 样本淹没。

运行默认模型：

python run.py ... --model PhaseRPO_RFRL_MLP

常用可调参数包括 `--mlp_hidden_dim`、`--mlp_dropout`、`--mlp_use_revin`、`--mlp_spectral_bins`、`--retrieval_mode`、`--retrieval_top_k`、`--retrieval_top_m`、`--retrieval_time_key_len`、`--phase_max_freqs`、`--phase_similarity_weight`、`--amplitude_similarity_weight`、`--rfrl_alpha_bins`、`--rfrl_no_retrieval_bias`、`--rfrl_regret_loss_weight`、`--rpo_gain_margin`、`--rfrl_gain_margin`、`--rpo_loss_weight`、`--rfrl_loss_weight`、`--host_anchor_loss_weight`、`--retrieval_adapter_loss_weight`、`--retrieval_correction_reg_weight`、`--retrieval_correction_clip`、`--retrieval_residual_init` 和 `--retrieval_cost`。
测试后可以用 `tools/analyze_retrieval_diagnostics.py <result_dir>` 查看 baseline、raw retrieval、adapter correction、oracle alpha、model action alpha、policy regret、false accept、false reject、abstention accuracy 和相似度相关性，判断瓶颈在检索、adapter、RPO 风险判断还是 RFRL 动作策略。

RAFT-RPO

当前还提供 `RAFT_RPO_MLP`，用于验证“先退化到 RAFT 检索，再只加 RPO”的路线。它参考 `example/RAFT - 原始版本/` 的核心检索方式：训练开始前用 train split 构建 lookback/future 记忆库，对输入窗口做多周期平均分解，在每个周期粒度上用相关性检索 top-m 历史窗口。和原始 RAFT 一样，代码仍保留 soft top-m 聚合后的 always-retrieve forecast 作为 reference；不同的是，新的 RAFT-RPO 还会保留每一个 top-m 历史候选 `r` 的 future residual，不再在检索阶段丢掉单候选身份。

在新的 RAFT-RPO 中，RPO 不是 RFRL，也不是相位 gate。RAFT 负责 recall，RPO 只在 RAFT 召回的 `G * topm` 个候选内部学习 rerank/gate：

* `reference`：默认是原始 RAFT always-retrieve forecast，也可以通过 `--rpo_utility_reference baseline` 改成 host baseline。
* `candidate_r`：某个 RAFT period/rank 的历史候选 future residual，替换该 period 的 soft 聚合残差后，经 RAFT fusion head 生成 per-candidate forecast。

训练时用真实未来值计算每个候选的 forecast utility，默认以 MAE 为偏好目标：

utility(q, r) = MAE(reference(q), y) - MAE(RAFT(q, r), y)

然后在同一个 query 的 top-M 内构造 `r+ / r-` 偏好对，满足 `MAE(r-) - MAE(r+) > --rpo_pair_margin` 的候选对才进入 reference-anchored RPO/DPO loss：

log pi_ref(r | q) = log softmax(sim_RAFT(q, r) / tau_ref)
log pi_theta(r | q) = log softmax((sim_RAFT(q, r) + alpha * s_theta(q, r)) / tau)
L_RPO = -log sigmoid(beta * [(log pi_theta(r+) - log pi_theta(r-)) - (log pi_ref(r+) - log pi_ref(r-))])

最终 forecast 不是直接替换 RAFT，而是先得到 reranked candidate mixture，再由 predicted utility gate 决定是否回退：

reranked = sum_r pi_theta(r | q) * RAFT(q, r)
final = p_accept * reranked + (1 - p_accept) * reference

如果 `--rpo_hard_eval` 打开，测试时会使用硬决策：`predicted_utility <= --rpo_gate_epsilon` 则回退到 reference，否则选择 `argmax pi_theta` 的候选。

运行示例：

python run.py \
  --task_name long_term_forecast --is_training 1 \
  --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --model_id ETTh1_96_96 --model RAFT_RPO_MLP \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --n_period 3 --topm 20 --des RAFT_RPO

测试后运行：

python tools/analyze_retrieval_diagnostics.py results/<result_dir>

RAFT-RPO 新日志会额外输出：

* reference/候选上界：`raft_always_retrieval_mse/mae`、`rpo_reranked_mse/mae`、`oracle_topm_rerank_mse/mae`。
* RPO 是否有用：`rpo_reranked_gain_vs_reference_mae`、`final_gain_vs_reference_mae`、`oracle_topm_gain_vs_reference_mae`、`rpo_gain_capture_vs_oracle_topm_mae`。
* rerank 质量：`pi_ref_entropy`、`pi_theta_entropy`、`kl(pi_theta || pi_ref)`、`RPO top1 equals oracle best candidate`、`RPO top1 differs from RAFT-sim top1`。
* gate 质量：`rpo_predicted_utility`、`rpo_accept_probability`、`false accept rate`、`false reject rate`、oracle-use/oracle-fallback 分片 gain。
* 检索瓶颈定位：period-level/rank-level candidate utility，以及 similarity/score 与 best candidate gain 的相关性。

判断逻辑：

* `oracle_topm_gain_vs_reference_mae <= 0`：RAFT top-M 里没有足够好的候选，RPO 学不到有效 rerank。
* `oracle_topm_gain_vs_reference_mae > 0` 但 `rpo_reranked_gain_vs_reference_mae <= 0`：top-M 有上界，RPO scorer/pairwise objective 没学到。
* `rpo_reranked_gain_vs_reference_mae > 0` 但 `final_gain_vs_reference_mae <= 0`：rerank 有收益，但 utility gate 错收/错拒。
* `final_gain_vs_reference_mae > 0`：RPO 相对 RAFT/reference 起到作用。

目录结构

TSF-Lib/
  run.py                          # 统一运行入口
  run_main.sh                     # 一键运行多个数据集
  exp/
    exp_basic.py                  # 设备设置与模型自动加载
    exp_long_term_forecasting.py  # 训练、验证、测试流程
  models/                         # 模型文件，新模型放这里即可
  layers/                         # 模型依赖的模块
  data_provider/                  # 数据加载逻辑
  utils/                          # 指标、工具函数、时间特征等
  scripts/                        # 各数据集运行脚本
  example/                        # 给 AI / 开发者参考的修改示例

安装

pip install -r requirements.txt

数据放置

数据不包含在仓库中，默认放在 ./dataset/ 下：

dataset/
  ETT-small/   ETTh1.csv  ETTh2.csv  ETTm1.csv  ETTm2.csv
  electricity/ electricity.csv
  traffic/     traffic.csv
  weather/     weather.csv
  Solar/       solar_AL.txt
  PEMS/        PEMS03.npz  PEMS04.npz  PEMS07.npz  PEMS08.npz

数据集来源与 TSLib 保持一致。

使用方式

运行单个数据集：

bash scripts/ETTh1.sh

修改训练参数：

bash scripts/ETTh1.sh --train_epochs 20 --learning_rate 0.001

一键运行多个数据集(推荐)：

bash run_main.sh （最推荐）

也可以直接调用 run.py：

python run.py \
  --task_name long_term_forecast --is_training 1 \
  --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --model_id ETTh1_96_96 --model PhaseRPO_RFRL_MLP \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7

如何添加新模型

在 models/ 下新建一个模型文件，例如：

models/MyIdea.py

文件中实现统一的 Model 类：

import torch.nn as nn
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        # 从 configs 中读取需要的参数
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, seq_len, enc_in]
        # 返回: [B, pred_len, enc_in]
        ...

然后直接运行：

python run.py \
  --model MyIdea \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7

或者：

bash scripts/ETTh1.sh --model MyIdea

如果模型需要新的超参数，只需要在 run.py 的 build_parser() 中添加对应参数，然后在模型中通过 configs.xxx 读取。

给 AI 使用的方式

本仓库适合作为 AI 修改时间序列预测模型的基础代码。

推荐使用方式：

1. 让 AI 先阅读整体框架，尤其是：
    * run.py
    * exp/exp_long_term_forecasting.py
    * models/
    * example/
2. 告诉 AI 新 idea 的目标。
3. 要求 AI 参考 example/ 中的写法，在 models/ 下新增或修改模型文件。
4. 使用现有脚本直接测试新模型：

bash scripts/ETTh1.sh --model NewModel

example/ 的作用是给 AI 一个明确的代码风格和修改参考，避免 AI 直接改乱主框架。

结果保存

实验结果会追加写入：

result_long_term_forecast.txt

设计原则

这个库只保留最核心的实验流程：

* 数据读取
* 模型加载
* 训练
* 验证
* 测试
* 指标记录

其他复杂功能尽量不加入。这样做的目的是让代码更容易被阅读、修改和复用，尤其适合快速验证新的 TSF idea。
