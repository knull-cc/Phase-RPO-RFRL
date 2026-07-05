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
* 内置基础模型：包含 DLinear、iTransformer、PatchTST，可作为 baseline 或改写模板。
* 提供 example：example/ 中的代码用于指导 AI 或开发者如何基于当前 base 修改模型。

Phase-RPO-RFRL

当前实现保留 DLinear 作为纯 baseline，但 Phase-RPO-RFRL 已不必绑定 DLinear。当前提供两种 Phase-RPO-RFRL 版本：

* `PhaseRPO_RFRL_MLP`：推荐的 backbone-style 默认版本，使用两层 MLP host。
* `PhaseRPO_RFRL_DLinear`：保留的 DLinear host 版本，用于和纯 DLinear baseline 对照。

这两类模型都会在训练开始前用 train split 构建时间安全的 retrieval bank，并按：

Host Model -> Phase-aware Retrieval -> RPO preference -> RFRL Controller -> Adaptive Fusion

生成最终预测。运行纯 DLinear baseline：

python run.py ... --model DLinear

运行推荐的 MLP backbone 版本：

python run.py ... --model PhaseRPO_RFRL_MLP

运行 DLinear host 版本：

python run.py ... --model PhaseRPO_RFRL_DLinear

常用可调参数包括 `--mlp_hidden_dim`、`--mlp_dropout`、`--phase_top_k`、`--phase_max_freqs`、`--phase_max_bank_size`、`--rpo_loss_weight`、`--rfrl_loss_weight` 和 `--retrieval_cost`。

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

切换模型：

bash scripts/ETTh1.sh --model iTransformer
bash scripts/ETTh1.sh --model PatchTST

修改训练参数：

bash scripts/ETTh1.sh --model iTransformer --train_epochs 5 --learning_rate 0.0005

一键运行多个数据集(推荐)：

bash run_main.sh （最推荐）
bash run_main.sh --model iTransformer

也可以直接调用 run.py：

python run.py \
  --task_name long_term_forecast --is_training 1 \
  --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --model_id ETTh1_96_96 --model iTransformer \
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
