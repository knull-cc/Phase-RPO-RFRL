# Phase-RPO-RFRL 需求文档

## 1. 文档目的

本文档用于明确一个面向时间序列预测的可插拔方法框架需求。该方法名称确定为 Phase-RPO-RFRL，其目标不是重新设计新的时序 backbone，而是在现有时间序列预测模型之外增加一个相位感知检索、检索偏好优化、强化学习式控制与自适应融合模块，从而缓解 retrieval-enhanced TSF 中“检索有时有益、有时有害”的核心问题。本文档的作用是把当前已经讨论过的方法图、研究动机和实验要求统一成一个可执行的需求定义，作为后续方法设计、代码实现、实验规划和论文写作的共同约束。

## 2. 问题定义

现有时间序列预测方法通常只基于当前 lookback window 建模未来走势，而 retrieval-augmented TSF 路线尝试从历史库中检索相似片段，为当前预测提供额外记忆支持。这条路线已经被多篇近期工作证明具备潜力，但也暴露出一个尚未被充分解决的问题：检索到的历史样本并不总是有帮助。某些被检索到的片段只是时间域上表面相似，但在周期位置、分布状态、演化方向或真实机制上并不一致，最终会误导预测模型。在非平稳条件下，这个问题会更严重，因为旧 regime 中相似的历史并不必然对新 regime 下的未来具有预测价值。



那么Phase-RPO-RFRL 需要解决的不是“如何让模型总是检索”，而是“如何让模型在每个样本上判断检索是否可靠、应该检索什么、应该信任多少检索结果，以及何时拒绝检索”。这意味着检索不再是一个固定前处理步骤，而应被视为预测过程中的一个受监督、受偏好约束、可动态控制的动作决策。

## 3. 方法定位

Phase-RPO-RFRL 必须被定位为一个插件式框架，而不是新的 backbone。它应当可以接入 DLinear、PatchTST、iTransformer、TimesNet、TSMixer 或其他能够输出基础预测的时间序列模型。Phase-RPO-RFRL 的职责是为这些 backbone 增加一个可训练的 retrieval-control branch，而不是替代主干模型完成所有预测任务。这个定位有两个直接要求。第一，方法在结构上必须保持 model-agnostic，但默认架构必须优先贴合“Host Model + Plug-in”形式，而不是任意发散到各种不受约束的接入方式。第二，实验设计必须体现这种可插拔性，不能只在一个高度特定的 backbone 上工作，否则贡献会退化成特定模型的小改造。

Phase-RPO-RFRL 同时也不是一个简单的 gate。它不能被设计成只有“检索”与“不检索”两个静态选项，否则创新性太薄，容易被归类为启发式后处理。Phase-RPO-RFRL 必须体现为一个更完整的 retrieval action policy，也就是模型能够学习什么时候检索、按什么记忆机制检索、检索后如何融合，以及在检索候选具有误导性时如何显式降低其权重或拒绝其参与预测。

## 4. 标准结构

Phase-RPO-RFRL 的标准结构必须严格对应如下主路径：输入时间窗口先进入 Host Model，Host Model 生成 baseline forecast，并向插件暴露可用的 features 或 state；插件内部按照 `Phase-aware Retrieval -> RPO -> RFRL Controller` 的顺序工作；插件输出 retrieval-enhanced forecast，并同时回传 retrieval guidance 作为 adaptive control 信号；随后由一个明确独立的 Adaptive Fusion 模块融合 baseline forecast 与 retrieval-enhanced forecast，最终生成 final forecast。

这一定义意味着，Adaptive Fusion 是确定模块，不是可选后处理。它必须在需求层面被单独承认，而不是仅作为某些实验里的附加技巧出现。后续实现可以选择线性融合、门控融合或条件融合，但结构上不能省略这一级。

这一定义也意味着，当 Host Model 能够暴露中间特征、隐状态或不确定性估计时，插件必须优先消费这些 features 或 state，并向 Host Model 或融合模块回传 retrieval guidance。这个接口在具备条件时属于硬约束，而不是可选优化。只有当某些极简 backbone 无法安全暴露中间状态时，才允许退化为仅基于输入与输出的弱接入实现。

## 5. 核心思想

Phase-RPO-RFRL 的核心思想由四个紧密耦合的部分组成。第一部分是 Phase-aware Retrieval。对当前输入窗口进行 FFT、STFT 或小波变换后，方法需要提取主导频率的幅值与相位信息。幅值用于描述某个周期成分的强度，相位用于描述当前窗口处于该周期的哪个位置。方法需要利用这一表示去检索相位对齐或近似对齐的历史片段，而不仅仅是在时间域中寻找形状相似的窗口。提出这一要求的原因是，时间域相似容易产生伪相似样本，而相位信息能够更直接地区分“看起来像”与“周期位置一致”。

第二部分是 Retrieval Preference Optimization。Phase-RPO-RFRL 不能假设检索总有益，而是需要通过偏好学习去建模“检索是否有益”。对同一个训练样本，方法需要能够比较至少两类预测：一种是不使用检索的 backbone 基础预测，另一种是使用相位检索后的增强预测。在条件允许时，也可以进一步比较时间域检索、幅值检索、相位检索、混合检索等不同动作的预测结果。然后根据真实未来值计算预测误差，把误差更低的检索动作视为 preferred，把误差更高或更不稳定的检索动作视为 rejected。RPO 在这里承担的是 utility-grounded preference 建模任务，它学习的不是文本答案偏好，而是检索决策偏好。

第三部分是 RFRL Controller。仅靠偏好学习并不足以在推理时做出细粒度控制，因此方法需要引入强化学习式或策略学习式的控制模块，把“历史上哪些检索动作更好”转化为“当前样本应如何选择检索动作”的动态策略。这个控制器需要根据当前窗口、频域特征、检索候选、一致性度量、不确定性以及已有预测信号做决策，输出是否启用检索、选择哪些候选、使用哪些频段、采用多大控制强度等动作。RFRL 的职责不是直接预测未来，而是对检索使用策略进行学习，并把 retrieval guidance 回传给 Host Model 或 Adaptive Fusion。

第四部分是 Adaptive Fusion。Adaptive Fusion 负责把 baseline forecast 与 retrieval-enhanced forecast 合成为 final forecast。它必须消费来自 RFRL Controller 的控制信号，并体现为一个明确的可分析模块，而不是模糊地隐藏在主干或检索分支内部。

## 6. 总体目标

Phase-RPO-RFRL 的首要目标是降低时间序列预测任务中的 MAE 和 MSE，尤其是在存在检索候选失配、周期错位、非平稳 shift 或错误类比历史的情况下，减少有害检索对预测结果的误导。相比于总是检索的 retrieval-augmented baseline，Phase-RPO-RFRL 应当在平均性能上取得可重复的误差改进，或者在 shift slice、hard slice、harmful retrieval slice 上表现出显著的稳健性优势。

Phase-RPO-RFRL 的第二目标是提供可解释的检索行为分析。方法不仅要输出预测结果，还要输出对检索决策的可分析信号，例如当前样本是否启用检索、使用了哪些候选、使用了哪些频段、retrieval guidance 强度是多少、最终融合权重是多少，以及对应的偏好分数或策略得分。这一点很重要，因为如果性能提升完全不可解释，reviewer 会倾向于把方法视为复杂堆叠而不是机制贡献。

Phase-RPO-RFRL 的第三目标是保留工程可实现性。方法应当在现有 TSF 代码库和标准 benchmark 上可以较低成本接入，不依赖从头训练大模型或构建超大外部语义系统。若方法过于复杂，导致实验和实现成本远高于收益，其研究价值会被明显削弱。

## 7. 非目标

Phase-RPO-RFRL 当前阶段不以构建新的通用 TSF backbone 为目标，不以大规模 foundation model 预训练为目标，也不以多模态文本检索为主目标。它也不追求通过单一启发式门控获得局部收益，因为这类设计难以支撑方法论文的主张。方法还不应把所有精力放在更强检索器本身上；检索器可以改进，但它不是方法贡献的唯一核心。真正的贡献应当落在“相位感知候选构造 + 偏好学习 + 动态策略控制 + 自适应融合”这条主线上。

## 8. 功能需求

Phase-RPO-RFRL 必须支持对输入时间窗口提取频域表示，并基于主导频率的相位信息构建检索键。这个检索模块至少要支持相位感知相似度，并允许后续扩展到相位与幅值联合相似度、相位一致性约束、多频段加权相似度等形式。系统必须能够从历史记忆库中返回候选片段，并保留每个候选的相似度、频域特征和时间索引信息，以供后续策略模块使用。

Phase-RPO-RFRL 必须支持至少两种预测路径，一种是无检索基础路径，另一种是带检索增强路径。这样做的原因不是简单地做 ensemble，而是为 RPO 提供偏好样本构造基础。若没有至少两个可比较动作，RPO 无法定义 utility preference。系统还应允许扩展到更多动作，例如时间域检索、相位检索、混合检索、no-retrieval、top-k variation、negative analog retrieval 等。

Phase-RPO-RFRL 必须包含一个偏好学习模块，用于根据真实未来值生成 action preference。该模块需要把“带检索预测优于无检索预测”或“某类检索优于另一类检索”转化为可训练信号，并输出一个面向检索动作的 preference score。这个 score 后续需要作为 RFRL 的 reward shaping 或 critic-like guidance 使用，因此不能只是实验期的离线统计量，而需要具备被模型消费的形式。

Phase-RPO-RFRL 必须包含一个策略控制模块，用于动态决定检索行为。该模块至少要支持以下控制变量：是否检索、选择哪些候选、候选权重分配、检索分支控制强度以及对融合模块的引导信号。若资源允许，该模块还应支持频段选择或 phase tolerance 选择，以验证“按相位检索”不是单一静态设计，而是可调度策略的一部分。

Phase-RPO-RFRL 必须包含一个明确独立的 Adaptive Fusion 模块，用于融合 baseline forecast 与 retrieval-enhanced forecast。这个模块必须是标准实现的一部分，并对外暴露融合权重、门控分数或其他可解释融合信号。

Phase-RPO-RFRL 必须保留模块级可拔插性。也就是说，同一套 retrieval-control framework 应能作用于不同 backbone，至少需要存在一种统一接口，例如输入一个 lookback window，输出基础预测、检索增强预测、融合预测以及相关的策略信号。这一接口需求会直接决定后续代码结构与实验可比性。

## 9. 数据与检索需求

Phase-RPO-RFRL 需要运行在标准 TSF benchmark 上，并优先选择那些具有明显周期性、准周期性或 shift 风险的数据集，例如 ETT、Weather、Traffic、Electricity、Exchange、ILI 等。数据切分必须保持时间安全，不能使用未来信息污染检索库或构造 retrieval pool。检索库原则上应从训练阶段历史窗口中构建，并严格与验证集、测试集的未来片段隔离。

Phase-RPO-RFRL 需要支持从历史库中检索 lookback-horizon pair，或者至少检索 lookback pattern 并让模型据此生成 horizon 预测。无论采用哪种记忆单元，都必须防止通过检索过程直接泄漏未来目标窗口。若一个候选片段包含与测试目标时间上重叠或不可接受接近的信息，该候选必须被排除。该需求不是附属约束，而是方法可发表性的底线之一。

## 10. 训练需求

Phase-RPO-RFRL 的训练流程至少应分为两个阶段。第一阶段是 backbone 与基础 retrieval branch 的可用性建立阶段，即系统能够稳定地产生无检索预测和带检索预测，并形成可比较的 action outcome。第二阶段是偏好优化与策略学习阶段，即使用真实未来误差构造 preference signal，再用这一信号约束或引导策略控制器。

RPO 的训练信号应当以预测误差改进为核心，而不是仅依赖检索相似度本身。一个候选即使相似度很高，只要它带来的 downstream forecast 更差，就应当被归入 rejected action。RFRL 的 reward 设计应当在基础 forecast utility 之上叠加 preference shaping、retrieval cost 和必要的不确定性惩罚，这样才能避免控制器学到“只要多检索就可能偶尔变好”的投机策略。

Phase-RPO-RFRL 还需要支持较稳健的训练退化路线。如果 RFRL 训练不稳定，系统应能退化为仅使用相位检索、RPO 和确定性的 Adaptive Fusion 版本，以保证方法至少保留一个可交付的中间结果。这个需求很现实，因为 RL 部分最容易成为工程瓶颈和论文风险点。

## 11. 实验需求

Phase-RPO-RFRL 的实验设计必须体现出方法贡献的分层结构，而不是只报告一个最终版本。最低要求是比较同一个 backbone 的四个版本：无检索 baseline、加入相位检索的版本、加入相位检索与 RPO 及 Adaptive Fusion 的版本、加入相位检索与 RPO、RFRL 及 Adaptive Fusion 的完整版本。这样才能回答三个关键问题：相位检索是否单独有用，RPO 是否能够稳定减少有害检索，RFRL 是否在此基础上进一步带来动态控制收益。

Phase-RPO-RFRL 的比较对象不能只包括无检索 backbone，还必须包括 always-retrieve 的 retrieval baseline、简单阈值门控、固定融合、stationarity-aware retrieval 或近期 retrieval-TSF 代表方法。否则方法即便优于 backbone，也无法说明它真正解决了 harmful retrieval 问题。

Phase-RPO-RFRL 的主指标应为 MAE 和 MSE，同时需要提供若干诊断指标，例如 retrieval gain、harmful retrieval rate、abstention rate、shift-slice performance、candidate acceptance rate、fusion weight distribution 等。这些诊断指标的作用不是替代主指标，而是帮助解释模型是否真的学会了“何时相信检索”。

## 12. 验收标准

Phase-RPO-RFRL 的最低验收标准是，在至少一个强 backbone 上，相对于无检索 baseline 和 always-retrieve baseline，在主要 benchmark 或有代表性的 shift slice 上取得稳定且可复现的 MAE/MSE 改进，并且该改进不是偶然来自数据泄漏、超参数堆叠或异常切分。更高一级的验收标准是，在两个或以上 backbone 上都能复现方向一致的收益，证明方法具有可插拔性而不是 backbone 特例。

从研究角度看，Phase-RPO-RFRL 还必须满足一个机制验收标准，即实验结果能够支撑如下判断：错误检索确实是常见问题，相位感知检索能够改善候选质量，RPO 能够学习检索动作偏好，RFRL 能够在推理时把这种偏好转化为更稳健的动作决策，而 Adaptive Fusion 负责把 baseline forecast 与 retrieval-enhanced forecast 合理整合为 final forecast。如果这条证据链断裂，例如只有完整模型有效但任何中间模块都无法解释其贡献，那么方法的论证力度会明显不足。

## 13. 风险与约束

Phase-RPO-RFRL 的主要风险之一是相位检索只在强周期数据上有明显价值，而在趋势主导、事件主导或弱周期数据上收益有限。因此方法不能把 phase-only similarity 作为唯一信号，至少要允许与幅值信息、时间域信息或不确定性信息结合。第二个风险是 RL 控制器训练成本高且不稳定，因此实现顺序上必须先验证相位检索和 RPO 的独立收益，再决定是否把 RFRL 作为完整论文版本的关键模块。第三个风险是方法被 reviewer 质疑为“只是一个更复杂的 gate”，所以在设计上必须保留多动作检索策略、偏好建模和动态融合，而不是停留在二元开关。

另一个必须明确的约束是计算预算。Phase-RPO-RFRL 不应要求每个样本做过重的全库频域搜索和复杂策略 rollout，否则即使精度有增益，工程代价也会削弱方法实用性。因此后续实现中需要关注 memory bank 构建方式、近邻检索效率和策略控制开销。

## 14. 交付物要求

Phase-RPO-RFRL 最终应至少形成四类交付物。第一类是方法定义文档，即当前这份需求文档以及后续更细的设计文档。第二类是可运行原型，能够在一个 backbone 和一个标准数据集上完成训练、检索、融合与评估。第三类是完整实验结果，至少覆盖主结果、分阶段消融、shift-aware 诊断和失败案例分析。第四类是论文材料，包括方法图、算法描述、实验表格、风险讨论和与近期 retrieval-TSF 方法的对比论证。

## 15. 当前结论

Phase-RPO-RFRL 应被视为 RAP 问题卡片的一个更具体、机制更完整的实现方向。RAP 提供的是研究问题层面的表述，即如何学习 retrieval action policy 来降低 TSF 的 MAE 和 MSE；Phase-RPO-RFRL 提供的是这个问题的一种具体方法答案，即用相位感知检索构造更高质量候选，用 RPO 学习检索动作偏好，再用 RFRL 把这种偏好转化为样本级检索控制策略，并通过 Adaptive Fusion 输出 final forecast。后续所有实现与实验都应围绕这一主线展开，而不应回退到“简单检索增强”或“单一 gate”这种过弱方案。
