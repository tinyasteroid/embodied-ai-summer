> 知乎论文解读：https://zhuanlan.zhihu.com/p/2044846506851684383

![](../../Attachments/Cosmos%203-1782823838239.webp)
**一句话总结**
Cosmos 3 实现**模型的“大一统”**，把 language / image / video / audio / action 都放进一个统一的 omnimodal world model 框架里。

**解决痛点**
当前专用型模型拼接式，系统架构复杂，且各模型输出空间不同，难以端到端协同优化，也缺乏了不同形态数据之间彼此交互的信号。

**创新点和贡献**
- 将多个专用模型变成一个 omnimodal world model，实现“大一统”，支持多输入组合
- 将 action 作为输入/输出的模态之一，可进入 diffusion subsequence，和 video/audio 等连续模态一起建模。
- MoT 架构，AR 保留语言和视觉理解能力，diffusion 负责连续模态生成，MoT 通过双 tower 和 shared attention 连接两者。
- Cosmos 3-Nano 基于下游的 post-training 可变成强 robot policy，并取得 SOTA。

## 模型架构
![](../../Attachments/Cosmos%203-1782824406682.webp)
### 核心思想
把所有模态的输入编码为 token，送进统一的 transformer 架构中。在同一个框架里融合了 LLM-style autoregressive modeling 和 diffusion-style generative modeling。

### 架构
#### Encoder
这里着重谈 action token 的表示，Cosmos 3 把不同 embodiment 的动作都拆成三类：
- **Ego poses**：agent 主观察坐标系的运动
- **Effector poses**：末端传感器
- **grasp states**：夹爪开合等 manipulation state

用法上支持三类任务，如 Figure 4 所示。
![](../../Attachments/Cosmos%203-1782825829915.webp)

在非语言模态加入可学习的 modality-specific embedding，让 Transformer 能区分模态。
#### Token arrangement
Cosmos 3 将 token sequence 固定拆成两段：[AR subsequence, DM subsequence]，在 DM subsequence 里不同模态的数据 token 也按照统一顺序进行排列。clean 在前，noisy 在后；vision -> audio -> action。

AR subsequence 负责理解和推理，生成时，AR 提供“条件上下文”，让模型看懂图像、理解语义。AR 段不能看 diffusion 段。

diffusion subsequence 中的 noisy tokens 被逐步 denoise 成 clean tokens，负责生成 图像、视频、音频、动作等连续模态。DM tokens 之间用 full bidirectional attention，这个过程需要 token 之间的双向通信，以保持视频、音频、动作之间的一致性。

$$
O_{\mathrm{AR}}
=
\mathrm{Attn}_{\mathrm{causal}}
(Q_{\mathrm{AR}}, K_{\mathrm{AR}}, V_{\mathrm{AR}})
$$

$$
O_{\mathrm{DM}}
=
\mathrm{Attn}_{\mathrm{full}}
(
Q_{\mathrm{DM}},
[K_{\mathrm{AR}}; K_{\mathrm{DM}}],
[V_{\mathrm{AR}}; V_{\mathrm{DM}}]
)
$$

| 子序列                   | 作用            | 生成方式                  |
| --------------------- | ------------- | --------------------- |
| AR subsequence        | 理解、推理、语言生成    | next-token prediction |
| Diffusion subsequence | 图像、视频、音频、动作生成 | iterative denoising   |

#### MoT Architecture
Mixture-of-Transformers 架构，分为了 Reasoner tower 和 Generator tower，采用**独立的两套参数**，通过 Joint attention 进行交互。

|Tower|处理对象|职责|
|---|---|---|
|Reasoner tower|AR subsequence|语言、自回归推理、理解|
|Generator tower|Diffusion subsequence|图像、视频、音频、动作生成|
Reasoner 是标准的 causal attention，进行 next token prediction。
Generator 是 full attention

## 训练流程和数据
![](../../Attachments/Cosmos%203-1782831146656.webp)

## 实验
![](../../Attachments/Cosmos%203-1782824652301.webp)结论：同一个 Cosmos 3 模型家族可以覆盖多类任务，并在多数任务上与专用模型竞争或超过开源专用模型。

Reasoner 仍然具备多模态理解能力。
Generator 可以覆盖图像、视频、音频、transfer generation *Sec. 6.2.1 到 Sec. 6.2.4。*

![](../../Attachments/Cosmos%203-1782830662945.webp)
*Sec. 6.2.5 Action Generation Evaluation*
实验比较了两种初始化：

| 初始化     | 含义                                                                                          |
| ------- | ------------------------------------------------------------------------------------------- |
| PT-init | 只从 pre-trained checkpoint 开始，没有见过 action-domain data                                        |
| MT-init | 从 mid-trained checkpoint 开始，已经见过多领域 action data，包括 forward dynamics、inverse dynamics、policy |
如果 MT-init 好于 PT-init，说明 action mid-training 能在多个 action domain 之间学到可迁移结构。

![](../../Attachments/Cosmos%203-1782830924279.webp)
*Sec. 6.2.5 Robot Policy，Table 19*
Cosmos 3-Nano post-train 到 DROID 机器人策略上，得到 Cosmos 3-Nano-Policy-DROID。在RoboLab simulation benchmark 和 RoboArena real-world benchmark 上都取得 SOTA。
证明了：经过 post-training 后，Cosmos 3 可以成为 robot policy。
