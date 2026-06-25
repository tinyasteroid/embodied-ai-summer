---
title: "WorldVLA: Towards Autoregressive Action World Model"
year: 2025
lab_company:
  - "DAMO Academy, Alibaba Group"
  - Hupan Lab
  - Zhejiang University
paper_type: "VLA / world model / action world model / autoregressive policy"
field:
  - embodied_ai
  - robot_learning
method_tags:
  - VLA
  - world_model
task_tags: []
paper_url: "https://arxiv.org/abs/2506.21539"
project_url: ""
code_url: "https://github.com/alibaba-damo-academy/WorldVLA"
dataset_url: ""
checkpoint_url: ""
---

# WorldVLA: Towards Autoregressive Action World Model

- Year / Venue：2025 / arXiv
- Lab / Org：DAMO Academy, Alibaba Group；Hupan Lab；Zhejiang University
- Type：VLA / world model / action world model / autoregressive policy
- Open-source（code, dataset）：Code released: https://github.com/alibaba-damo-academy/WorldVLA
- Link：`https://arxiv.org/abs/2506.21539`

> 一句话定位：
> WorldVLA 是一篇尝试把 **Vision-Language-Action Model（VLA）** 和 **World Model** 统一到同一个 autoregressive framework 中的工作，使模型同时具备动作生成和未来图像预测能力。

> Key words：
> VLA, World Model, Action World Model, Autoregressive Model, Action Chunking, Attention Mask, LIBERO, Discrete Action Token, Image Generation

## Problem

这篇论文主要解决两个瓶颈：

第一，现有 VLA 通常只把 action 当作输出，即给定图像和语言指令后直接预测动作。这类模型可以利用 MLLM 的视觉和语言能力，但缺少对 action 本身及其物理后果的理解。换句话说，模型知道“该做什么”，但不一定真正建模“这个动作会让世界如何变化”。

第二，传统 world model 能根据当前状态和动作预测未来视觉状态，但它通常不能直接生成可执行动作，因此不能单独作为机器人 policy 使用。

WorldVLA 试图弥合这两个方向的缺口：让同一个模型既能作为 action model 输出机器人动作，也能作为 world model 预测动作导致的未来图像。

## Core idea

WorldVLA 的核心想法是：把 text、image、action 都离散化为 token，并放入同一个 autoregressive model 中统一建模。模型一方面学习 $\text{image + language} \rightarrow \text{action}$，另一方面学习 $\text{image + action} \rightarrow \text{future image}$。

这样做的直觉是：如果模型不仅要预测动作，还要预测动作执行后的未来图像，那么它会被迫学习更好的 action understanding 和 environment dynamics，从而反过来提升动作生成质量。

此外，作者发现 autoregressive 方式连续生成多个 action 时容易出现误差累积，因此提出 action attention mask：生成当前 action 时屏蔽 prior actions，使当前 action 主要依赖 text/image context，而不是依赖前面可能已经预测错误的 action。

## Contributions

1. 提出 **WorldVLA**：一个 autoregressive action world model，统一 action understanding / action generation / image understanding / image generation。
2. 将 VLA 和 world model 整合到同一个框架中：
    - action model：根据图像和语言生成动作；
    - world model：根据图像和动作预测未来图像；
    - 两者共享统一的 token-based autoregressive backbone。
3. 提出 **action attention masking strategy**，解决 action chunk generation 中的误差传播问题。传统 causal mask 会让后续动作依赖前面动作，若前面 action 错误，后续 action 会继续偏离；WorldVLA 通过屏蔽 prior actions 缓解这一问题。
4. 在 LIBERO benchmark 上验证：WorldVLA 相比 standalone action model 和 standalone world model 都有提升，说明 action model 与 world model 存在互相增强关系。

## Method

### Inputs / outputs

- Observation：历史图像观测 $\{o_{t-h}, \ldots, o_t\}$
- Language：自然语言任务指令，例如 “What action should the robot take to ?”
- State / Proprioception：论文主要描述中没有突出使用 proprioception，核心输入集中在 image、language、action token
- Action：离散化后的机器人动作 token，包括 3 个相对位置、3 个相对角度、1 个 gripper state，共 7 个 action tokens
- Other：future image / next frame，用于 world model 的训练目标

### Pipeline

$$
\text{语言指令 + 历史图像} \rightarrow \text{text/image tokenizer} \rightarrow \text{autoregressive backbone} \rightarrow \text{action tokens} \rightarrow \text{action de-tokenizer} \rightarrow \text{robot action}
$$

同时：

$$
\text{当前图像 + action} \rightarrow \text{image/action tokenizer} \rightarrow \text{autoregressive backbone} \rightarrow \text{future image tokens} \rightarrow \text{image de-tokenizer} \rightarrow \text{next frame}
$$

更简洁地说：

$$
\begin{aligned}
\text{Action Model:}\quad &\text{Text + Image} \rightarrow \text{Action} \\
\text{World Model:}\quad &\text{Image + Action} \rightarrow \text{Future Image} \\
\text{WorldVLA:}\quad &\text{Text + Image + Action} \rightarrow \text{Action + Future Image}
\end{aligned}
$$

### Key design

- 最关键的结构设计：
    1. 使用三个 tokenizer：text tokenizer、image tokenizer、action tokenizer。
    2. 将图像、文本、动作统一到离散 token 空间。
    3. 使用 Chameleon 作为初始化，因为 Chameleon 本身是统一 image understanding 和 image generation 的模型。
    4. 通过混合 action model data 和 world model data 进行联合训练。
    5. 对 action chunk generation 使用特殊 attention mask，避免后续动作依赖前面预测出的动作。
- 和普通 VLA / policy / world model 的区别：
    - 普通 VLA：$\text{Text + Vision} \rightarrow \text{Action}$，动作只作为输出。
    - 普通 world model：$\text{Vision + Action} \rightarrow \text{Future Vision}$，不能直接输出 policy action。
    - WorldVLA：$\text{Text + Vision + Action} \rightarrow \text{Action + Future Vision}$，既能生成动作，也能预测动作导致的未来视觉状态。

## Evaluation

- Setting：simulation
- Task：robot manipulation；包含短程任务和 long-horizon tasks
- Dataset / Benchmark：LIBERO，包括 LIBERO-Spatial、LIBERO-Object、LIBERO-Goal、LIBERO-Long、LIBERO-90
- Baseline：
    - Continuous action models：Diffusion Policy、Octo、DiT Policy、Seer、OpenVLA-OFT、UVA
    - Discrete action models：OpenVLA、WorldVLA
- Metric：
    - Action model：Success Rate（SR）
    - World model：FVD、PSNR、SSIM、LPIPS
- Main result：
    - WorldVLA 在 discrete action model 中优于 OpenVLA。
    - OpenVLA average SR 为 76.5。
    - WorldVLA 256×256 average SR 为 79.1。
    - WorldVLA 512×512 average SR 为 81.8。
    - 加入 world model 后，action model 的平均成功率从 62.8 提升到 67.2。
    - naive action chunking 会导致性能下降，从 62.8 降到 54.0。
    - 使用 proposed action attention mask 后，成功率提升到 76.6；再结合 world model 后达到 78.1。
    - world model 也从 action model 中受益，尤其在长视频预测中，Action World Model 相比 pure world model 有更好的 FVD 和更合理的视觉生成结果。
- Convincing?
    - 对“world model objective 能帮助 action generation”这一点，实验比较有说服力，尤其是 action model ablation。
    - 对“action attention mask 能缓解 action chunk 自回归误差累积”这一点，也有较直接的实验支持。
    - 但对“WorldVLA 是通用机器人基础模型”这一点，证据还不充分，因为实验主要集中在 LIBERO 仿真环境，没有大规模真实机器人验证。
    - 此外，WorldVLA 并没有全面超过 continuous action model。比如 OpenVLA-OFT 在 LIBERO 上的 average SR 更高，因此 WorldVLA 的主要价值更应理解为“统一 action-world 建模方向的验证”，而不是当前最强 policy。

## Limitations

- 实验主要在 LIBERO 仿真 benchmark 上完成，缺少真实机器人实验，不能直接证明其 sim-to-real 或真实环境泛化能力。
- 模型采用 discrete action token，会带来动作离散化误差；论文自己也提到 discrete action model 通常可能弱于 continuous action model。
- world model 主要预测视觉未来状态，但图像生成质量和动作控制质量之间的关系仍需要更系统验证。
- action attention mask 虽然缓解了误差传播，但它也削弱了 action chunk 内部动作之间的显式时序依赖，是否会影响动作平滑性和长期协调仍值得进一步研究。
- 目前主要验证的是短期 next-frame / action chunk 场景，对更长 horizon、更复杂真实任务、多机器人本体迁移的覆盖不足。

## Takeways

- 对我理解具身智能最有用的点：
    1. VLA 不一定只能做 $\text{observation} \rightarrow \text{action}$，还可以结合 world model 学习动作带来的未来状态变化。
    2. action 不能简单等同于 text token。动作 token 缺少大规模预训练，因此 autoregressive action generation 更容易出现误差累积。
    3. world model 的价值不只是“生成视频”，更重要的是为 policy 提供对 physical dynamics 的隐式约束。
    4. action chunking 在机器人控制中很重要，但不同架构对 action chunking 的处理方式不同：离散自回归模型需要解决 token-level error propagation，flow/diffusion policy 则更自然地并行生成连续动作。
- 对后续项目 / 复现 / 汇报可能有用的点：
    1. 可以把 WorldVLA 作为 “VLA + World Model” 路线的代表性论文，用于说明 VLA 正在从 semantic action prediction 走向 dynamics-aware policy learning。
    2. 如果后续复现 LIBERO baseline，可以重点关注它的 action attention mask 设计，而不是完整复现整个 WorldVLA。
    3. 汇报时可以把它和 OpenVLA、π0、DreamZero 放在一条线上：
        - OpenVLA：离散 action-token VLA；
        - π0：continuous action flow model；
        - WorldVLA：discrete autoregressive action-world model；
        - DreamZero：基于 video diffusion backbone 的真实机器人 WAM。
    4. 对工程实现而言，WorldVLA 的关键不是简单加一个 world loss，而是 action / image / text 的 tokenization、loss balance、attention mask 和 action chunk decoding。

## Question and future following

- 最需要弄清楚的问题：
    1. action attention mask 是否会牺牲 action chunk 内部动作连续性？
    2. world model 预测 future image 对真实机器人 policy 到底有多大帮助？
    3. discrete action token 路线是否适合高频、精细、灵巧操作？
    4. WorldVLA 如果加入 proprioception、多视角输入、wrist camera，结构是否仍然有效？
    5. 在真实机器人中，image generation objective 是否会引入过多视觉细节噪声，反而影响 action learning？
- 后续要追的论文方向：
    1. OpenVLA / OpenVLA-OFT：理解离散 action-token VLA 的基础路线。
    2. π0：理解 continuous action + flow matching 的 VLA 路线。
    3. DreamZero / WAM：理解基于 video diffusion backbone 的 world-action joint modeling。
    4. GR-1 / GR-2：理解 video prediction 如何用于机器人策略学习。
    5. Diffusion Policy / ACT：理解 action chunking 在非自回归连续动作策略中的作用。
