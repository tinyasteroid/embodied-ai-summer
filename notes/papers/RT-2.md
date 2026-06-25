---
title: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control"
year: 2023
lab_company:
  - Google DeepMind
paper_type: "VLA / robot foundation model / generalist robot policy"
field:
  - embodied_ai
  - robot_learning
method_tags:
  - VLA
task_tags: []
paper_url: ""
project_url: "https://robotics-transformer2.github.io"
code_url: ""
dataset_url: ""
checkpoint_url: ""
---

# RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control

- Year / Venue：2023 / arXiv
- Lab / Org：Google DeepMind
- Type：VLA / robot foundation model / generalist robot policy
- Open-source（code, dataset）：模型和训练代码未完整开源；论文提供 project website 与实验展示。可记录为 **closed / not fully open-source**。
- Link：`https://robotics-transformer2.github.io`

> 一句话定位：
> RT-2 是 VLA 路线的代表性工作，它把机器人低层动作离散化成类似文本 token 的形式，使预训练 VLM 能够在同一个输出空间中同时生成自然语言和机器人动作，从而把 web-scale vision-language pretraining 中的语义理解能力迁移到真实机器人控制。

> Key words：
> Vision-Language-Action Model, VLA, RT-2, action tokenization, co-fine-tuning, web knowledge transfer, robot control, semantic generalization, emergent capabilities, chain-of-thought reasoning

## Problem

这篇论文要解决的核心瓶颈是：

现有机器人策略虽然可以通过 imitation learning 学会一些具体操作，但泛化能力有限，尤其难以利用互联网规模视觉语言数据中的语义知识。另一方面，大型 VLM 具备强视觉识别、语言理解和简单推理能力，但它们输出的是文本，而真实机器人需要输出低层连续控制动作，例如末端执行器的位移、旋转和夹爪控制。

因此，RT-2 关注的问题是：

> 能否把大规模预训练 VLM 直接接入机器人低层控制，使一个端到端模型既能理解图像和语言，又能输出可执行的机器人动作？

更具体地说，它试图解决三个问题：

1. 如何把 robot action 放进 VLM 的输出空间？
2. 如何在不遗忘 VLM 原有语义能力的前提下学习机器人动作？
3. 互联网视觉语言预训练能否提升机器人在 novel objects、novel instructions、symbol understanding 和 simple reasoning 上的泛化能力？

## Core idea

RT-2 的核心想法很直接：**把机器人动作当成另一种“语言”来建模。**

它将机器人动作离散化为 action tokens，然后把机器人控制任务改写成类似 VQA 的形式：

```text
Q: What action should the robot take to <task instruction>?
A: action token sequence
```

训练时，模型同时看两类数据：一类是机器人轨迹数据，目标是输出 action tokens；另一类是原始 web-scale vision-language data，例如 VQA、captioning 等，目标是保留视觉语言理解能力。这样，模型不仅学会从图像和指令预测动作，还能继承 VLM 中已有的物体、符号、关系和简单推理知识。

## Contributions

1. **提出 Vision-Language-Action Model，VLA，的通用范式**
    RT-2 将 VLM 扩展为能够直接输出机器人动作的模型，不再只是用 VLM 做高层规划或语义模块，而是把它接入低层闭环控制。

2. **提出 action-as-text 的简单实现路径**
    将连续机器人动作离散化为 token，让动作预测可以使用与文本生成相同的自回归 token prediction 框架。

3. **提出 co-fine-tuning 训练策略**
    不只用机器人数据微调 VLM，而是将机器人轨迹数据和原始视觉语言数据共同训练，以保留 web-scale pretraining 中的语义知识。

4. **展示 web knowledge transfer 到机器人控制的效果**
    RT-2 在 novel objects、unseen backgrounds、unseen environments 以及语义指令理解上优于 RT-1、VC-1、R 3 M、MOO 等基线。

5. **展示 emergent semantic capabilities**
    模型能够处理 robot data 中没有明确出现过的符号、关系和简单推理任务，例如把物体放到数字或图标附近、选择最大/最小物体、识别接近掉落的袋子等。

6. **初步探索 chain-of-thought for robot control**
    论文通过让模型先生成自然语言 Plan，再生成 Action tokens，展示了 VLA 中融合语义推理和低层动作生成的可能性。

## Method

### Inputs / outputs

- Observation：机器人相机图像，主要为真实机器人视觉观测
- Language：自然语言任务指令，例如 pick / move / place 等
- State / Proprioception：论文主方法重点在图像与语言到动作，未把 proprioception 作为核心展开
- Action：机器人末端执行器动作，包括 6-DoF translation / rotation、gripper extension、terminate command
- Other：web-scale VQA、captioning、interleaved image-text data；机器人 demonstration trajectory

### Pipeline

```text
Camera image + language instruction
→ Pretrained VLM backbone（PaLI-X / PaLM-E）
→ Output action tokens
→ De-tokenize action tokens
→ Low-level robot action
→ Closed-loop robot control
```

更细一点：

```text
机器人轨迹动作
→ 离散化为 256-bin action tokens
→ 构造成 VQA-like prompt
→ 与 web-scale vision-language data co-fine-tune
→ 推理时约束输出为合法 action tokens
→ 反解码成机器人动作执行
```

### Key design

- 最关键的结构设计：
    - **Action tokenization**：把连续动作离散化为 token。
    - **Co-fine-tuning**：机器人动作数据与原始视觉语言数据联合训练。
    - **Output constraint**：机器人控制任务中约束模型只输出合法 action tokens。
    - **Large VLM backbone**：使用 PaLI-X、PaLM-E 等大规模 VLM 作为基础模型。
- 和普通 VLA / policy / world model 的区别：
    - 和传统 policy 相比：RT-2 不是从零训练视觉编码器和动作头，而是直接复用 VLM 的预训练能力。
    - 和高层 VLM planner 相比：RT-2 不是只输出任务计划，而是直接输出低层动作。
    - 和后来的 OpenVLA 相比：RT-2 更像早期 closed VLA 代表，证明了范式有效；OpenVLA 更强调开源、轻量化和可微调。
    - 和 WAM / world model 相比：RT-2 主要从 static image-language pretraining 继承语义知识，不显式预测 future video / future world state，因此对新物理运动的泛化有限。

## Evaluation

- Setting：real robot + simulation
- Task：
    - seen tasks
    - unseen objects
    - unseen backgrounds
    - unseen environments
    - emergent semantic reasoning tasks
    - Language-Table simulation benchmark
- Dataset / Benchmark：
    - RT-1 机器人数据：13 robots，17 months，office kitchen environment
    - web-scale VLM 数据：VQA、captioning、interwoven image-text examples
    - Language-Table simulation
- Baseline：
    - RT-1
    - VC-1 + RT-1 backbone
    - R 3 M + RT-1 backbone
    - MOO
    - BC-Zero / LAVA 等 simulation baseline
- Metric：
    - real-world rollout success rate
    - generalization success across objects / backgrounds / environments
    - emergent skill success rate
    - ablation comparison across model size and training strategy
- Main result：
    - Seen tasks 上 RT-2 与 RT-1 差距不一定特别大；
    - 在 unseen objects / backgrounds / environments 上，RT-2 明显优于传统机器人策略；
    - emergent capability 上，RT-2 显著强于 RT-1 和 VC-1；
    - co-fine-tuning 优于只用 robot data fine-tuning；
    - 更大模型通常带来更强泛化；
    - Language-Table simulation 中 RT-2-PaLI-3 B 达到 90 ± 10，高于 BC-Zero、RT-1、LAVA。

### Convincing?

整体来说，实验对“VLM 语义知识可以迁移到机器人控制”这一结论是有说服力的，原因是：

1. 评估覆盖了真实机器人、仿真和多个泛化维度；
2. baseline 比较包含 RT-1、预训练视觉表征方法和使用 VLM 的 MOO；
3. ablation 验证了 co-fine-tuning、模型规模和 VLM pretraining 的作用；
4. emergent task 设计能较好地区分“学会动作”与“理解语义后复用动作”。

但它对“学会全新物理技能”并不充分。RT-2 更强的是语义泛化，而不是动作泛化。

## Limitations

1. **不会凭空获得新动作技能**
    RT-2 的 physical skills 仍受限于 robot data 中已有的动作分布。Web-scale pretraining 带来的是语义概念、物体关系、符号理解和简单推理能力，不是新的 manipulation skill。

2. **实时推理成本高**
    最大的 RT-2-PaLI-X-55 B 需要 multi-TPU cloud service，控制频率约 1–3 Hz；5 B 版本约 5 Hz。对于高频、灵巧、双臂操作任务，这仍然是瓶颈。

3. **动作离散化可能限制控制精度**
    将连续动作分桶成 token 简化了训练接口，但可能损失连续控制中的精细动作表达能力。这也是后续 π0 等工作转向 continuous action / flow matching 的原因之一。

4. **主要评估语义泛化，不等于完整通用机器人能力**
    论文展示了 novel objects、symbols、relations 和 simple reasoning，但长时程任务规划、复杂接触操作、高精度装配等能力仍未解决。

5. **系统封闭，复现难度高**
    依赖 PaLI-X / PaLM-E、Google 内部机器人数据与云端 TPU 推理，普通研究者难以完整复现。

## Takeways

- 对我理解具身智能最有用的点：
    1. **VLA 的本质是把 vision、language、action 放到一个统一模型里。**
    2. RT-2 说明大规模 VLM 的语义能力可以迁移到机器人任务，但这种迁移主要发生在 object / language / relation / reasoning 层面。
    3. action tokenization 是早期 VLA 的关键技巧：把控制问题改写成 token prediction。
    4. co-fine-tuning 很重要，因为只用机器人数据微调会损害 VLM 原有的语义知识。
    5. VLA 并不自动解决物理技能泛化，后续 WAM / video model 路线正是在补这个短板。
- 对后续项目 / 复现 / 汇报可能有用的点：
    1. 如果向师兄汇报，可以把 RT-2 定位为 **VLA 路线的概念奠基论文之一**。
    2. 可与 OpenVLA 对比：RT-2 证明范式，OpenVLA 推动开源和可微调。
    3. 可与 π0 对比：RT-2 是 discrete action tokens，π0 是 continuous action flow matching。
    4. 可与 DreamZero / WAM 对比：RT-2 继承 VLM 语义先验，WAM 试图继承 video/world dynamics 先验。
    5. 如果做复现，RT-2 本身不适合作为 baseline；更适合选择 OpenVLA、π0、Octo 或 LeRobot 中可运行的模型。

## Question and future following

- 最需要弄清楚的问题：
    1. action tokenization 相比 continuous action head 的精度损失有多大？
    2. co-fine-tuning 中 web data 和 robot data 的比例如何影响语义保持和动作学习？
    3. RT-2 的 emergent capabilities 是否来自真正的推理，还是来自 VLM 中已有概念的模式匹配？
    4. VLA 在高频控制、双臂操作、接触丰富任务中为什么容易受限？
    5. 为什么后续方法从 action tokens 转向 diffusion / flow matching / action chunking？
- 后续要追的论文方向：
    1. RT-1：理解 Google Robotics Transformer 的前身。
    2. OpenVLA：理解开源 VLA 如何复现和改进 RT-2 范式。
    3. π0：理解 continuous action flow model 如何解决离散 action token 的局限。
    4. RT-X / Open X-Embodiment：理解 cross-embodiment 数据规模化。
    5. WorldVLA / DreamZero / WAM：理解 world model 如何补足 VLA 对物理动态建模不足的问题。
