---
title: "RynnVLA-002: A Unified Vision-Language-Action and World Model"
year: 2026
lab_company:
  - "DAMO Academy, Alibaba Group"
  - Hupan Lab
  - Zhejiang University
paper_type: "WAM / unified VLA-world model"
field:
  - embodied_ai
  - robot_learning
  - vision_language_action
  - world_model
method_tags:
  - VLA
  - world_model
  - action_world_model
  - Chameleon
  - autoregressive_model
  - action_chunking
  - continuous_action_head
  - Action_Transformer
  - attention_mask
task_tags:
  - robot_manipulation
  - LIBERO
  - LeRobot
  - SO 100
  - pick_and_place
  - real_robot
paper_url: "https://arxiv.org/abs/2511.17502"
project_url: ""
code_url: "https://github.com/alibaba-damo-academy/RynnVLA-002"
dataset_url: ""
checkpoint_url: ""
---

# RynnVLA-002: A Unified Vision-Language-Action and World Model

## 1. 一句话总结

这篇论文主要解决：传统 VLA 只把 action 当作输出，缺少对“动作会如何改变世界”的显式建模；而普通 world model 虽然能预测未来状态，却不能直接作为 policy 输出动作。RynnVLA-002 将 VLA model 和 world model 统一到同一个 autoregressive action-world model 中，让模型同时学习 action prediction 和 action-conditioned future image prediction，并通过 continuous Action Transformer 提升真实机器人上的动作平滑性与泛化能力。

更简洁地说：**它是 WorldVLA 思路的工程化升级版，用 world modeling 辅助 VLA 学动态，用连续动作头解决真实机器人控制问题。**

## 2. 核心贡献

### 2.1 核心贡献 1

提出 **RynnVLA-002**，一个统一的 **Vision-Language-Action + World Model** 框架。

它把两类任务放到同一个模型中：

- VLA task：根据语言指令、机器人状态、历史图像预测动作；
- World model task：根据当前图像和动作预测下一帧图像。

这样做的核心目的不是单纯“生成未来图像”，而是让模型在训练中学习 action 与 world transition 的关系，从而增强动作生成能力。

### 2.2 核心贡献 2

在离散动作生成中引入 **action attention mask**，缓解 autoregressive action chunk generation 的错误传播问题。

普通自回归生成动作序列时，后一个 action 会依赖前一个 action。如果前面预测错，错误会继续传递。RynnVLA-002 的做法是让当前 action 只看 text / image / state 等上下文，而不看前面的 action，从而减少 chunk 内部的 error accumulation。

### 2.3 核心贡献 3

加入 **continuous Action Transformer head**，解决离散 action token 在真实机器人上的动作不平滑和泛化差问题。

论文发现，离散 action chunk 在 LIBERO 仿真中表现较好，但在真实 SO 100 机器人上容易失败，原因包括：

- 大型 autoregressive 模型在小规模真实数据上容易过拟合；
- action mask 让 chunk 内各动作相互独立，可能破坏动作连续性；
- 离散 token 生成速度慢，轨迹容易抖动。

因此作者加入一个较小的 Action Transformer，用 learnable action queries 并行输出连续 action chunk，并用 L 1 loss 监督。这个模块是 RynnVLA-002 能在真实机器人上工作的关键。

## 3. 数据与任务设置

### 3.1 Data

论文包含两类主要数据：

1. LIBERO 仿真数据
    用于评估模型在标准 manipulation benchmark 上的性能。作者对数据做了清洗，包括去除失败轨迹和 no-op actions。

2. LeRobot SO 100 真实机器人数据
    作者自己采集了两个 pick-and-place 任务的数据，均来自人工遥操作示范：

    - Place the block inside the circle：248 条 demonstrations；
    - Place strawberries in the cup：249 条 demonstrations。

论文强调 RynnVLA-002 在主要 LIBERO 实验中没有额外大规模 robot-data pretraining，而是通过统一 action-world objective 获得较强表现。

### 3.2 Robot / Embodiment

仿真部分主要基于 LIBERO benchmark 中的机器人 manipulation 环境。

真实机器人部分使用：

- LeRobot SO 100 robotic arm；
- 单臂操作；
- 任务类型偏基础 pick-and-place；
- 输入包含 front camera、wrist camera 和 proprioceptive state。

### 3.3 Simulation or Real Robot

两者都有。

Simulation：

- LIBERO-Spatial
- LIBERO-Object
- LIBERO-Goal
- LIBERO-Long

Real Robot：

- LeRobot SO 100
- 两个真实 pick-and-place 任务
- 三种测试场景：single-target、multi-target、with distractors

### 3.4 Task

仿真任务覆盖：

- Spatial：空间关系理解；
- Object：物体识别与操作；
- Goal：目标变化下的任务执行；
- Long：长时程 manipulation。

真实任务覆盖：

- 将方块放入圆圈；
- 将草莓放入杯子；
- 在多目标和干扰物场景下按指令完成操作。

整体任务仍以基础 manipulation 为主，没有覆盖更复杂的移动操作、双臂操作、柔性物体或长程真实任务。

## 4. 模型结构

### 4.1 Vision encoder

RynnVLA-002 初始化自 **Chameleon**，因为 Chameleon 本身支持统一的图像理解与图像生成。

图像部分使用 Chameleon 的 image tokenizer：

- 基于 VQ-GAN；
- 图像 token codebook size 为 8192；
- 256×256 图像对应 256 个 image tokens；
- 512×512 图像对应 1024 个 image tokens；
- 支持 front-view 和 wrist-view 图像输入。

这里的关键点是：它不是只用视觉 encoder 把图像编码成 hidden states，而是将图像离散化为 tokens，使图像既能被理解，也能被生成。

### 4.2 Language model

模型采用 Chameleon-style autoregressive backbone。

语言部分使用 text tokenizer，输入 prompt 主要有两类：

VLA mode：

```text
What action should the robot take to <task>?
```

World model mode：

```text
Generate the next frame based on the current image and the action.
```

同一个 backbone 根据不同 query 执行不同任务：

- 被问“应该采取什么动作”时，作为 VLA policy；
- 被要求“根据当前图像和动作生成下一帧”时，作为 world model。

### 4.3 Action representation

论文同时支持离散动作和连续动作。

离散动作：

- 将每个 action dimension 离散到 256 个 bins；
- action token 与 image token、text token、state token 共享同一个 vocabulary；
- 动作包括相对位置、相对姿态和 gripper state；
- 适合统一建模，但真实机器人上动作不够平滑。

连续动作：

- 不经过 tokenization；
- 由 Action Transformer 直接输出 raw continuous actions；
- 更适合真实机器人控制。

### 4.4 Action decoder / head

模型有两个 action 输出路径。

第一条是 discrete action autoregressive head：

- 基于 token prediction；
- 使用 cross-entropy loss；
- 可与 image/text/state token 统一建模；
- 主要用于保持 action-world token-level unified modeling。

第二条是 continuous Action Transformer：

- 输入 language、image、state tokens；
- 使用 learnable action queries；
- 并行输出整个 action chunk；
- 使用 L 1 regression loss；
- 优点是更平滑、更快、更不容易在小规模真实数据上过拟合。

对真实机器人而言，continuous Action Transformer 是更关键的执行模块。

### 4.5 World model component

World model 的任务是：

$$
\text{current image + action} \rightarrow \text{next image}
$$

它根据当前图像和动作预测未来视觉状态。论文认为这会迫使模型学习：

- 动作如何改变环境；
- 物体运动和接触关系；
- 当前 action 与下一状态之间的动态关系；
- 对 action 的更深层理解。

它的训练目标是生成未来 image tokens，使用 image token cross-entropy loss。

需要注意：RynnVLA-002 推理做控制时，并不是先 rollout 多个未来图像再做 planning。它的 world model 主要作为训练时的辅助任务和共享表征学习信号，而不是显式 model-based planning。

### 4.6 Latent passing / intermediate representation

这篇论文的 intermediate representation 可以理解为共享的 token space 和 shared backbone hidden states。

核心机制是：

```text
image / text / state / action
→ tokenizer
→ shared vocabulary
→ Chameleon-style autoregressive backbone
→ discrete action tokens / future image tokens / continuous Action Transformer actions
```

其中：

- VLA task 提供 action supervision；
- world model task 提供 future image supervision；
- 两者共享 backbone；
- continuous Action Transformer 读取 backbone 表征并输出连续动作。

所以 RynnVLA-002 的“latent passing”不是像模块化系统那样把 world model 输出显式传给 policy，而是在同一个模型内部通过 joint training 共享表征。

## 5. Evaluation

### 5.1 Benchmark

主要 benchmark：

- LIBERO-Spatial
- LIBERO-Object
- LIBERO-Goal
- LIBERO-Long

LIBERO 主要用于检验模型的仿真 manipulation 能力，包括空间关系、物体识别、任务目标变化和长时程任务。

### 5.2 Real robot

真实机器人实验使用 LeRobot SO 100。

任务：

1. Place the block inside the circle
    更偏基础目标检测和抓取执行。

2. Place strawberries in the cup
    更依赖细粒度定位和 grasp-point prediction。

测试场景：

- Single-Target：桌面上只有一个目标；
- Multi-Target：桌面上有多个目标；
- With Distractors：桌面上有目标和干扰物。

主要结果：

- block task：
    - RynnVLA-002：90 / 90 / 80
    - 在 multi-target 和 distractor 场景中优于或持平于 π0、π0.5、GR 00 T N 1.5。
- strawberry task：
    - RynnVLA-002：80 / 80 / 50
    - 整体与强 baseline 接近，在 multi-target 下表现较好。

### 5.3 Simulation

LIBERO 上，RynnVLA-002 分为 discrete 和 continuous 两个版本。

主要结果：

- RynnVLA-002-Discrete：
    - Spatial：94.2
    - Object：96.8
    - Goal：94.6
    - Long：87.6
    - Average：93.3
- RynnVLA-002-Continuous：
    - Spatial：99.0
    - Object：99.8
    - Goal：96.4
    - Long：94.4
    - Average：97.4

这个结果说明 continuous action head 不仅真实机器人更重要，在 LIBERO 上也达到更高平均成功率。

### 5.4 Main metric

VLA / policy 评估：

- Success Rate

World model 评估：

- FVD
- PSNR
- SSIM
- LPIPS

其中 policy 部分更重要，world model 指标用于验证未来图像预测质量。

### 5.5 Baseline

LIBERO 中比较的 baseline 包括：

- LAPA
- TraceVLA
- OpenVLA
- SpatialVLA
- NORA
- CoT-VLA
- π0-FAST
- MolmoAct
- FlowVLA
- UniVLA
- Diffusion Policy
- Octo
- MDT
- DiT Policy
- MaIL
- ThinkAct
- π0
- SmolVLA
- π0.5
- OpenVLA-OFT
- X-VLA
- EO-1
- UVA

真实 SO 100 中比较的 baseline：

- GR 00 T N 1.5
- π0
- π0.5

这些 baseline 大多有官方 pretrained checkpoints，而 RynnVLA-002 在相同 SO 100 数据上训练/微调后进行比较。

## 6.我的理解与疑问

我认为这篇论文最值得记住的是：

**RynnVLA-002 不是简单地“给 VLA 加一个视频预测任务”，而是试图把 action prediction 和 world prediction 放进同一个 action-world modeling 框架中。** 它的核心思想是：如果模型需要根据 action 预测未来图像，它就必须理解动作和物理变化之间的关系；这种理解反过来能帮助它更好地产生动作。

第二个值得记住的点是：**离散 action token 对统一建模很方便，但真实机器人控制更依赖连续动作头。** 这和 π0、Diffusion Policy、ACT 等工作的直觉是一致的：真实控制不仅要“选对动作语义”，还要保证动作轨迹连续、稳定、低抖动。

第三个值得记住的点是：**wrist camera 和 proprioceptive state 对真实操作不是可选项。** 真实任务里，末端执行器附近的局部视角和机器人自身状态会直接影响抓取时机、夹爪闭合和放置精度。

我还没理解清楚的是：

1. world model 在推理阶段有没有被真正用于 planning？
    目前看，它主要是训练时的辅助任务，不是显式 rollout candidate actions 再选择最优 action。

2. continuous Action Transformer 和 ACT / Diffusion Policy / π0 flow matching head 的本质差异是什么？
    它更像一个并行 action chunk regression head，而不是 diffusion / flow matching policy。

3. 为什么 world model training 对 action generation 的提升可以这么明显？
    论文给出了经验结果，但从机制上还需要进一步理解：提升来自更好的视觉表征、更好的 action understanding，还是对物理动态的隐式建模？

4. 图像 token 生成的计算成本是否会限制实际扩展？
    论文自己也提到，真实任务训练成本较高，一个真实任务大约需要数天训练，主要瓶颈来自 image generation token 数量。

和我已读论文的关系：

- 和 OpenVLA / RT-2 的关系：
    OpenVLA、RT-2 主要是典型 VLA 范式，即 image + language → action。RynnVLA-002 在此基础上进一步加入 world model 任务，让 action 也作为输入参与未来图像预测。

- 和 WorldVLA 的关系：
    RynnVLA-002 可以看作 WorldVLA 的升级版。WorldVLA 已经提出 action-world model 和 action attention mask；RynnVLA-002 在此基础上加入 state、wrist camera、continuous Action Transformer，并扩展到真实 SO 100 机器人实验。

- 和 π0 的关系：
    π0 的重点是 VLM backbone + flow matching continuous action expert，用连续动作建模解决高频、灵巧控制。RynnVLA-002 的重点是 unified VLA + world model，同时也承认真实机器人需要 continuous action head。

- 和 DreamZero / WAM 的关系：
    DreamZero 更偏向基于 pretrained video diffusion backbone 的 World Action Model，强调从视频动态先验中获得 zero-shot generalization。RynnVLA-002 更像 Chameleon-style token unification 路线，重点在统一 action / image / state / text 的理解与生成。

后续可以追踪的问题 / 论文：

1. WorldVLA
    用来理解 RynnVLA-002 的前身：为什么要统一 action 和 image generation，以及 action attention mask 是怎么来的。

2. π0 / π0.5
    用来理解 continuous action head、flow matching 和真实机器人控制中的动作平滑问题。

3. DreamZero / World Action Models
    用来理解更大规模 WAM 路线：视频扩散模型如何作为机器人 policy 的物理动态先验。

4. ACT / Diffusion Policy
    用来补齐 action chunking、连续控制、轨迹平滑和真实机器人 imitation learning 的基础。

5. 后续如果要复现，建议优先看：
    - GitHub 训练脚本；
    - LIBERO evaluation；
    - SO 100 数据格式；
    - continuous Action Transformer 部分；
    - world model ablation。
