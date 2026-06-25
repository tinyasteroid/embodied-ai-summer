---
title: "Causal World Modeling for Robot Control"
year: 2026
lab_company:
  - Robbyant
paper_type: "WAM / robot foundation model / causal world model"
field:
  - embodied_ai
  - robot_learning
  - world_model
  - vla
method_tags:
  - LingBot-VA
  - World Action Model
  - autoregressive_diffusion
  - flow_matching
  - Mixture-of-Transformers
  - inverse_dynamics
  - closed_loop_rollout
  - asynchronous_inference
task_tags:
  - long_horizon_manipulation
  - precision_manipulation
  - deformable_object_manipulation
  - bimanual_manipulation
  - simulation_benchmark
paper_url: "https://arxiv.org/abs/2601.21998v2"
project_url: "https://technology.robbyant.com/lingbot-va"
code_url: "https://github.com/robbyant/lingbot-va"
dataset_url: ""
checkpoint_url: "https://huggingface.co/robbyant/lingbot-va"
---

# Causal World Modeling for Robot Control

## 1. 一句话总结

这篇论文主要解决：传统 VLA 直接从当前 observation 映射到 action，容易缺少显式物理动态建模、长时记忆和闭环纠错能力；作者提出 **LingBot-VA**，用自回归 diffusion world model 同时学习未来视觉状态预测和动作执行，让机器人先“想象世界如何变化”，再通过 inverse dynamics 推断应该执行的动作。

## 2. 核心贡献

### 2.1 核心贡献 1：提出 LingBot-VA，自回归 Video-Action World Model

LingBot-VA 将机器人控制建模为 **visual dynamics prediction + inverse dynamics**：

```text
观测历史 + 动作历史 + 语言指令
→ 预测未来视觉 latent
→ 根据预测的视觉变化解码动作
→ 执行动作并接收真实环境反馈
```

和普通 VLA 的区别是，它不是只学习 $\text{observation} \rightarrow \text{action}$ 的 reactive mapping，而是显式学习“动作如何导致视觉世界变化”，再从这种变化中推出可执行动作。

### 2.2 核心贡献 2：用 MoT 统一 video tokens 和 action tokens

模型采用 **Mixture-of-Transformers, MoT** 架构，把 video stream 和 action stream 放到统一的自回归序列中建模。Video stream 初始化自 Wan 2.2-5 B 视频生成模型，负责捕捉视觉动态；action stream 更窄，负责低维动作生成。两者通过 shared attention / cross-modal fusion 交互，从而实现视觉未来预测和动作推断的对齐。

### 2.3 核心贡献 3：面向真实机器人部署的闭环与异步推理

论文不仅提出模型结构，还设计了实际部署方案：

- **Closed-loop rollout**：每轮执行后用真实 observation 更新上下文，缓解 open-loop drift。
- **KV cache**：保留历史 video-action trajectory，提高长时记忆。
- **Noisy History Augmentation**：允许推理时部分去噪，降低未来视频生成成本。
- **Asynchronous inference**：机器人执行当前 action chunk 时，模型并行预测下一段 video/action chunk，提高控制效率。

## 3. 数据与任务设置

### 3.1 Data

论文涉及两类数据来源：

1. **Pretraining data**
    - diverse in-the-wild videos
    - robot action data
    - 目标是让模型同时获得视频动态先验和机器人动作对齐能力。
2. **Post-training / finetuning data**
    - 真实机器人任务中，每个任务使用少量 demonstrations，例如论文中强调 50 demonstrations 下的真实任务适配。
    - RoboTwin 2.0 中使用 clean scene demonstrations 和 heavily randomized scene demonstrations。
    - LIBERO 中每个 suite 包含 10 个任务，每个任务 50 demonstrations。

### 3.2 Robot / Embodiment

论文实验覆盖：

- 真实机器人操作任务；
- RoboTwin 2.0 中的双臂 manipulation setting；
- LIBERO 中的标准机器人操作 benchmark。

具体真实机器人平台细节在当前笔记中暂不展开，需要后续读实验附录或代码进一步核验。

### 3.3 Simulation or Real Robot

两者都有。

**Real robot：**

- long-horizon manipulation
- precision manipulation
- deformable object manipulation

**Simulation：**

- RoboTwin 2.0
- LIBERO-Spatial
- LIBERO-Object
- LIBERO-Goal
- LIBERO-Long

### 3.4 Task

真实机器人任务大致分为三类：

1. **Long-horizon**
    - Make Breakfast
    - Unpack Delivery
2. **Precision**
    - Insert Tubes
    - Pick Screws
3. **Deformable objects**
    - Fold Clothes
    - Fold Pants

仿真任务：

- RoboTwin 2.0：50 个双臂操作任务，按 horizon 分析。
- LIBERO：Spatial / Object / Goal / Long 四个 suites。

## 4. 模型结构

### 4.1 Vision encoder

模型使用 causal video VAE 将视觉 observation 编码到 latent space：

$$
o_t \rightarrow z_t
$$

其中 $z_t$ 是视觉 latent tokens。这样做是为了避免直接在 pixel space 上建模未来视频，降低计算成本，并让视频生成模型可以在连续 latent space 中进行 flow matching。

### 4.2 Language model

语言指令作为 task prompt 输入模型，用于约束未来视觉预测和动作生成。

这篇论文不是以 LLM/VLM 作为唯一主干的传统 VLA，而是更强调 video diffusion backbone 和 video-action world modeling。语言模块主要提供任务条件，而不是像 OpenVLA / RT-2 那样把动作视为语言 token 输出。

### 4.3 Action representation

动作是连续控制信号，不是离散 action token。

论文将 action vector 通过轻量 MLP 投影成 action token embedding，使其能够和 video latent tokens 一起进入统一序列：

$$
a_t \rightarrow \text{action embedding}
$$

由于动作频率通常高于视频帧变化，论文采用 video sparsification：视频帧下采样，而每个视频帧关联多个连续动作。形式上可理解为：

$$
[z_t, a_{t,1}, a_{t,2}, \ldots, a_{t,\tau}, z_{t+1}, \ldots]
$$

这使得模型可以在较低视频生成频率下维持较高动作控制频率。

### 4.4 Action decoder / head

动作生成通过 **inverse dynamics** 完成。

模型先预测未来视觉 latent，再根据预测的视觉变化、历史 observation 和历史 action 解码动作：

$$
\hat{z}_{t+1:t+K},\ z_{\le t},\ a_{<t} \rightarrow a_{t:t+K-1}
$$

直观理解：模型不是直接问“当前图像下该输出什么动作”，而是问“如果未来应该变成这样，那么当前应该执行什么动作”。

### 4.5 World model component

World model component 是 LingBot-VA 的核心。

它根据 observation history 和 action history 预测未来一段视觉 latent：

$$
z_{t+1:t+K} \sim p_\theta(\cdot \mid z_{\le t}, a_{<t})
$$

这里的 world model 不是外部 simulator，也不是单独训练的 reward model，而是和 action model 共同训练、共同推理的 video-action generative model。

### 4.6 Latent passing / intermediate representation

核心 intermediate representation 是 **future visual latent**。

流程可以概括为：

```text
observation
→ causal video VAE
→ visual latent z_t
→ autoregressive video-action MoT
→ predicted future visual latent z_hat
→ inverse dynamics action decoder
→ action chunk
→ robot execution
→ new real observation feedback
```

这篇论文最值得注意的 latent passing 是：

1. video latent 不只是用于生成可视化视频，而是作为动作生成的动态表征；
2. action tokens 和 video tokens 被交织到同一序列中，因此两种模态可以互相 condition；
3. closed-loop rollout 会用真实 observation 替换或更新预测上下文，减少长程误差积累。

## 5. Evaluation

### 5.1 Benchmark

主要 benchmark：

- **RoboTwin 2.0**
    - 50 个双臂 manipulation 任务。
    - Easy：固定初始配置。
    - Hard：随机物体姿态和场景布局。
    - 论文按 horizon = 1 / 2 / 3 分析长程能力。
- **LIBERO**
    - LIBERO-Spatial
    - LIBERO-Object
    - LIBERO-Goal
    - LIBERO-Long

### 5.2 Real robot

真实机器人实验覆盖 6 个任务：

|类型|任务|主要考察点|
|---|---|---|
|Long-horizon|Make Breakfast, Unpack Delivery|长时记忆、阶段推进、闭环纠错|
|Precision|Insert Tubes, Pick Screws|视觉-动作细粒度对齐|
|Deformable|Fold Clothes, Fold Pants|对非刚体动态的预测和动作规划|

论文结论是：LingBot-VA 在这些真实任务上相对 π0.5 有更好的 success rate / progress score，尤其在长程、精细操作和柔性物体任务上体现出 world modeling 的优势。

### 5.3 Simulation

**RoboTwin 2.0：**

- Average 50 Tasks：
    - Easy：LingBot-VA 约 92.93%
    - Hard：LingBot-VA 约 91.55%
- Horizon = 3 上提升更明显，说明自回归机制和 KV cache 对长程任务有帮助。

**LIBERO：**

- Spatial：98.5%
- Object：99.6%
- Goal：97.2%
- Long：98.5%
- Average：98.5%

### 5.4 Main metric

主要指标：

- **Success Rate**
- **Progress Score**
- **Task completion speed / efficiency**
- **Ablation success rate**
- **Sample efficiency under different number of demonstrations**

其中 Success Rate 用于衡量任务是否完成；Progress Score 更适合长程任务，因为即使任务没完全成功，也能反映模型推进到哪个阶段。

### 5.5 Baseline

主要 baseline：

- π0
- π0.5
- X-VLA
- Motus
- WAN / Wan 2.2-5 B fine-tuning baseline
- synchronous vs asynchronous variants
- naive async variant

对比重点不是只看平均成功率，而是看：

1. 是否在 long-horizon task 上更稳；
2. 是否在少量 demonstrations 下更高效；
3. 是否在 hard / randomized setting 下保持泛化；
4. 异步推理是否显著牺牲性能。

## 6. 我的理解与疑问

我认为这篇论文最值得记住的是：

**LingBot-VA 把 WAM 路线讲得很清楚：video generation 不是为了生成“好看的视频”，而是为了学习动作相关的世界动态表征。**
它的核心范式是：

```text
不是 observation → action
而是 observation/action history → future visual latent → action
```

这意味着机器人策略不再只是模仿当前帧下的专家动作，而是尝试学习“动作会让世界如何变化”，再根据这种变化推断动作。这是 VLA 到 WAM 的关键范式变化。

我还没理解清楚的是：

1. **test-time future imagination 到底有多必要？**
    LingBot-VA 在推理时显式生成未来 visual latent，但 Fast-WAM 这类工作认为视频建模的主要价值可能在训练阶段，而不是推理阶段真的生成未来。

2. **论文中的 causal 更偏工程结构还是物理因果？**
    当前理解中，它主要指 causal attention / autoregressive factorization / 真实反馈闭环，而不是严格因果推断意义上的 causal discovery。

3. **Noisy History Augmentation 的边界在哪里？**
    如果未来 latent 太 noisy，action decoder 是否仍然可靠？不同任务是否需要不同程度的 partial denoising？

4. **异步推理是否会在真实强扰动场景下失效？**
    FDM-grounded async 能缓解 naive async 的退化，但如果机器人执行过程中发生未建模扰动，预测上下文是否还能可靠纠偏？

和我已读论文的关系：

- **RT-2 / OpenVLA**：代表传统 VLA 路线，核心是从 vision-language backbone 迁移语义能力到 action prediction。它们更偏 $\text{image + language} \rightarrow \text{action}$。
- **π0 / π0.5**：用 flow matching action head 生成连续动作，强调高频、流畅、灵巧控制，但不是显式 video world model。
- **WorldVLA / RynnVLA-002**：也尝试统一 action model 和 world model，但更偏多模态 token 统一建模，部分版本使用离散 action token 或额外 Action Transformer。
- **DreamZero / WAM**：同属 WAM 路线，强调 joint video-action prediction、zero-shot generalization 和 cross-embodiment transfer。
- **LingBot-VA**：更强调 causal autoregressive video-action world modeling、KV cache 长时记忆、closed-loop rollout 和异步部署。
- **Fast-WAM**：后续可以用来反问 LingBot-VA：推理时真的需要生成未来视频吗？
- **Flash-WAM**：后续可以用来解决 LingBot-VA 的推理延迟问题，对 WAM 做 step distillation。

后续可以追踪的问题 / 论文：

1. **WAM 是否真的需要 test-time future imagination？**
    - Fast-WAM
2. **WAM 如何做到实时控制？**
    - Flash-WAM
    - DreamZero-Flash 类优化
3. **World model 和 VLA 应该统一训练还是分离训练？**
    - LingBot-VA
    - DreamZero
    - WorldVLA
    - RynnVLA-002
    - Motus
4. **视频预测质量和 action success rate 是否强相关？**
    - 这决定了继续扩大 video backbone 是否真的有价值。
5. **真实机器人上的泛化是否来自 video prior、robot data scale，还是 closed-loop deployment？**
    - 后续读论文时需要重点看 ablation，而不是只看主结果。
