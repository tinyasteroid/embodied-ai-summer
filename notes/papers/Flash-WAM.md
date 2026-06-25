---
title: "Flash-WAM: Modality-Aware Distillation for World Action Models"
year: 2026
lab_company:
  - Northeastern University
  - University of Georgia
  - EmbodyX Inc.
paper_type: "WAM acceleration / diffusion distillation / real-time robot policy"
field:
  - embodied_ai
  - robot_learning
  - world_action_model
method_tags:
  - WAM
  - step_distillation
  - consistency_distillation
  - flow_matching
  - diffusion_policy
  - modality_aware_distillation
  - real_time_control
  - video_action_generation
task_tags:
  - bimanual_manipulation
  - humanoid_manipulation
  - simulation_benchmark
  - real_robot
  - long_horizon
paper_url: "https://arxiv.org/abs/2606.05254"
project_url: "https://flashwam.github.io/"
code_url: ""
dataset_url: ""
checkpoint_url: ""
---

# Flash-WAM: Modality-Aware Distillation for World Action Models

## 1. 一句话总结

这篇论文主要解决：**World Action Models（WAM）虽然能通过“预测未来视频 + 生成对应动作”获得较强的物理动态建模能力，但 diffusion denoising 步数太多，导致推理速度过慢、难以实时闭环控制。Flash-WAM 通过 modality-aware step distillation，把 video stream 和 action stream 分别采用不同的 consistency function 进行蒸馏，将 WAM 的 video/action 生成压缩到 1–2 步，从而实现接近实时的机器人控制。**

更直观地说：**Flash-WAM 不是提出一个新的 VLA/WAM 架构，而是解决“WAM 太慢，如何部署到真实机器人上”的问题。**

## 2. 核心贡献

### 2.1 核心贡献 1：指出普通 consistency distillation 不能直接用于 WAM

Flash-WAM 首先分析了为什么将 LCM / consistency distillation 这类图像或视频生成加速方法直接套到 WAM 上会失败。

WAM 同时生成两类对象：

- video latent：高维、冗余、能容忍较大噪声；
- action sequence：低维、精度敏感、低噪声区域的学习信号非常关键。

由于 video 和 action 使用不同的 SNR-shifted noise schedules，它们在训练时落入不同的 noise regime。普通 LCM 使用统一的 consistency function，会导致 action stream 在 low-noise 区域梯度信号过弱，从而造成动作蒸馏失败。

核心结论：**WAM 是 joint video-action diffusion，不是普通单模态 diffusion，不能用同一个蒸馏函数同时处理 video 和 action。**

### 2.2 核心贡献 2：提出 modality-aware consistency distillation

Flash-WAM 的方法是为不同模态选择不同的 consistency function：

- 对 video stream：使用适合 high-noise regime 的 variance-preserving parametrization；
- 对 action stream：使用适合 low-noise regime 的 linear-gradient-scaling parametrization。

其背后的直觉是：

- 视频生成更关心高噪声下的稳定性和视觉结构保持；
- 动作生成更关心低噪声下的精度，因此需要避免低噪声区域梯度消失。

这使得 student model 可以在保持原始 WAM 推理结构的情况下，将多步 video/action diffusion 压缩到少步甚至单步推理。

### 2.3 核心贡献 3：实现接近实时的 WAM 推理

论文将 Flash-WAM 应用于 LingBot-VA。原始 LingBot-VA 在 RoboTwin 2.0 上每个 action chunk 需要：

- video denoising：25 steps；
- action denoising：50 steps；
- 总延迟约 8.1 s。

Flash-WAM 将其压缩为：

- 1 video step + 2 action steps，或
- 1 video step + 1 action step。

主要结果：

- RoboTwin 2.0：1 v/2 a 达到 85.5% success rate；
- LIBERO：1 v/2 a 达到 95.7% success rate；
- Unitree G 1 真实机器人：1 v/2 a 平均成功率 60%；
- 推理延迟从 8.1 s 降到约 348 ms，实现约 23× speedup。

## 3. 数据与任务设置

### 3.1 Data

这篇论文的重点不是重新收集大规模机器人数据，而是在已有 WAM teacher model 上做 step distillation。

主要依赖：

- Teacher model：LingBot-VA，作为被蒸馏的原始 WAM；
- Simulation evaluation：
    - RoboTwin 2.0；
    - LIBERO；
- Real-world evaluation：
    - Unitree G 1 humanoid robot；
    - 每个真实任务收集 50 条 teleoperated demonstrations；
    - 每个方法每个任务评估 10 次 independent rollouts。

注意：论文的核心不是数据规模或数据构造，而是 **如何把已有 WAM 的多步 diffusion 推理压缩到少步/单步推理**。

### 3.2 Robot / Embodiment

论文涉及两类主要 embodiment：

1. Simulation：
    - RoboTwin 2.0 中的双臂操作任务；
    - LIBERO 中的机器人操作任务。
2. Real Robot：
    - Unitree G 1 humanoid robot；
    - 使用 Unitree Dex 1-1 grippers；
    - 真实任务包括开锅盖放土豆、在干扰物中拿红瓶、拿粉色物体并放到标记位置等。

### 3.3 Simulation or Real Robot

两者都有。

Simulation：

- RoboTwin 2.0：
    - 50 个 manipulation tasks；
    - Clean split；
    - Randomized split；
    - 用于评估任务成功率和对随机扰动的鲁棒性。
- LIBERO：
    - Spatial；
    - Object；
    - Goal；
    - Long-horizon。

Real Robot：

- Unitree G 1；
- 3 个真实操作任务；
- 每个任务 10 次 rollout。

### 3.4 Task

任务类型主要是 manipulation，包括：

- bimanual manipulation；
- object picking；
- distractor handling；
- goal-conditioned placement；
- long-horizon manipulation；
- humanoid robot manipulation。

从任务角度看，Flash-WAM 不强调“新任务能力”，而强调 **在保留原始 WAM 操作能力的同时显著降低推理延迟**。

## 4. 模型结构

### 4.1 Vision encoder

Flash-WAM 本身不重新设计 vision encoder，而是继承 LingBot-VA 这类 WAM 的已有视觉/视频生成 backbone。

论文关注的是 video/action diffusion stream 的蒸馏，而不是视觉编码器结构。

可以理解为：

```text
原始 WAM 的视觉/视频生成模块保持不变
→ Flash-WAM 在其上做 step distillation
→ 让 video denoising 从多步变成 1 步
```

### 4.2 Language model

语言指令作为 context 的一部分输入 WAM。

Flash-WAM 没有重点讨论新的 language model 或 language encoder 设计，语言主要作为已有 LingBot-VA/WAM 中的条件信息存在。

简化理解：

```text
past observations + past actions + language instruction
→ context C
→ 生成未来 video latents
→ 生成对应 action sequence
```

### 4.3 Action representation

动作被建模为 action sequence / action chunk。

在 WAM 中，动作生成不是普通 VLA 那种直接从图像和语言输出动作，而是：

```text
先预测未来视觉状态
→ 再生成与未来视觉变化一致的动作
```

论文将这一过程称为 inverse dynamics，即从预测的视觉变化中恢复对应动作。

动作流的特点：

- 低维；
- 精度敏感；
- 对 low-noise region 的梯度信号要求高；
- 普通 LCM 蒸馏容易导致动作质量崩溃。

### 4.4 Action decoder / head

Flash-WAM 不主要修改 action decoder/head，而是修改 action stream 的 distillation parametrization。

核心设计：

$$
f_a(x_\sigma, \sigma) = x_\sigma - \sigma v_\theta(x_\sigma, \sigma)
$$

其意义是让 action stream 在 low-noise regime 中保持线性梯度缩放，避免普通 LCM 在 sigma 接近 0 时梯度二次消失。

直观解释：

- 普通 LCM：低噪声下动作学习信号太弱；
- Flash-WAM action parametrization：低噪声下仍保留足够梯度；
- 结果：动作蒸馏不会 collapse。

### 4.5 World model component

Flash-WAM 面向的是 World Action Model，而不是普通 VLA。

WAM 的核心建模形式：

$$
\begin{aligned}
x_v &\sim p_\theta(x_v \mid C) \\
x_a &\sim p_\theta(x_a \mid x_v, C)
\end{aligned}
$$

其中：

- $x_v$ 是未来 video latents；
- $x_a$ 是对应的 action sequence；
- $C$ 包含历史 observation、历史 action 和 language instruction。

这说明 WAM 的动作不是孤立生成的，而是 grounded in predicted future states。

Flash-WAM 保留这个 WAM 推理结构，只是把两个 diffusion 过程压缩为少步推理。

### 4.6 Latent passing / intermediate representation

WAM 的中间表示是 predicted future video latents。

推理流程可以概括为：

```text
历史观测 + 历史动作 + 语言指令
→ context C
→ video diffusion 预测未来 video latents
→ action diffusion 根据未来 video latents 生成 action chunk
→ 执行动作
```

Flash-WAM 的改动在于：

```text
原始 WAM:
video denoising 多步 + action denoising 多步

Flash-WAM:
video denoising 1 step + action denoising 1/2 steps
```

它不是删除 video generation，而是保留“未来视频作为动作生成依据”的原始 WAM 结构，并通过 distillation 加速。

## 5. Evaluation

### 5.1 Benchmark

主要 benchmark：

1. RoboTwin 2.0
    - 50 个任务；
    - Clean split；
    - Randomized split；
    - 评估 bimanual manipulation 和 domain randomization 下的鲁棒性。
2. LIBERO
    - Spatial；
    - Object；
    - Goal；
    - Long-horizon；
    - 评估不同泛化维度下的 success rate。
3. Unitree G 1 real-world evaluation
    - 3 个真实机器人操作任务；
    - 每个任务 10 次 rollout。

### 5.2 Real robot

真实机器人平台：

- Unitree G 1 humanoid robot；
- Unitree Dex 1-1 grippers。

任务：

- T 1：打开锅盖并把土豆放入锅中；
- T 2：在黄色干扰瓶存在的情况下拿起红瓶；
- T 3：拿起粉色物体并放到目标标记位置。

主要结果：

```text
LingBot-VA teacher, 3v/10a:
Average = 66.7%

Flash-WAM, 1v/2a:
Average = 60.0%

Flash-WAM, 1v/1a:
Average = 50.0%
```

直观理解：真实机器人上性能有下降，但 Flash-WAM 在大幅减少 denoising steps 的情况下，仍恢复了大部分 teacher performance。

### 5.3 Simulation

#### RoboTwin 2.0

主要结果：

```text
LingBot-VA teacher, 25v/50a:
Average = 91.25%

Flash-WAM, 1v/2a:
Average = 85.54%
Speedup = 19.0×

Flash-WAM, 1v/1a:
Average = 81.41%
Speedup = 23.3×
```

对比普通方法：

```text
Naive Joint LCM, 1v/2a:
Average = 23.97%

Naive Joint LCM, 1v/1a:
Average = 36.32%
```

这说明普通 joint LCM 会明显 collapse，而 Flash-WAM 的 modality-aware design 是必要的。

#### LIBERO

主要结果：

```text
LingBot-VA teacher, 20v/50a:
Average = 98.6%

Flash-WAM, 1v/2a:
Average = 95.7%
Speedup = 13.7×

Flash-WAM, 1v/1a:
Average = 95.1%
Speedup = 16.3×
```

LIBERO 上性能保持更好，说明 Flash-WAM 在标准仿真 benchmark 上可以较稳定地保留 teacher 的能力。

### 5.4 Main metric

主要指标：

- Success Rate；
- Average Success Rate；
- Per-chunk inference latency；
- Speedup；
- NFE configuration：
    - $25v/50a$ 表示 25 video denoising steps + 50 action denoising steps；
    - $1v/2a$ 表示 1 video step + 2 action steps；
    - $1v/1a$ 表示 video/action 都单步推理。

论文关注的核心 trade-off：

```text
减少 denoising steps
→ 降低 per-chunk latency
→ 尽量保持 success rate
```

### 5.5 Baseline

主要 baselines：

- LingBot-VA teacher；
- Naive Joint LCM；
- Video-only LCM；
- DMD 2；
- π0；
- π0.5；
- X-VLA；
- Motus。

其中最关键的对比是：

```text
Flash-WAM vs Naive Joint LCM
```

因为它直接验证了论文主张：**video/action joint diffusion 不能用单一 consistency function 统一蒸馏，必须 modality-aware。**

## 6. 我的理解与疑问

我认为这篇论文最值得记住的是：

**Flash-WAM 是 WAM 走向实际部署的一篇系统加速论文。它不主要解决“机器人怎么泛化”，而是解决“已经较强的 WAM 怎么实时跑”。其核心洞察是 video 和 action 虽然都在 diffusion 框架里生成，但二者的噪声分布、精度要求、梯度需求完全不同，因此需要分模态蒸馏。**

对我理解具身智能最有用的点：

1. **WAM 的推理瓶颈很现实。**
    如果每个 action chunk 需要数秒生成，即使模型泛化能力强，也很难用于闭环控制。

2. **video generation 和 action generation 的误差性质不同。**
    视频可以“看起来差不多”，但动作必须精确，否则机器人会直接失败。

3. **WAM 加速不能只靠工程优化。**
    KV cache、异步执行、CUDA 优化有用，但根本问题还是 denoising steps 太多。Flash-WAM 直接压缩 diffusion steps，因此是和系统优化互补的方向。

4. **Flash-WAM 和 DreamZero-Flash 不能混淆。**
    DreamZero-Flash 是 DreamZero 内部通过 decoupled denoising schedule 做少步推理优化；Flash-WAM 是一篇独立论文，用 modality-aware consistency distillation 蒸馏已有 WAM。

我还没理解清楚的是：

- consistency function family 的理论推导是否真的对所有 WAM 都成立，还是主要针对 LingBot-VA 的噪声调度和结构？
- Flash-WAM 对更大规模 WAM，例如 DreamZero 或 Motus，是否同样有效？
- 单步 video generation 质量下降时，action stream 是否会在更复杂长程任务中受到更明显影响？
- 真实机器人实验只有 3 个任务，每个任务 10 次 rollout，是否足以证明真实部署稳定性？
- Flash-WAM 是否开源了代码和 checkpoint？当前论文首页只明确给出 project page，代码仓库需要进一步确认。

和我已读论文的关系：

- 和 **RT-2 / OpenVLA** 的关系：
    RT-2 / OpenVLA 是典型 VLA，直接从视觉和语言预测动作；Flash-WAM 面向的是 WAM，即先预测未来视觉状态，再生成对应动作。Flash-WAM 不是 VLA 主线，而是 WAM 部署加速主线。

- 和 **π0** 的关系：
    π0 使用 flow matching 生成连续 action chunk，重点是高频、连续、灵巧控制；Flash-WAM 也涉及 flow matching / diffusion，但它关注的是 joint video-action WAM 的 step distillation，而不是重新设计动作头。

- 和 **WorldVLA / RynnVLA-002** 的关系：
    WorldVLA / RynnVLA-002 关注 VLA 和 world model 如何统一建模；Flash-WAM 假设已有 WAM，进一步解决推理速度过慢的问题。

- 和 **DreamZero / WAM** 的关系：
    DreamZero 展示 WAM 可以通过视频预测获得更强的物理先验、zero-shot generalization 和 cross-embodiment transfer；Flash-WAM 则回答“这样的 WAM 如何压缩到实时推理”。

后续可以追踪的问题 / 论文：

- LingBot-VA：Flash-WAM 的 teacher model，需要理解其 WAM 架构；
- DreamZero：理解更大规模 WAM 如何利用 video diffusion backbone；
- Motus：另一个 joint video-action WAM 系统；
- Fast-WAM / GigaWorld-Policy：与 Flash-WAM 不同，它们尝试减少或避免 test-time video generation；
- LCM / consistency models：理解 Flash-WAM 的 distillation 基础；
- DMD 2 / progressive distillation：对比不同 diffusion acceleration 方法；
- DreamZero-Flash：与 Flash-WAM 对比，理解 decoupled denoising schedule 和 modality-aware distillation 的区别。
