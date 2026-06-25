---
title: "Fast-WAM: Do World Action Models Need Test-time Future Imagination?"
year: 2026
lab_company:
  - "IIIS, Tsinghua University"
  - Galaxea AI
paper_type: "WAM / efficient robot foundation model / representation learning"
field:
  - embodied_ai
  - robot_learning
  - world_action_model
method_tags:
  - WAM
  - video_co_training
  - test_time_future_imagination
  - latent_world_representation
  - flow_matching
  - DiT
  - MoT
  - action_expert
  - real_time_control
task_tags:
  - robot_manipulation
  - bimanual_manipulation
  - towel_folding
  - LIBERO
  - RoboTwin
paper_url: "https://arxiv.org/abs/2603.16666"
project_url: "https://yuantianyuan01.github.io/FastWAM/"
code_url: ""
dataset_url: ""
checkpoint_url: ""
---

# Fast-WAM: Do World Action Models Need Test-time Future Imagination?

## 1. 一句话总结

这篇论文主要解决：**World Action Model（WAM）是否必须在测试时显式生成未来视频才能获得强动作性能**。论文提出 **Fast-WAM**，在训练时保留 future video prediction 作为 world-modeling 辅助目标，但在推理时跳过 future video generation，直接用视频 DiT 单次前向得到的 latent world representation 生成动作，从而显著降低推理延迟，同时保持接近 imagine-then-execute WAM 的性能。

更直观地说：

> Fast-WAM 的核心观点是：WAM 的收益可能主要来自“训练时学会预测未来”，而不是“推理时真的生成未来”。

---

## 2. 核心贡献

### 2.1 核心贡献 1：重新拆解 WAM 的收益来源

现有 WAM 通常把两个因素绑定在一起：

1. **Training-time video modeling**：训练时预测未来视频，让模型学习物理动态和时序结构。
2. **Test-time future imagination**：推理时显式生成未来视频，再根据未来视频生成动作。

Fast-WAM 的核心问题是：
**WAM 的性能提升到底来自训练阶段的视频建模，还是来自推理阶段的未来想象？**

论文通过 controlled variants 把这两个因素拆开比较，发现：

- 保留 video co-training，但取消 test-time future generation，性能仍然很强；
- 去掉 video co-training 后，性能下降更明显；
- 因此，WAM 的主要收益可能更多来自训练阶段的 world representation learning。

### 2.2 核心贡献 2：提出 Fast-WAM，推理时不再显式生成未来视频

传统 imagine-then-execute WAM 通常是：

```text
当前图像 + 语言指令 → 生成未来视频 → 根据未来视频生成动作
```

Fast-WAM 改成：

```text
当前图像 + 语言指令 → 视频 DiT 单次编码 → latent world representation → action expert 生成动作
```

也就是说，Fast-WAM 训练时仍然做 video prediction，但推理时不 denoise future video tokens，不显式生成未来帧。

这使它的推理接口更接近普通 VLA：

$$
\text{observation + language} \rightarrow \text{action}
$$

但它的训练信号又不同于普通 VLA，因为它额外受到 future video prediction objective 的约束。

### 2.3 核心贡献 3：在保持性能的同时显著降低推理延迟

Fast-WAM 在 simulation benchmark 和 real-world task 上表现接近强 WAM baseline，同时推理延迟明显降低。

核心结果：

- RoboTwin 2.0：Fast-WAM 达到 **91.8% average success rate**，接近带 embodied pretraining 的 LingBot-VA **92.2%**。
- LIBERO：Fast-WAM 达到 **97.6% average success rate**。
- Real-world towel folding：Fast-WAM latency 约 **190 ms**，明显低于 Fast-WAM-Joint 的 **580 ms** 和 Fast-WAM-IDM 的 **810 ms**。
- 去掉 video co-training 后，real-world towel folding 成功率降到 **10%**，说明 video co-training 是关键因素。

---

## 3. 数据与任务设置

### 3.1 Data

论文没有做大规模 embodied pretraining，而是在具体 benchmark 和真实任务上训练评估。

主要数据来源：

1. **LIBERO**
    - 包含 LIBERO-Spatial、LIBERO-Object、LIBERO-Goal、LIBERO-Long 四个 suite。
    - 每个 suite 10 个任务。
    - 每个 suite 500 条 demonstrations。
    - 总计 40 个任务，评估 2000 trials。
2. **RoboTwin 2.0**
    - 双臂操作 benchmark。
    - 超过 50 个任务。
    - 训练数据包括：
        - 2,500 条 clean scene demonstrations；
        - 25,000 条 heavy scene randomization demonstrations。
    - 评估 clean 和 randomized 两种设置。
3. **Real-world towel folding**
    - 平台：Galaxea R 1 Lite。
    - 数据：60 小时遥操作 demonstrations。
    - 任务：折叠毛巾 / 布料，属于长程、可变形物体操作任务。

### 3.2 Robot / Embodiment

- Simulation：
    - LIBERO 中的机器人操作环境。
    - RoboTwin 2.0 中的双臂机器人操作环境。
- Real Robot：
    - Galaxea R 1 Lite。
    - 任务是 long-horizon towel folding。

### 3.3 Simulation or Real Robot

两者都有：

- Simulation：
    - LIBERO
    - RoboTwin 2.0
- Real Robot：
    - Galaxea R 1 Lite towel folding

这点比较重要，因为论文不仅在仿真中验证“去掉 test-time future imagination 是否可行”，也在真实长程布料操作任务中验证 latency 和 task efficiency。

### 3.4 Task

任务类型主要包括：

- 语言条件下的机器人操作；
- 双臂 manipulation；
- 长程任务；
- 可变形物体操作；
- OOD / randomized scene 下的泛化；
- action chunk generation。

论文为了 controlled comparison，主要聚焦 **single action chunk generation**，没有展开完整 outer autoregressive rollout。

---

## 4. 模型结构

### 4.1 Vision encoder

Fast-WAM 使用 **Wan 2.2-5 B** 中的视频生成组件作为视觉和世界建模 backbone。

具体包括：

- pretrained video VAE：把图像观测编码为 latent video tokens；
- video DiT：作为 world modeling backbone；
- 当前观测帧被编码成 clean first-frame latent tokens，作为共享视觉锚点。

推理时，模型只保留当前帧 latent tokens，不再实例化 future noisy video tokens。

### 4.2 Language model

语言部分使用 Wan 2.2-5 B 内置的 **T 5 text encoder**。

语言指令通过 cross-attention 提供给 video tokens 和 action tokens。

可以理解为：

```text
language instruction → T5 encoder → cross-attention condition
```

论文没有把语言部分作为主要创新点，重点在于 video DiT 和 action expert 的结构设计。

### 4.3 Action representation

动作以 **action chunk** 的形式建模。

设当前观测为 $o$，语言指令为 $l$，动作 chunk 为：

$$
a_{1:H}
$$

论文中 action horizon 设置为：

$$
H = 32
$$

训练时对 action chunk 使用 flow matching 目标。

### 4.4 Action decoder / head

Fast-WAM 引入 **action expert DiT** 来生成动作 chunk。

结构上属于 **Mixture-of-Transformer（MoT）**：

```text
Video DiT branch + Action Expert DiT branch + shared attention
```

实现细节：

- video branch 来自 Wan 2.2-5 B；
- action expert 使用与 video branch 类似的结构，但 hidden dimension 更小；
- action expert hidden dimension 为 1024；
- action expert 约 1 B 参数；
- 总模型约 6 B 参数。

动作生成采用 flow matching：

$$
\begin{aligned}
y_t &= (1 - t)y + t\epsilon \\
L_{\mathrm{FM}}(y) &= \mathbb{E}\left[\left\lVert f_\theta(y_t, t, o, l) - (\epsilon - y) \right\rVert^2\right]
\end{aligned}
$$

其中 $y$ 可以是 action chunk，也可以是 future video latent。

### 4.5 World model component

Fast-WAM 的 world model component 不是在推理时显式生成未来视频，而是在训练阶段提供辅助监督。

训练时：

```text
当前帧 latent + 语言指令 → 预测 future video latents
```

推理时：

```text
当前帧 latent + 语言指令 → video DiT 单次前向 → latent world representation
```

因此，Fast-WAM 中的 world model 更像是 **representation shaping module**，而不是 deployment-time visual planner。

这和 DreamZero / LingBot-VA 这类 imagine-then-execute WAM 不同。后者在推理时仍然显式生成未来视觉状态，而 Fast-WAM 只在训练时利用视频预测目标。

### 4.6 Latent passing / intermediate representation

Fast-WAM 的关键中间表征是：

$$
z(o,l)
$$

它表示 video backbone 基于当前观测和语言生成的 latent world representation。

传统 WAM 通常建模：

$$
p(a_{1:H} \mid o,l) = \int p(v_{1:T} \mid o,l)\,p(a_{1:H} \mid o,l,v_{1:T})\,dv_{1:T}
$$

Fast-WAM 改为：

$$
p_\theta(a_{1:H} \mid o,l) = p_\theta(a_{1:H} \mid z(o,l))
$$

核心区别：

- 传统 WAM：$z$ 来自显式采样 / denoising 未来视频；
- Fast-WAM：$z$ 来自单次前向编码，不生成未来帧。

#### Attention mask 设计

Fast-WAM 把 token 分成三组：

1. clean first-frame latent tokens；
2. noisy future video tokens；
3. action tokens。

训练时：

- future video tokens 可以看 clean first-frame tokens；
- action tokens 可以看 clean first-frame tokens；
- action tokens 可以在 action branch 内部 bidirectional attention；
- **action tokens 不能看 future video tokens**；
- clean first-frame tokens 不看其他 tokens。

这个设计非常关键，因为它防止 action branch 在训练时“偷看未来视频”。
如果 action tokens 能看到 future video tokens，那么推理时去掉 future video branch 会造成训练-推理不一致。

---

## 5. Evaluation

### 5.1 Benchmark

主要 benchmark：

1. **LIBERO**
    - LIBERO-Spatial
    - LIBERO-Object
    - LIBERO-Goal
    - LIBERO-Long
2. **RoboTwin 2.0**
    - 双臂操作任务
    - clean / randomized 两类评估设置
    - 50+ tasks
3. **Real-world towel folding**
    - Galaxea R 1 Lite
    - 长程可变形物体操作任务

### 5.2 Real robot

真实机器人实验是 towel folding。

设置：

- Robot：Galaxea R 1 Lite
- Data：60 小时 teleoperated demonstrations
- Task：long-horizon towel folding
- Metrics：
    - success rate
    - average completion time
    - inference latency

主要现象：

- Fast-WAM 系列中，有 video co-training 的方法整体表现明显强于无 video co-training 的版本。
- Fast-WAM latency 约 190 ms。
- Fast-WAM-Joint latency 约 580 ms。
- Fast-WAM-IDM latency 约 810 ms。
- Fast-WAM w.o. video co-train 成功率下降到 10%，说明训练时 video modeling 对真实任务非常关键。

### 5.3 Simulation

#### RoboTwin 2.0

主要结果：

|Method|Embodied PT|Clean|Rand.|Average|
|---|--:|--:|--:|--:|
|π0|✓|65.92|58.40|62.2|
|π0.5|✓|82.74|76.76|79.8|
|Motus|✓|88.66|87.02|87.8|
|Motus from WAN 2.2|✗|77.56|77.00|77.3|
|LingBot-VA|✓|92.90|91.50|92.2|
|LingBot-VA from WAN 2.2|✗|80.60|-|80.6|
|**Fast-WAM**|✗|91.88|91.78|**91.8**|
|Fast-WAM-Joint|✗|90.84|90.32|90.6|
|Fast-WAM-IDM|✗|91.16|91.34|91.3|
|Fast-WAM w.o. video co-train|✗|82.76|84.80|83.8|

关键结论：

- Fast-WAM 不使用 embodied pretraining，也能达到 91.8%。
- Fast-WAM 与 Fast-WAM-Joint / Fast-WAM-IDM 差距很小。
- 去掉 video co-training 后下降到 83.8%，下降幅度更大。

#### LIBERO

主要结果：

|Method|Embodied PT|Spatial|Object|Goal|Long|Average|
|---|--:|--:|--:|--:|--:|--:|
|OpenVLA|✓|84.7|88.4|79.2|53.7|76.5|
|π0|✓|96.8|98.8|95.8|85.2|94.1|
|π0.5|✓|98.8|98.2|98.0|92.4|96.9|
|LingBot-VA|✓|98.5|99.6|97.2|98.5|98.5|
|Motus|✓|96.8|99.8|96.6|97.6|97.7|
|**Fast-WAM**|✗|98.2|100.0|97.0|95.2|**97.6**|
|Fast-WAM-Joint|✗|99.6|99.4|98.2|96.8|98.5|
|Fast-WAM-IDM|✗|98.8|97.8|97.8|97.6|98.0|
|Fast-WAM w.o. video co-train|✗|89.2|99.2|95.4|90.0|93.5|

关键结论：

- Fast-WAM 平均成功率 97.6%，接近强 WAM baseline。
- Fast-WAM-Joint 和 Fast-WAM-IDM 略高，但差距不大。
- 去掉 video co-training 后下降到 93.5%，尤其 Spatial 和 Long 退化明显。

### 5.4 Main metric

主要指标：

- success rate
- average completion time
- inference latency
- clean / randomized scene average success

其中 real-world towel folding 同时关注 success rate 和 completion time，因为这个任务不仅要完成，还要避免通过反复试错拖很久才完成。

### 5.5 Baseline

主要 baseline：

- OpenVLA
- π0
- π0.5
- Motus
- Motus from WAN 2.2
- LingBot-VA
- LingBot-VA from WAN 2.2
- Fast-WAM-Joint
- Fast-WAM-IDM
- Fast-WAM w.o. video co-train

#### Baseline 类型理解

- **OpenVLA / π0 / π0.5**：VLA / robot foundation model 路线。
- **LingBot-VA / Motus**：更接近 WAM / video-action modeling 路线。
- **Fast-WAM-Joint**：模拟 joint video-action denoising WAM。
- **Fast-WAM-IDM**：模拟 video-then-action / inverse dynamics WAM。
- **Fast-WAM w.o. video co-train**：验证 video co-training 是否真的重要。

---

## 6. 我的理解与疑问

我认为这篇论文最值得记住的是：

**Fast-WAM 的核心贡献不是单纯加速，而是重新解释 WAM 的收益来源。**
它说明 WAM 的价值未必在于推理时显式生成未来视频，而可能在于训练时通过 video prediction objective 学到更好的 world-grounded representation。这个判断对理解 WAM 后续发展很重要，因为它把 WAM 从“必须 imagine-then-execute”的范式中解放出来，转向“world modeling as representation learning”。

我还没理解清楚的是：

1. 如果任务需要显式规划、反事实比较或多候选未来评估，Fast-WAM 是否仍然足够？
2. 论文主要做 single action chunk generation，并省略 outer autoregressive rollout；在更长任务链中，test-time future imagination 是否会重新变得重要？
3. video co-training 的权重 $\lambda$ 对最终 action performance 有多敏感？
4. Wan 2.2-5 B 的视频先验到底贡献了多少？如果换成更小的视频 backbone，Fast-WAM 还能否保持优势？
5. Fast-WAM 是否真的学到了物理动态，还是只是利用 video objective 做了更强的视觉时序正则化？

和我已读论文的关系：

1. **和 DreamZero / WAM 的关系**
    - DreamZero 强调 joint video-action prediction，并在推理时显式生成未来视频和动作。
    - Fast-WAM 则追问：DreamZero 这类 WAM 的收益是否真的需要 test-time future generation？
    - 结论倾向于：训练时 video prediction 更关键，推理时未来视频生成可能不是必要条件。
2. **和 Flash-WAM 的关系**
    - Flash-WAM：保留视频和动作扩散生成流程，用 modality-aware distillation 把 denoising steps 压缩到极少。
    - Fast-WAM：直接取消推理时 future video generation，把 video DiT 当作 single-pass world encoder。
    - 简单区分：
        - Flash-WAM：还要想象未来，但把想象变快。
        - Fast-WAM：训练时学会想象，推理时不显式想象。
3. **和 RynnVLA-002 / WorldVLA 的关系**
    - RynnVLA-002 / WorldVLA 强调 VLA 和 world model 在同一个框架中互相增强。
    - Fast-WAM 更进一步区分：world modeling 可以主要作为训练目标存在，不一定要在部署时显式调用 future image generation。
    - 这说明 VLA + world model 的结合方式并不只有“推理时生成未来图像”一种。
4. **和 OpenVLA / π0 的关系**
    - OpenVLA / π0 更接近 direct policy：observation + language → action。
    - Fast-WAM 推理接口也像 direct policy，但训练时多了 future video prediction，因此比普通 direct VLA 多了 world-modeling supervision。
    - 可以把 Fast-WAM 理解为一种 “VLA-like inference + WAM-style training” 的折中形态。

后续可以追踪的问题 / 论文：

1. **Test-time imagination 是否真的不重要？**
    - 需要追后续在更长程任务、规划任务、反事实控制任务上的研究。
2. **World modeling 到底应该作为训练目标还是推理模块？**
    - Fast-WAM 倾向训练目标。
    - DreamZero / LingBot-VA / Motus 更倾向推理模块。
    - Flash-WAM 是保留推理模块但加速。
3. **视频生成模型如何转化为机器人策略？**
    - 重点关注：
        - video DiT as encoder
        - video prediction as auxiliary loss
        - future video as planning representation
        - inverse dynamics from predicted video
4. **可以复现的小实验**
    - 在 LIBERO 上复现一个简化版：
        - baseline：observation → action
        - - video prediction auxiliary loss
        - 推理时是否使用 predicted future latent
    - 对比：
        - no video loss
        - video loss only during training
        - video generation at inference
    - 观察 success rate 和 latency 的 trade-off。
