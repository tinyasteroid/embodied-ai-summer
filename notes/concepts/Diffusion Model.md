Hugging Face Diffusion Models Class
https://huggingface.co/learn/diffusion-course/unit0/1

## Intuition
diffusion 是一种从“噪声中逐步生成目标变量”的建模方式，在机器人论文里，这个目标变量从 image 变成了 action sequence / latent state / trajectory。
需要理解<mark style="background: #FF5582A6;">“给什么加噪、模型预测什么、condition 是什么、去噪多少步、输出如何用于控制、为什么要用 diffusion？”</mark>。

diffusion 的第一步是**人为添加噪声**，比较符合直觉的思考是：如果我知道如何一步步把真实数据破坏成噪声，那我能不能训练一个模型，学会反过来一步步把噪声修复成真实数据？
> diffusion 本意是“扩散”，就像一滴墨水滴进清水里，diffusion model 要做的是从一杯均匀混乱的水里，还原出最初那滴墨水的形状。

所以，diffusion 由两个过程组成：
```text
Forward process:
真实数据 -> 加噪 -> 纯噪声

Reverse process:
纯噪声 -> 去噪 -> 真实数据
```
Forward process 是人为设计的，不需要学习；
Reverse process 才是模型要学习的。

training & inference
加噪过程：
xt = sqrt(alpha_bar_t) * x 0 + sqrt(1 - alpha_bar_t) * epsilon

模型常见训练目标：
`model(xt, t) ≈ epsilon`，也就是让模型预测加入到数据的**噪声**。使用 MSE 作为损失函数。
由于噪声来自标准高斯分布，形式更简单，相较于预测真实数据也更加稳定。

推理开始时的 noisy action sequence 通常直接从标准高斯分布采样，再通过 diffusion model 实现一步步去噪。

**Diffusion Model 学到了什么？**
在不同噪声强度下，如何把当前样本往真实数据分布方向修正。

### Conditional Diffusion
`model(xt, t, condition) -> noise`

### Diffusion Policy
Diffusion Policy 通常学习 `observation -> action sequence`，也就是给定当前观察，生成一段未来动作。

**为什么机器人动作适合 diffusion？**
Diffusion 可以建模复杂的分布，同一个 observation 下，可以采样出多个合理 action sequence。而普通 MSE 回归可能学到多个动作方案的平均值，导致动作不合理。


与 VLA 的关系
常作为 diffusion action head / flow matching action head，用来生成连续、平滑的动作序列。
作为 VLA 中 action generation head。

### Latent Diffusion
具身智能中，模型可能给出 future latent states、world model latent 等，再在 latent 上做 diffusion：`latent noise -> denoised latent -> decoder -> data`

DDPM