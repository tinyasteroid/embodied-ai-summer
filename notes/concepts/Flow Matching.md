## Intuition
Flow matching 的目标是学习一个连续的 vector field，把噪声分布逐步“流动”到真实数据分布。

在机器人动作生成里，可以理解为：
- 训练时
	模型学习如何把一团高斯噪声，往哪个方向修动作，逐步变成真实 action chunk。
- 推理时
	模型不断告诉你“下一步应该往哪个方向走”，最后把纯噪声走成可执行动作。

## Action Flow Matching
step 1. 先采样一个噪声，和真实动作 $a_{t:{t+H}}$ 的 shape 一致：
$$
\omega \sim \mathcal{N}(0, I)
$$

step 2. 然后在噪声和真实动作之间构建一个中间状态 noisy action chunk：
$$
a_{t:t+H}^{\tau,\omega}
=
\tau a_{t:t+H}
+
(1-\tau)\omega
$$

其中：
- $\tau \in [0, 1]$ 是 flow matching 的时间步，当 $\tau = 0$ 时，表示纯噪声；当 $\tau = 1$ 时，表示真实动作。

step 3. 然后训练模型预测 flow vector field：$\omega - a_{t:t+H}$

损失函数：
$$
\left\|
\omega - a_{t:t+H}
-
f^a_\theta(a_{t:t+H}^{\tau,\omega}, o_t, \ell)
\right\|^2
$$

其中：
- $f_\theta^a(a_{t:t+H}^{\tau,\omega}, o_t,\ell)$ 是 action expert neural network，输入是括号内的东西，输出是一个 vector field，和 action chunk 保持 shape 一致。

> [!info] 如何理解这个损失函数？
> 本质上是一个**监督回归**问题，我们构造的 noisy action 是已知的，那么这条直线路径上的“正确速度”就是已知的：$v^\star = a_{t:t+H} - \omega$ ，于是训练变成一个监督回归问题，
> $$
> f_\theta^a(\text{noisy action}, \text{condition}, \tau)
> \approx
> v^\star
> $$
> 回归 loss 就是 MSE
> $$
> \left\|
> v^\star
> -
> f_\theta^a(\cdot)
> \right\|^2
> $$

训练目的是让 action expert 学会：当前 $a_{t:t+H}^{\tau,\omega}$ 应该沿着哪个速度方向变化。

最终输出：$v_{\theta}$ ，推理时：
$$
\frac{d x}{d\tau}
=
v_\theta(x,\tau,o_t,\hat{\ell})
$$
