## 一句话总结
**解决痛点**
1. 机器人学习长期受限于数据稀缺、实体异构和评估标准失真
2. 多数 VLA 模型在分布内 benchmark 上表现尚可，但泛化性差，现有 benchmark 无法区分模型是否仅通过分布内模式。


**创新点和贡献**
核心要点：<mark style="background: #FF5582A6;">*alignment first，then scale*
</mark>

- 
- 评估体系需要 OOD，来衡量模型真正的泛化能力


进行统一的三层 alignment：

|对齐层面|直观理解|作用|
|---|---|---|
|Representation Alignment|把不同机器人的状态/动作放进统一 canonical state-action template|解决“不同机器人数据格式不一样”的问题|
|Motion Alignment|用 camera-frame delta pose 表示末端动作|让视觉上相似的动作在数值空间中也更接近|
|Behavior Alignment|用 system prompt 和 in-context adaptation 让模型根据历史执行状态适配不同 embodiment|让模型能根据当前机器人表现隐式识别本体差异|
## 数据
它构造了一个约 **38,100 小时**的预训练语料，来源包括三类：机器人操作数据、人类第一视角操作视频、Human-to-Robot 合成数据。
> 核心是用 开放机器人数据 + 人类视频，替代完全依赖私有大规模机器人采集。



## Takeways
- alignment 是 scale 的前提。cross- embodiment 要让状态、动作等都进入统一 formulation。
- 机器人 foundation model 的瓶颈不只是模型结构，而是数据能否被统一对齐。
- **ODD Evaluation**，而不只是 LIBERO / RoboTwin 的 benchmark。
- Human-to-Robot synthesis 可能是行之有效的扩展机器人数据规模的重要路线。