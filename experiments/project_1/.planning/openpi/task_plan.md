# 任务计划：OpenPI / π0.5 flow matching action shape

## 目标

理解 OpenPI / π0.5 中 `sample_actions`、denoising / flow matching 生成 continuous action chunk 的代码路径和关键 shape。

## 当前阶段

阶段 1：源码定位

## 阶段安排

### 阶段 1：源码定位
- [ ] 阅读 `repos/openpi`
- [ ] 定位 pi0 config、PyTorch model、policies、transforms
- [ ] 将关键文件记录到 `findings.md`
- **状态：** 进行中

### 阶段 2：静态 shape 追踪
- [ ] 追踪 observation / state / image / prompt 结构
- [ ] 追踪 `sample_actions`
- [ ] 追踪 denoising step 和 flow prediction shape
- **状态：** 待开始

### 阶段 3：手动 probe
- [ ] 必要时编写最小 fake tensor probe
- [ ] 由用户手动运行 shape 测试
- [ ] 将输出保存到 `outputs/shape_logs/`
- **状态：** 待开始

### 阶段 4：理解沉淀
- [ ] 将用户自己的理解更新到 `note.md`
- [ ] 明确标注需要 GPU 环境验证的部分
- **状态：** 待开始

## 关键问题

1. π0.5 输出的是 action token 还是 continuous action？
2. `x_t`、`noise`、`v_t` 和最终 actions 的 shape 分别是什么？
3. `action_horizon` 和 `action_dim` 在哪里配置？

## 错误记录

| 错误 | 尝试次数 | 处理方式 |
|------|----------|----------|
