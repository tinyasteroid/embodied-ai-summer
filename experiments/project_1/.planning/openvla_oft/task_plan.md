# 任务计划：OpenVLA-OFT continuous action chunk shape

## 目标

理解 OpenVLA-OFT 中 continuous action head、proprio/state 输入和 action chunk 输出的代码路径与关键 shape。

## 当前阶段

阶段 1：源码定位

## 阶段安排

### 阶段 1：源码定位
- [ ] 阅读 `repos/openvla-oft`
- [ ] 定位 action head 和 proprio projector 相关代码
- [ ] 将关键文件记录到 `findings.md`
- **状态：** 进行中

### 阶段 2：静态 shape 追踪
- [ ] 追踪进入 action head 的 hidden state
- [ ] 追踪 proprio/state 输入路径
- [ ] 找到 action horizon 和 action dim 的定义位置
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

1. OpenVLA-OFT 是否仍然依赖 discrete action token？
2. 哪个 tensor 进入 continuous action head？
3. action chunk horizon 和 action dim 分别在哪里定义？

## 错误记录

| 错误 | 尝试次数 | 处理方式 |
|------|----------|----------|
