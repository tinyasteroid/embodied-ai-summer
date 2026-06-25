# 任务计划：OpenVLA action token decode shape

## 目标

理解 OpenVLA 原版从 image + instruction 到 action token，再 decode 成 continuous action 的代码路径和关键 shape。

## 当前阶段

阶段 1：源码定位

## 阶段安排

### 阶段 1：源码定位
- [ ] 阅读 `repos/openvla`
- [ ] 定位 `ActionTokenizer`、HF model wrapper、deploy 入口
- [ ] 将关键文件记录到 `findings.md`
- **状态：** 进行中

### 阶段 2：静态 shape 追踪
- [ ] 追踪 processor 输出字段
- [ ] 追踪 generated token slicing
- [ ] 追踪 action token decode
- **状态：** 待开始

### 阶段 3：手动 probe
- [ ] 必要时编写最小 probe 脚本
- [ ] 由用户手动运行 shape / tokenizer 测试
- [ ] 将输出保存到 `outputs/shape_logs/`
- **状态：** 待开始

### 阶段 4：理解沉淀
- [ ] 将用户自己的理解更新到 `note.md`
- [ ] 明确标注需要 GPU 环境验证的部分
- **状态：** 待开始

## 关键问题

1. action token ids 是如何从 generated ids 中截取出来的？
2. `decode_token_ids_to_actions` 前后的精确 shape 是什么？
3. 哪些结论来自脚本验证，哪些结论来自源码静态推导？

## 错误记录

| 错误 | 尝试次数 | 处理方式 |
|------|----------|----------|
