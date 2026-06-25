# 进度记录：OpenVLA

## 会话：2026-06-25

### 阶段 1：源码定位

- **状态：** 进行中
- 已完成操作：
  - 初始化 OpenVLA 源码阅读的 planning 文件。
  - 使用 shallow clone 将 `openvla/openvla` 克隆到 `repos/openvla`。
  - 删除 `repos/openvla/.git`，使该目录仅作为 source-only 源码副本。
- 创建或修改的文件：
  - `.planning/openvla/task_plan.md`
  - `.planning/openvla/findings.md`
  - `.planning/openvla/progress.md`
  - `.planning/openvla/note.md`
  - `repos/openvla/`（source-only，已被父仓库忽略）

## 测试结果

| 测试 | 输入 | 预期 | 实际 | 状态 |
|------|------|------|------|------|
| 源码文件存在 | `repos/openvla/prismatic/vla/action_tokenizer.py` | 文件存在 | 文件存在 | 通过 |
| Git 状态已移除 | `find repos/openvla -name .git -print` | 无输出 | 无输出 | 通过 |
| 关键符号可搜索 | `rg "class ActionTokenizer" repos/openvla` | 找到匹配 | 找到匹配 | 通过 |

## 错误日志

| 时间 | 错误 | 尝试次数 | 处理方式 |
|------|------|----------|----------|
