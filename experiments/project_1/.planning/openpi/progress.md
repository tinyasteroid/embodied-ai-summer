# 进度记录：OpenPI / π0.5

## 会话：2026-06-25

### 阶段 1：源码定位

- **状态：** 进行中
- 已完成操作：
  - 初始化 OpenPI / π0.5 源码阅读的 planning 文件。
  - 使用 shallow clone 将 `Physical-Intelligence/openpi` 克隆到 `repos/openpi`。
  - 成功更新 `third_party/aloha` submodule。
  - `third_party/libero` submodule update 长时间卡住，因此改用 direct shallow clone 获取 LIBERO 源码阅读副本。
  - 删除 OpenPI 和 third-party 源码目录中的 `.git` 元数据。
- 创建或修改的文件：
  - `.planning/openpi/task_plan.md`
  - `.planning/openpi/findings.md`
  - `.planning/openpi/progress.md`
  - `.planning/openpi/note.md`
  - `repos/openpi/`（source-only，已被父仓库忽略）

## 测试结果

| 测试 | 输入 | 预期 | 实际 | 状态 |
|------|------|------|------|------|
| 源码目录存在 | `repos/openpi/src/openpi` | 目录存在 | 目录存在 | 通过 |
| 第三方源码存在 | `repos/openpi/third_party/aloha`, `repos/openpi/third_party/libero` | 目录存在 | 目录存在 | 通过 |
| Git 状态已移除 | `find repos/openpi -name .git -print` | 无输出 | 无输出 | 通过 |
| 关键符号可搜索 | `rg "sample_actions|denoise" repos/openpi` | 找到匹配 | 找到匹配 | 通过 |

## 错误日志

| 时间 | 错误 | 尝试次数 | 处理方式 |
|------|------|----------|----------|
| 2026-06-25 | `third_party/libero` submodule clone 长时间卡住 | 1 | 中断卡住的 submodule update，改用 direct shallow clone 获取 source-reading copy |
