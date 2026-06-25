# 发现记录：OpenVLA

## 需求约束

- 聚焦 OpenVLA 原版 action token encode / decode 路径。
- 不在 MacBook Air 上运行完整 OpenVLA 7B checkpoint。
- 优先使用用户手动运行的轻量 probe 脚本做 shape 检查。

## 源码发现

- `repos/openvla` 已作为 source-only 源码目录，可用于阅读。
- 删除 Git 元数据前记录的 commit：`c8f03f4`。
- 已验证关键入口：`repos/openvla/prismatic/vla/action_tokenizer.py`。

## 技术决策

| 决策 | 理由 |
|------|------|
| 先做静态 trace | 避免过早安装重型依赖或加载 checkpoint |

## 资源路径

- `repos/openvla/prismatic/vla/action_tokenizer.py`
- `repos/openvla/prismatic/extern/hf/modeling_prismatic.py`
- `repos/openvla/vla-scripts/deploy.py`
