# 发现记录：OpenVLA-OFT

## 需求约束

- 聚焦 continuous action head 和 action chunk shape。
- 不在 MacBook Air 上运行完整 checkpoint forward。
- 优先做静态 trace 和 fake tensor probe。

## 源码发现

- `repos/openvla-oft` 已作为 source-only 源码目录，可用于阅读。
- 删除 Git 元数据前记录的 commit：`e4287e9`。
- 已验证关键入口：`repos/openvla-oft/experiments/robot/openvla_utils.py`。

## 技术决策

| 决策 | 理由 |
|------|------|
| 暂不做完整 editable install | OpenVLA-OFT 依赖较重，且包含 forked dependencies；当前学习目标不需要完整安装 |

## 资源路径

- `repos/openvla-oft/experiments/robot/openvla_utils.py`
- `repos/openvla-oft/experiments/robot/libero/run_libero_eval.py`
- action head / proprio projector 相关文件
