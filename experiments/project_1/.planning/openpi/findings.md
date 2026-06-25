# 发现记录：OpenPI / π0.5

## 需求约束

- 聚焦 flow matching / denoising action generation。
- 不在 MacBook Air 上运行完整 π0.5 checkpoint forward。
- 优先做静态 trace 和 fake tensor probe。

## 源码发现

- `repos/openpi` 已作为 source-only 源码目录，可用于阅读。
- 删除 Git 元数据前记录的 OpenPI commit：`15a9616`。
- 删除 Git 元数据前，`third_party/aloha` 已 checkout 到 `d1dc83a`。
- `third_party/libero` 因 submodule update 卡住，改用 direct shallow clone 获取，记录 commit 为 `8f1084e`。
- 已验证关键入口：`repos/openpi/src/openpi/models_pytorch/pi0_pytorch.py`。

## 技术决策

| 决策 | 理由 |
|------|------|
| 初期避免完整 `uv sync` | OpenPI 依赖包含较重的训练和推理栈，当前学习目标不需要完整安装 |

## 资源路径

- `repos/openpi/src/openpi/models/pi0_config.py`
- `repos/openpi/src/openpi/models_pytorch/pi0_pytorch.py`
- `repos/openpi/src/openpi/policies/`
- `repos/openpi/src/openpi/transforms.py`
