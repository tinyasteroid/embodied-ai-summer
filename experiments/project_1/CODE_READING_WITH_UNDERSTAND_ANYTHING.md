# 使用 Understand Anything 辅助源码阅读

本文件用于在当前 `experiments/project_1/` workspace 中，使用 Understand Anything 辅助阅读 OpenVLA、OpenVLA-OFT 和 OpenPI / π0.5 源码。

当前目标不是完整复现、训练、微调或 benchmark，而是围绕 VLA action generation pipeline 建立源码地图，重点理解 action token、continuous action head、flow matching / denoising 和 action chunk 的 shape 流程。

AI 主要用于定位入口、解释调用链、生成候选 probe 思路和提示风险；打印 action shape、运行脚本、观察输出和判断结论，应尽量由用户手动完成。

## 0. 当前目录约定

当前目录结构以本 workspace 为准：

```text
experiments/project_1/
├── AGENTS.md
├── task.md
├── CODE_READING_WITH_UNDERSTAND_ANYTHING.md
├── repos/
│   ├── openvla/
│   ├── openvla-oft/
│   └── openpi/
├── scripts/
├── outputs/
│   └── shape_logs/
└── .planning/
    ├── openvla/
    ├── openvla_oft/
    └── openpi/
```

第三方源码目录已经删除各自 `.git` 状态，仅作为 source-only 阅读副本使用。`repos/`、`.venv/`、`.uv-cache/` 已由本目录 `.gitignore` 忽略，不应提交。

Understand Anything 生成的 `.understand-anything/` 默认会落在被分析的源码目录内；如果在 `repos/<project>/` 内运行，它也会随 `repos/` 一起被父仓库忽略。

## 1. 通用使用流程

先进入本任务目录：

```bash
cd /Users/minghao/Documents/intern/embodied-ai-summer/experiments/project_1
```

每次开始分析某个项目之前，先打开对应 planning 文件：

```text
.planning/openvla/task_plan.md
.planning/openvla/findings.md
.planning/openvla/progress.md
.planning/openvla/note.md
```

使用 Understand Anything 时，优先只分析与当前问题相关的源码范围，避免一开始扫描整个大仓库：

```text
/understand repos/openvla/prismatic/vla --language zh
/understand repos/openvla-oft/prismatic --language zh
/understand repos/openpi/src/openpi/models_pytorch --language zh
```

需要全局结构时，再对单个源码仓库运行：

```text
/understand repos/openvla --language zh
/understand repos/openvla-oft --language zh
/understand repos/openpi/src/openpi --language zh
```

查看图谱：

```text
/understand-dashboard
```

深入解释某个文件：

```text
/understand-explain <file-path>
```

围绕调用链提问：

```text
/understand-chat 请解释 <入口函数/脚本> 到 <目标函数/模块> 的调用链，并列出关键文件、核心类/函数、输入输出 shape，以及哪些结论需要 GPU 环境验证。
```

Understand Anything 的回答只作为辅助材料。最终需要把用户确认后的理解沉淀到对应 `.planning/<project>/note.md`，把源码发现写到 `findings.md`，把实际操作和错误写到 `progress.md`。

## 2. OpenVLA 阅读路线

对应目录：

```text
repos/openvla/
.planning/openvla/
```

优先分析范围：

```text
/understand repos/openvla/prismatic/vla --language zh
/understand repos/openvla/prismatic/extern/hf --language zh
/understand repos/openvla/vla-scripts --language zh
```

重点文件：

```text
repos/openvla/prismatic/vla/action_tokenizer.py
repos/openvla/prismatic/extern/hf/modeling_prismatic.py
repos/openvla/vla-scripts/deploy.py
```

推荐问题：

```text
/understand-chat 请解释 OpenVLA 中 image + instruction 如何进入 processor，并说明 `input_ids` 和 `pixel_values` 的 shape 来源。
```

```text
/understand-chat 请围绕 `ActionTokenizer` 解释 action token ids 如何 encode / decode 成 continuous action，并标注每一步 shape。
```

```text
/understand-chat 请解释 `modeling_prismatic.py` 中生成 token 到 action decode 的路径，区分源码推导和需要实际 forward 验证的部分。
```

记录位置：

```text
.planning/openvla/findings.md
.planning/openvla/progress.md
.planning/openvla/note.md
outputs/shape_logs/openvla_*.log
```

## 3. OpenVLA-OFT 阅读路线

对应目录：

```text
repos/openvla-oft/
.planning/openvla_oft/
```

优先分析范围：

```text
/understand repos/openvla-oft/prismatic --language zh
/understand repos/openvla-oft/experiments/robot --language zh
/understand repos/openvla-oft/vla-scripts --language zh
```

重点文件和模块：

```text
repos/openvla-oft/experiments/robot/openvla_utils.py
repos/openvla-oft/experiments/robot/libero/run_libero_eval.py
repos/openvla-oft/prismatic/extern/hf/modeling_prismatic.py
action_head / proprio_projector 相关代码
```

推荐问题：

```text
/understand-chat 请解释 OpenVLA-OFT 是否仍依赖 discrete action token，并说明 continuous action head 的输入和输出 shape。
```

```text
/understand-chat 请定位 action chunk 的 horizon 和 action_dim 在哪里定义，并解释它们如何影响最终 action shape。
```

```text
/understand-chat 请解释 proprio/state 如何进入模型，以及它和 hidden_states 如何交互。
```

记录位置：

```text
.planning/openvla_oft/findings.md
.planning/openvla_oft/progress.md
.planning/openvla_oft/note.md
outputs/shape_logs/openvla_oft_*.log
```

## 4. OpenPI / π0.5 阅读路线

对应目录：

```text
repos/openpi/
.planning/openpi/
```

优先分析范围：

```text
/understand repos/openpi/src/openpi/models_pytorch --language zh
/understand repos/openpi/src/openpi/models --language zh
/understand repos/openpi/src/openpi/policies --language zh
/understand repos/openpi/src/openpi/transforms.py --language zh
```

重点文件：

```text
repos/openpi/src/openpi/models/pi0_config.py
repos/openpi/src/openpi/models_pytorch/pi0_pytorch.py
repos/openpi/src/openpi/policies/
repos/openpi/src/openpi/transforms.py
```

优先追踪主线：

```text
observation / state / image / language prompt
→ sample_actions
→ denoise_step
→ v_t / flow prediction
→ actions: [B, action_horizon, action_dim]
```

推荐问题：

```text
/understand-chat 请解释 OpenPI / π0.5 的 `sample_actions` 输入输出，并标注 `observation.state`、`noise`、`x_t`、`v_t`、`actions` 的 shape。
```

```text
/understand-chat 请解释 `denoise_step` 在 flow matching 采样中的作用，以及它和 action_horizon、action_dim 的关系。
```

```text
/understand-chat 请定位 `action_horizon` 和 `action_dim` 的配置来源，并说明不同 pi0 / pi0_fast / pi05 配置是否存在差异。
```

记录位置：

```text
.planning/openpi/findings.md
.planning/openpi/progress.md
.planning/openpi/note.md
outputs/shape_logs/openpi_*.log
```

## 5. 每次阅读后的记录格式

每完成一个局部问题，不直接把 Agent 回答当成最终结论。应先由用户确认，再记录到对应文件。

`findings.md` 记录源码发现：

```text
## 源码发现

- 入口：
- 关键函数：
- 调用链：
- 从源码推导的 shape：
- 暂不确定 / 需要 GPU 环境验证：
```

`progress.md` 记录实际操作：

```text
## 会话：YYYY-MM-DD

### 阶段 X：...

- 已完成操作：
- 用户手动运行的命令：
- 输出日志：
- 遇到的问题：
```

`note.md` 记录用户自己的理解：

```text
## 我的理解

## 关键代码入口

## Shape 结论

| 阶段 | Shape | 证据 |
|------|-------|------|

## 问题
```

如果实际运行了 probe 脚本，输出保存到：

```text
outputs/shape_logs/
```

日志中要明确区分：

```text
verified by script
inferred from source
needs GPU verification
```

## 6. 使用原则

1. 不要一开始全量精读或全量扫描所有源码，先围绕当前 action shape 问题做局部分析。
2. 不要让 AI 替代用户完成所有阅读和判断；AI 只辅助定位、解释、生成候选 probe 和提示风险。
3. 不在本机追求完整 checkpoint forward，不安装重型官方依赖。
4. 不直接修改第三方源码；需要验证时在 `scripts/` 中写独立 probe。
5. 不把 Understand Anything 输出、第三方源码、`.venv/`、`.uv-cache/` 纳入 Git。
6. 最终沉淀位置是 `.planning/<project>/note.md`，后续可作为实验报告素材。
