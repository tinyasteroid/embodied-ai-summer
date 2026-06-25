# Task: OpenVLA / OpenVLA-OFT / π0.5 Action Shape Code Reading

## 1. Task Scope

当前任务不是完整复现 OpenVLA、OpenVLA-OFT 或 π0.5，也不是进行训练、微调或 benchmark。

本任务的核心是通过阅读和轻量调试开源代码，理解 Vision-Language-Action Models 如何从视觉、语言和机器人状态生成动作输出，并重点追踪 action encode / decode、continuous action head、flow matching / denoising 过程中的关键 tensor shape。

本地开发环境主要是 MacBook Air，因此本任务以源码阅读、函数定位、轻量脚本和 shape 分析为主。完整 checkpoint 推理、GPU forward 验证、LIBERO / ALOHA / real robot evaluation 不作为本地必须完成目标。

调试和阅读过程中不要过分依赖 AI。AI 可以辅助定位源码、解释调用链、生成候选 probe 脚本和提示风险，但打印 action shape、运行脚本、观察输出、判断结论是否成立，应尽量由用户在 AI 辅助下手动完成。

---

## 2. Main Goals

本任务需要完成以下目标：

1. 阅读 OpenVLA 原版代码，理解：

   * prompt / image processor 输入流程；
   * LLM 生成 action token 的过程；
   * action token ids 如何被截取；
   * action token ids 如何 decode 成 continuous robot action；
   * `ActionTokenizer` 的 encode / decode 逻辑。

2. 阅读 OpenVLA-OFT 代码，理解：

   * 是否仍然依赖离散 action token；
   * continuous action head 的输入与输出；
   * action chunk 输出机制；
   * action horizon 和 action dimension 的定义位置；
   * 与 OpenVLA 原版离散 action token 路线的区别。

3. 阅读 π0.5 / OpenPI 代码，理解：

   * observation、state、image、language prompt 的输入结构；
   * flow matching action expert；
   * `sample_actions`、`denoise_step` 等核心函数；
   * denoising / sampling 生成 continuous action chunk 的过程；
   * action 输出 shape，例如 `[B, action_horizon, action_dim]`。

4. 在阅读代码过程中补充必要机器人基础概念，例如：

   * action dimension；
   * proprioception；
   * end-effector action；
   * joint action；
   * action chunk；
   * action horizon；
   * control frequency；
   * gripper command。

---

## 3. Source Repositories

需要阅读的主要仓库：

```text
OpenVLA:
https://github.com/openvla/openvla

OpenVLA-OFT:
https://github.com/moojink/openvla-oft

OpenPI / π0.5:
https://github.com/Physical-Intelligence/openpi
```

源码应放在本任务目录下统一管理：

```text
experiments/project_1/repos/
```

如果只学习源码，可以 `git clone` 后删除第三方仓库自身的 `.git` 状态，避免把第三方源码作为嵌套 Git 仓库管理。

每个源码项目的调试过程应使用 planning-with-files 记录：

```text
experiments/project_1/.planning/openvla/
experiments/project_1/.planning/openvla_oft/
experiments/project_1/.planning/openpi/
```

每个项目目录下至少保留 `task_plan.md`、`findings.md`、`progress.md` 和 `note.md`。其中 `note.md` 用于沉淀用户自己的理解，后续作为实验报告素材，不要求一开始写成正式报告。

---

## 4. Key Files

重点阅读文件包括但不限于：

```text
OpenVLA:
- prismatic/vla/action_tokenizer.py
- prismatic/extern/hf/modeling_prismatic.py
- vla-scripts/deploy.py

OpenVLA-OFT:
- experiments/robot/openvla_utils.py
- experiments/robot/libero/run_libero_eval.py
- action_head / proprio_projector 相关代码

OpenPI / π0.5:
- src/openpi/models/pi0_config.py
- src/openpi/models_pytorch/pi0_pytorch.py
- src/openpi/policies/
- src/openpi/transforms.py
```

---

## 5. Analysis Focus

代码阅读时优先回答以下问题。

### OpenVLA

1. 输入 image 和 instruction 后，processor 输出哪些字段？
2. `input_ids`、`pixel_values` 的 shape 是什么？
3. 模型生成的是哪些 token？
4. action token ids 如何被截取出来？
5. action token ids 如何 decode 成 continuous action？
6. 最终 action shape 是 `[action_dim]` 还是 `[B, action_dim]`？

重点 shape：

```text
input_ids: [B, L]
pixel_values: [B, C, H, W]
generated_ids: [B, T]
action_token_ids: [B, action_dim]
action: [action_dim] or [B, action_dim]
```

### OpenVLA-OFT

1. continuous action head 的输入是什么？
2. 是否输出 action chunk？
3. action chunk 的 horizon 和 action dimension 分别在哪里定义？
4. proprio/state 如何进入模型？
5. 与 OpenVLA 原版相比，decode 过程发生了什么变化？

重点 shape：

```text
hidden_states: [B, L, D]
proprio/state: [B, state_dim]
action_chunk: [B, T, action_dim] or [T, action_dim]
```

### π0.5 / OpenPI

1. π0.5 是否显式输出 action token？
2. `sample_actions` 的输入输出是什么？
3. denoising 过程中的 `x_t` / `noise` shape 是什么？
4. `v_t` 或 flow prediction 的 shape 是什么？
5. 最终 actions 的 shape 是什么？

重点 shape：

```text
observation.state: [B, state_dim]
tokenized_prompt: [B, L]
x_t / noise: [B, action_horizon, action_dim]
v_t: [B, action_horizon, action_dim]
actions: [B, action_horizon, action_dim]
```

---

## 6. Local Execution Boundary

MacBook Air 本地主要完成：

* 克隆和阅读代码；
* 搜索关键函数；
* 运行轻量 Python 脚本；
* 调试 `ActionTokenizer` 等不依赖大模型 checkpoint 的模块；
* 保存关键脚本输出，方便后续复盘和向师兄汇报。

MacBook Air 本地不强制完成：

* 加载完整 OpenVLA 7B checkpoint；
* 运行 OpenVLA-OFT 真实 checkpoint forward；
* 运行 π0.5 / OpenPI 真实 checkpoint forward；
* 跑 LIBERO / ALOHA / real robot evaluation；
* 训练、微调或 benchmark。

如果后续需要验证真实模型输出 shape，应在 NVIDIA GPU 环境中进行一次最小 forward debug。

---

## 7. Debugging Requirements

在修改或新增代码时，请遵守以下要求：

1. 不要大规模改动第三方仓库源码。
2. 如需实验，应优先在 `scripts/` 中写独立 probe 脚本。
3. 所有调试脚本应尽量只做一件事，例如：

   * 打印 action tokenizer 输入输出；
   * 打印 config 中的 action horizon / action dim；
   * 静态追踪某个 forward 函数的输入输出 shape。

4. 输出日志保存到 `outputs/shape_logs/`。
5. 每个源码项目的调试过程应同步更新 `.planning/<project>/` 下的 planning 文件。
6. `note.md` 应记录用户自己的理解、关键源码入口、shape 结论和仍不理解的问题。
7. 调试记录应明确区分：

   * 已通过脚本验证的 shape；
   * 从源码静态推导的 shape；
   * 需要 GPU 环境进一步验证的 shape。

本项目的优先级是：

```text
先理解代码主线
→ 再定位关键函数
→ 再打印或推导 tensor shape
→ 再补充必要机器人概念
→ 最后形成可复盘的 shape 结论
```

不要一开始追求完整跑通模型，也不要偏离到系统学习机器人学课程。当前任务的核心是通过 OpenVLA、OpenVLA-OFT 和 π0.5 的代码，建立对 embodied action generation pipeline 的工程直觉。
