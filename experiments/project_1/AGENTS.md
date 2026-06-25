# AGENTS.md - Project 1 Code Reading Workspace

## Scope

本目录用于 OpenVLA、OpenVLA-OFT 和 π0.5 / OpenPI 的源码阅读、轻量调试与 action shape 分析。

当前目标不是完整复现模型，不是训练、微调、benchmark，也不是搭建完整机器人仿真环境。

所有操作应服务于理解 VLA action generation pipeline，重点关注 action token、continuous action head、flow matching、denoising 和 action chunk 的 shape 流程。

## Language

默认使用中文记录和沟通，保留必要英文技术名词，例如 `ActionTokenizer`、`sample_actions`、`denoise_step`、`action_horizon`、`action_dim`、`proprioception`。

不确定的结论必须标注“暂不确定”“从源码推导”或“需要 GPU 环境验证”。

## Directory Layout

推荐使用以下结构：

```text
repos/                 第三方源码，只用于阅读
scripts/               独立 probe 脚本
outputs/shape_logs/    脚本输出和 shape 日志
.planning/<project>/   每个源码项目的 planning-with-files 记录
task.md                当前任务说明
```

不要随意重命名或移动上述目录。

## Third-party Source

第三方仓库源码统一放在 `repos/` 下；如果只是学习源码，可以在 `git clone` 后删除第三方仓库自身的 `.git` 状态。

不要大规模改动第三方仓库源码；如需验证逻辑，优先在 `scripts/` 中写独立脚本，通过 `sys.path` 临时引用源码。

## Python Environment

本目录使用一个轻量 `uv` 环境服务所有源码阅读和 probe 脚本。

不要为 OpenVLA、OpenVLA-OFT、OpenPI 分别创建完整官方环境。

不要默认执行 `pip install -e .`、`uv sync` 或完整官方安装命令。

优先安装轻量依赖，例如 `numpy`、`torch`、`transformers`、`tokenizers`、`pillow`、`rich`。

暂不安装 `flash-attn`、CUDA JAX、TensorFlow/RLDS、LIBERO、ALOHA、robosuite、wandb 等重型依赖，除非用户明确要求。

## Debugging Rules

probe 脚本应尽量只验证一个问题，优先验证不依赖大模型 checkpoint 的模块，例如 tokenizer、config、transform、fake tensor shape、局部函数输入输出。

可以使用假图像、假 token、fake tensor 或 mock config 做 shape 验证。

不要在 MacBook Air 上追求完整 OpenVLA 7B、OpenVLA-OFT 或 π0.5 checkpoint forward；真实 checkpoint forward 应标注为 NVIDIA GPU 环境后续验证项。

## Human-in-the-loop

AI 主要用于定位代码入口、解释调用链、生成候选 probe 脚本和指出风险。

打印 action shape、运行调试脚本、观察输出、判断是否符合预期，应尽量由用户在 AI 辅助下手动完成。

不要让 AI 直接替代用户完成所有阅读、调试和结论判断。

重要结论应能回溯到源码位置、用户实际运行输出或明确的静态推导。

## Planning Files

每阅读或调试一个源码项目，都应使用 planning-with-files 记录过程。

推荐项目名使用 `openvla`、`openvla_oft`、`openpi`。

每个项目至少保留 `.planning/<project>/task_plan.md`、`findings.md`、`progress.md` 和 `note.md`。

`task_plan.md` 记录阶段和状态，`findings.md` 记录源码发现，`progress.md` 记录操作过程和错误。

`note.md` 用于沉淀用户自己的理解，后续作为实验报告素材，不要求一开始写成正式报告。

## Logging

脚本输出应保存到 `outputs/shape_logs/`，文件名应能看出模块和目的，例如 `openvla_action_tokenizer.log`。

日志中应明确区分 `verified by script`、`inferred from source`、`needs GPU verification`。

不要保存大模型权重、数据集、缓存或无关临时文件到本目录。

## Git Hygiene

当前仓库是学习过程记录仓库，不是第三方源码镜像仓库。

不要提交 `.venv/`、第三方仓库 `.git/`、模型权重、大数据集或缓存目录。

提交前应检查变更范围，避免把无关文件混入同一次提交；不要 commit、push 或打开 PR，除非用户明确要求。

## Collaboration Principle

优先先理解代码主线，再定位关键函数，再打印或推导 tensor shape。

保持改动局部、清晰、可复盘，不要把本任务扩展成泛机器人学课程或泛 AI 知识库。
