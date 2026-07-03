
## 1. 概述

**OpenVLA**（Open Vision-Language-Action Model）是 Stanford 等机构在 2024 年开源的大规模 VLA 模型。

**核心思路**
**将机器人控制问题转化为 LLM 的 next-token-prediction 任务** —— 输入机器人视角图片和自然语言指令，让 LLM 直接生成代表末端执行器动作的 token，再把离散 token 解码回连续动作值。

**基础模型***
**Llama 2 7B** + **Prismatic** 视觉编码器（DINOv2 + SigLIP 双塔融合）

---

## 2. 架构主线

```
┌──────────────────────────────────────────────────────────────────┐
│  输入                                                             │
│  ├── 图像: 机器人视角 RGB 图 (640×480)                              │
│  └── 指令: "What action should the robot take to {task}?"         │
│                                                                   │
│  ┌─────────────────────┐     ┌──────────────────┐                 │
│  │  Vision Encoder      │     │  Text Tokenizer   │                │
│  │  (DINOv2 + SigLIP)   │     │  (Llama tokenizer) │               │
│  │  ↓                   │     │  ↓                 │               │
│  │  2-layer MLP         │     │  subword token ids │               │
│  │  Projector           │     │  [1, batch, seq]  │               │
│  └─────────┬────────────┘     └────────┬──────────┘              │
│            │                           │                          │
│            ▼                           ▼                          │
│  ┌──────────────────────────────────────────────────┐             │
│  │              Llama 2 7B (backbone)               │             │
│  │  image tokens + text tokens → next-token-predict │             │
│  │  → 生成 action tokens (词表尾部 256 个 token)      │             │
│  └──────────────────────┬───────────────────────────┘             │
│                         ▼                                         │
│  ┌──────────────────────────────────────────────────┐             │
│  │  Action Tokenizer Decode                         │             │
│  │  token_id → bin_index → bin_center → action      │             │
│  └──────────────────────┬───────────────────────────┘             │
│                         ▼                                         │
│  输出                                                             │
│  action = [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]             │
│  7 维连续动作，范围 [-1, 1]                                         │
└──────────────────────────────────────────────────────────────────┘
```
---

## 3. Action Tokenizer

> 源码：`repos/openvla/prismatic/vla/action_tokenizer.py`

### 3.1 核心思想

将连续动作的每个维度离散化到 256 个 bin，映射为 LLM tokenizer 词表中的 token，让 LLM 像生成文本一样生成动作。

### 3.2 数学模型

```
连续动作空间: a ∈ ℝ^d, 每个维度值域 [-1, 1]

训练前预处理:
  q01 = 1st quantile, q99 = 99th quantile  →  忽略 outlier

bins = np.linspace(-1, 1, 256)         # 256 个等距边界点
bin_centers = (bins[:-1] + bins[1:]) / 2  # 相邻边界中点

encode: a_i → digitize → bin_index_i → vocab_size - bin_index_i → token_id_i
decode: token_id_i → vocab_size - token_id_i → bin_index_i → bin_centers[bin_index_i]
```

### 3.3 Encode（连续动作 → token string）

```python
# action_tokenizer.py: ActionTokenizer.__call__()
action = np.array([0.1, -0.5, 0.8, 0.0, 0.3])

action = np.clip(action, a_min=-1.0, a_max=1.0)
discretized_action = np.digitize(action, self.bins)
# discretized_action = array([141, 64, 230, 128, 166])

# 映射到词表尾部
token_ids = tokenizer.vocab_size - discretized_action
# token_ids = [32001-141, 32001-64, ...] = [31860, 31937, 31771, 31873, 31835]

# decode 为 token string
token_str = tokenizer.decode(token_ids.tolist())
# "<tok_31860><tok_31937><tok_31771><tok_31873><tok_31835>"
```

**Shape**
`(action_dim,)` → `np.digitize` → `(action_dim,)` → `vocab_size - bin` → `(action_dim,)` → `tokenizer.decode()` → `str`

### 3.4 Decode（token ids → 连续动作）

```python
# action_tokenizer.py: decode_token_ids_to_actions()
fake_ids = np.array([32000, 31950, 31800, 31920, 31980])

discretized_actions = tokenizer.vocab_size - fake_ids
# [1, 51, 201, 81, 21]

discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=255)
# [0, 50, 200, 80, 20]

decoded = bin_centers[discretized_actions]
# [-0.99607843, -0.60392157, 0.57254902, -0.36862745, -0.83921569]
```

**Shape**
`(action_dim,)` token ids → `vocab_size - ids` → `(action_dim,)` bin indices → `bin_centers[ids]` → `(action_dim,)` continuous values in `[-1, 1]`
---

## 4. Processor 路径

> 源码：`repos/openvla/prismatic/extern/hf/processing_prismatic.py`

### 4.1 图像处理管线

```
PIL Image (640×480)
  → letterbox pad: 640×480 → 640×640（等比例缩放 + padding 填充正方形）
  → resize: 640×640 → 224×224
  → center crop: 224×224 → 224×224
  → ToTensor: [0, 255] uint8 → [0, 1] float32
  → Normalize: mean=0.5, std=0.5 → [-1, 1] float32
  → stack: batch → (batch, 3, 224, 224) float32
```

```python
class PrismaticImageProcessor(ImageProcessingMixin):
    # 关键参数
    image_resize_strategy = "letterbox"    # 保持宽高比，padding 补全
    input_sizes = [(3, 224, 224)]          # 输入尺寸
    means = [(0.5, 0.5, 0.5)]
    stds = [(0.5, 0.5, 0.5)]
    # 归一化公式: (x/255 - 0.5) / 0.5 = x/127.5 - 1
    # 效果: 将 [0, 255] 映射到 [-1, 1]
```

### 4.2 文本 Tokenizer

OpenVLA 使用标准的 **Llama conversation template**：

```
"A chat between a curious user and an artificial intelligence assistant. "
"The assistant gives helpful, detailed, and polite answers to the user's questions. "
"USER: What action should the robot take to {instruction}? ASSISTANT:"
```

`tokenizer(prompt)` 按 Llama 词表切割成 subword token ids → `[1, token_count]`。

### 4.3 PrismaticProcessor 合并

```python
# processing_prismatic.py: PrismaticProcessor.__call__()
class PrismaticProcessor(ProcessorMixin):
    def __call__(self, text, images, padding=False, return_tensors="pt"):
        # Step 1: 图像处理
        pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        # Shape: (batch, 3, 224, 224)

        # Step 2: 文本处理
        text_inputs = self.tokenizer(text, return_tensors=return_tensors, padding=padding)

        # Step 3: 合并
        return BatchFeature(data={
            **text_inputs,           # input_ids, attention_mask
            "pixel_values": pixel_values,
        })
```

**输出：**

| 字段 | Shape | 说明 |
|------|-------|------|
| `input_ids` | `(batch, seq_len)` | 文本 token ids |
| `attention_mask` | `(batch, seq_len)` | 注意力掩码 |
| `pixel_values` | `(batch, 3, 224, 224)` | 归一化后的图像张量 |

这是给 `OpenVLAForActionPrediction.predict_action(**outputs)` 的完整输入。

---

## 5. 推理全链路：`predict_action`

> 源码：`repos/openvla/prismatic/extern/hf/modeling_prismatic.py:L507`
> ⚠️ 此路径需要 GPU + checkpoint，未实际运行验证，以下为源码阅读。

```python
# modeling_prismatic.py: OpenVLAForActionPrediction.predict_action()
def predict_action(self, image, unnorm_key, instruction, uncenter_action=True):
    # Step 1: 构造 prompt + 编码
    prompt = self._build_prompt(instruction)  # Llama conversation template
    inputs = self.processor(text=prompt, images=image)
    # inputs: {input_ids, attention_mask, pixel_values}

    # Step 2: 插入空格 token 29871（匹配训练时的 prompt 格式）
    # Llama tokenizer 的 "▁" 空格前缀 token，确保 action token 与训练格式一致

    # Step 3: LLM generate
    generated_ids = self.model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=self.action_dim,   # 只生成 action_dim 个 token
        ...
    )
    # generated_ids: (1, prompt_len + action_dim)

    # Step 4: 截取新生成的 action token ids
    predicted_action_token_ids = generated_ids[0, -self.action_dim:]
    # Shape: (action_dim,)

    # Step 5: decode → continuous action
    normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(
        predicted_action_token_ids.cpu().numpy()
    )
    # Shape: (action_dim,), 值域 [-1, 1]

    # Step 6: unnormalize（可选）
    if uncenter_action:
        action_low = norm_stats[unnorm_key]["action"]["q01"]
        action_high = norm_stats[unnorm_key]["action"]["q99"]
        mask = norm_stats[unnorm_key]["action"]["mask"]  # 某些维度不需要缩放

        actions = np.where(
            mask,
            0.5 * (normalized_actions[dim] + 1) * (action_high - action_low) + action_low,
            normalized_actions[dim],
        )
    else:
        actions = normalized_actions

    return actions  # Shape: (action_dim,)
```

### 5.1 Unnormalize 公式

$$a_{\text{final}} = \begin{cases} 0.5 \times (a_{\text{norm}} + 1) \times (q_{99} - q_{01}) + q_{01}, & \text{if mask = True} \\ a_{\text{norm}}, & \text{if mask = False} \end{cases}$$

- `0.5 × (a_norm + 1)` 将 `[-1, 1]` 映射到 `[0, 1]`
- 再乘以 `(q99 - q01)` 加 `q01`，映射回原始量纲
- `mask` 用于标识哪些维度需要 unnormalize（部分维度可能本身就是 `[-1, 1]` 的范围，无需缩放）
- `action_dim` 取决于训练数据集的 action space，通过 `norm_stats[unnorm_key]["action"]["q01"]` 的长度确定