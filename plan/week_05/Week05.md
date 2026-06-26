# Week 5 — VLM 基础 + LoRA 微调 + Action Tokenizer

> **本周定位**：从纯机器人 stack 跨到 **多模态大模型**。VLA = VLM + Action Head，所以你必须先掌握 VLM。

---

## 本周目标

完成本周后，你应该能：
- [ ] 解释 CLIP / SigLIP / DINOv2 的训练目标和适用场景
- [ ] 从零写 LLaVA 风格的 VLM（视觉 encoder + projector + LLM）
- [ ] 用 LoRA / QLoRA 微调 Qwen2.5-VL-7B 在自录数据上
- [ ] 实现至少 2 种 action tokenizer（VQ-VAE / BPE-based / FAST）
- [ ] 跑通 OpenVLA inference，理解端到端 pipeline

---

## 本周可交付物

```
week5-vlm/
├── day29_transformer_review/  # FlashAttn / RoPE / GQA 复习
├── day30_clip_dino/           # CLIP / DINOv2 表征对比
├── day31_llava_arch/          # 自建 mini-LLaVA
├── day32_lora_qwen/           # LoRA 微调 Qwen2.5-VL
├── day33_action_tokenizer/    # VQ + FAST + BPE
├── day34_openvla/             # OpenVLA 推理 pipeline
└── day35_recap/
```

---

## Day 29 — Transformer 进阶（FlashAttn / RoPE / GQA）

### 早晨论文 (2h)
- *FlashAttention-2* (Dao 2023)
- *RoFormer: Enhanced Transformer with Rotary Position Embedding* (Su 2021)
- *GQA: Training Generalized Multi-Query Transformer Models* (Ainslie 2023)

### Daily Tasks
1. 在 Day 2 的 mini-GPT 上加 RoPE，对比 absolute position embedding
2. 用 `flash-attn` 替换标准 attention，测速度
3. 实现 Grouped-Query Attention
4. 阅读 Llama / Qwen 源码对应模块

### Code Template (RoPE)

```python
# day29_transformer_review/rope.py
def precompute_rope(dim, max_len, base=10000):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)  # [max_len, dim/2]

def apply_rope(x, cos, sin):
    # x: [B, n_head, T, head_dim]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None, :x.size(2)], sin[None, None, :x.size(2)]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
```

### Tuning Checklist
- [ ] FlashAttn 在长序列（>1024）速度差 5-10x
- [ ] GQA `n_kv_heads = n_heads / 4` 是常用配置（Llama3）
- [ ] RoPE `base=10000` 标准，`base=1000000` 用于长上下文外推

---

## Day 30 — CLIP / SigLIP / DINOv2 表征对比

### 早晨论文 (2h)
- *Learning Transferable Visual Models From Natural Language Supervision* (CLIP, Radford 2021)
- *Sigmoid Loss for Language Image Pre-Training* (SigLIP, Zhai 2023)
- *DINOv2: Learning Robust Visual Features without Supervision* (Oquab 2023)

### Daily Tasks
1. 用 `open_clip` / `transformers` 加载 CLIP-ViT-L/14、SigLIP、DINOv2-Large
2. 在同一组机器人图片上提取特征
3. 用 cosine similarity 做 retrieval task
4. 把三种 feature 各喂给 Day 11 的 BC，对比下游成功率

### 三者对比

| 维度 | CLIP | SigLIP | DINOv2 |
|------|------|--------|--------|
| 监督信号 | 图文对（softmax） | 图文对（sigmoid） | 自监督（无文本） |
| 文本理解 | ✅ | ✅ | ❌ |
| 物体定位 | 中 | 中 | 强 |
| dense feature | 弱 | 中 | 强 |
| 机器人视觉 | 中 | 较好 | 较好 |
| 默认推荐 | VLA conditioning | VLM backbone | 几何 / 操作 |

### Code Template (Feature Extraction)

```python
# day30_clip_dino/extract.py
import torch
from transformers import AutoModel, AutoProcessor

models = {
    "clip":   "openai/clip-vit-large-patch14",
    "siglip": "google/siglip-large-patch16-256",
    "dinov2": "facebook/dinov2-large",
}

def extract(model_id, images):
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().cuda()
    inputs = proc(images=images, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state.mean(dim=1)  # global feature
```

### Tuning Checklist
- [ ] DINOv2 对视角变化最鲁棒（机器人视觉常用）
- [ ] CLIP 全局信息强，但 patch-level 弱
- [ ] BC + DINOv2 通常比 BC + ResNet18 高 5-15%

---

## Day 31 — 自建 Mini-LLaVA

### 早晨论文 (2h)
- *Visual Instruction Tuning* (LLaVA, Liu 2023)
- *LLaVA-NeXT* / *Qwen2-VL* tech report

### Daily Tasks
1. 用 SigLIP-base + Qwen2.5-0.5B 拼一个 mini-VLM
2. 实现 image projector（MLP, vision dim → LLM dim）
3. 在 LLaVA-Pretrain 的 300 条样本上做 instruction tuning（小规模）
4. 用一张图问「这是什么？」看模型回答

### Mini-LLaVA 架构

```
┌─────────────┐
│   Image     │
└──────┬──────┘
       │
┌──────▼─────────┐
│  Vision Tower  │   SigLIP, frozen
│  → [N, D_v]    │
└──────┬─────────┘
       │
┌──────▼─────────┐
│  MM Projector  │   2-layer MLP, trainable
│  → [N, D_l]    │
└──────┬─────────┘
       │
       │  ←──── concat with text token embeddings
       │
┌──────▼─────────┐
│  LLM Decoder   │   Qwen2.5, LoRA
│  → tokens      │
└────────────────┘
```

### Code Template (Mini-LLaVA Forward)

```python
# day31_llava_arch/mini_llava.py
class MiniLLaVA(nn.Module):
    def __init__(self, vision_id="google/siglip-base-patch16-224", llm_id="Qwen/Qwen2.5-0.5B"):
        super().__init__()
        self.vision = AutoModel.from_pretrained(vision_id)
        self.llm    = AutoModelForCausalLM.from_pretrained(llm_id)
        d_v = self.vision.config.hidden_size
        d_l = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(d_v, d_l), nn.GELU(), nn.Linear(d_l, d_l),
        )
        for p in self.vision.parameters(): p.requires_grad = False  # freeze

    def forward(self, image, input_ids, attention_mask, labels=None):
        v = self.vision(pixel_values=image).last_hidden_state  # [B, N, d_v]
        v = self.projector(v)                                   # [B, N, d_l]
        t = self.llm.get_input_embeddings()(input_ids)          # [B, T, d_l]
        # 在 input_ids 中预留 N 个 <image> token，替换为 v
        # 这里简化：直接 prepend v
        emb = torch.cat([v, t], dim=1)
        attn = torch.cat([torch.ones(v.size(0), v.size(1), device=v.device), attention_mask], dim=1)
        out = self.llm(inputs_embeds=emb, attention_mask=attn, labels=labels)
        return out
```

### Tuning Checklist
- [ ] 训练阶段 1：只训 projector（vision freeze, LLM freeze）
- [ ] 训练阶段 2：projector + LLM (LoRA)
- [ ] 学习率：projector `1e-3`, LoRA `2e-4`

---

## Day 32 — LoRA / QLoRA 微调 Qwen2.5-VL

### 早晨论文 (1.5h)
- *LoRA: Low-Rank Adaptation of Large Language Models* (Hu 2021)
- *QLoRA: Efficient Finetuning of Quantized LLMs* (Dettmers 2023)

### Daily Tasks
1. 录 50 张机器人桌面场景 + caption（"a robot arm picking a red block"）
2. 用 `peft` + `transformers` 微调 Qwen2.5-VL-7B
3. 对比微调前后在你领域数据上的 caption 质量
4. 测量显存占用（QLoRA 应能 24GB 跑 7B）

### Code Template (PEFT LoRA)

```python
# day32_lora_qwen/train_lora.py
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",
                                            quantization_config=bnb, device_map="auto")

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()  # 应 < 1%
```

### LoRA 关键超参

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `r` | 8 / 16 / 32 | 大任务用 32 |
| `lora_alpha` | 2r | scaling factor |
| `lora_dropout` | 0.05 | |
| `target_modules` | qkvo + ffn | 全 linear 都加 |
| `lr` | 2e-4 | LoRA 比 full FT 高 5-10x |
| 训练精度 | bf16 | fp16 容易 NaN |

### Tuning Checklist
- [ ] QLoRA 显存：7B model 大约 14GB，配 24GB GPU 还有余量
- [ ] `gradient_checkpointing=True`，省一半显存
- [ ] 训练崩溃：先关 4bit，再加 r

### Common Pitfalls
- 微调数据少（< 100 条）容易过拟合，加 weight_decay=0.05
- 中文模型注意 tokenizer 行为
- LoRA 不能动 embedding 层（除非 `modules_to_save=["lm_head", "embed_tokens"]`）

---

## Day 33 — Action Tokenizer 全家桶

### 早晨论文 (2h)
- *RT-2* 中的 action discretization（每维 256 bins）
- *FAST: Efficient Action Tokenization for Vision-Language-Action Models* (Pertsch 2025)
- *VQ-VAE* (van den Oord 2017) — 用于 RDT 等

### Daily Tasks
1. 实现 3 种 tokenizer：
   - **Naive bin discretization**（RT-1/RT-2 风格）
   - **VQ-VAE** action codebook
   - **FAST**（DCT + BPE，2025 SOTA）
2. 在 LeRobot Push-T 数据上对比重建误差
3. 看哪种对 chunk 友好

### 三种 Tokenizer 速记

| 方法 | 思路 | 优点 | 缺点 |
|------|------|------|------|
| Bin (RT-2) | 每维独立 256 bins | 简单 | 长 chunk 效率低 |
| VQ-VAE | 动作序列 → discrete codes | chunk 友好 | 训练麻烦 |
| FAST | DCT 频域 + BPE | 极高压缩比 | 实现复杂 |

### Code Template (Naive Bin)

```python
# day33_action_tokenizer/bin.py
class BinTokenizer:
    def __init__(self, n_bins=256, low=-1.0, high=1.0):
        self.n_bins, self.low, self.high = n_bins, low, high
        self.bins = np.linspace(low, high, n_bins + 1)
    def encode(self, a):  # [..., D] -> [..., D] long
        a = np.clip(a, self.low, self.high)
        return np.digitize(a, self.bins) - 1
    def decode(self, idx):  # [..., D] long -> [..., D]
        return (idx + 0.5) * (self.high - self.low) / self.n_bins + self.low
```

### Code Template (VQ-VAE)

```python
# day33_action_tokenizer/vqvae.py
class VQEmbedding(nn.Module):
    def __init__(self, K=512, D=64):
        super().__init__()
        self.K, self.D = K, D
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1/K, 1/K)
    def forward(self, z):  # [B, T, D]
        flat = z.view(-1, self.D)
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.embedding.weight.t()
             + self.embedding.weight.pow(2).sum(1))
        idx = d.argmin(-1)
        zq = self.embedding(idx).view(z.shape)
        # straight-through estimator
        zq = z + (zq - z).detach()
        loss = ((zq.detach() - z)**2).mean() + 0.25 * ((zq - z.detach())**2).mean()
        return zq, idx.view(z.shape[:-1]), loss
```

### Tuning Checklist
- [ ] VQ codebook 死亡（dead codes）：用 EMA 更新或 reset 机制
- [ ] FAST 实现可参考 `physical-intelligence/openpi` 中的版本
- [ ] 重建 MSE 应 < 1e-3

---

## Day 34 — OpenVLA 推理 + 数据 pipeline

### 早晨论文 (2h)
- *OpenVLA: An Open-Source Vision-Language-Action Model* (Kim 2024)
- 关键：基于 Llama2-7B + DINOv2 + SigLIP，输出 7-DoF action token

### Daily Tasks
1. clone `openvla/openvla`
2. 下载 7B checkpoint，跑 zero-shot inference
3. 在自录的 30 条数据上做 LoRA 微调（约 1 小时）
4. 评估微调前后在 LeRobot Pusht 上成功率

### OpenVLA 架构速记

```
[Image] → DINOv2 + SigLIP (concat) → Projector → Llama2-7B
                                                   ↓
[Instruction] ─────────────────────→ tokenizer ────┤
                                                   ↓
                                                 Action token (×7)
                                                   ↓
                                                 De-tokenize
                                                   ↓
                                                 7-DoF delta EE pose
```

### 推理 Pipeline

```python
# day34_openvla/infer.py
from transformers import AutoModelForVision2Seq, AutoProcessor

processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    trust_remote_code=True, low_cpu_mem_usage=True,
).to("cuda")

prompt = "In: What action should the robot take to pick up the red block?\nOut:"
image = Image.open("scene.jpg")
inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
print(action)  # [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
```

### Tuning Checklist
- [ ] `unnorm_key` 必须正确（数据集名），否则动作 scale 错
- [ ] LoRA 微调 4090 24GB 刚好（batch_size=1, grad_accum=4）
- [ ] 推理速度：7B 模型在 4090 上 ~6 Hz，需要量化或蒸馏才能上真机

### Common Pitfalls
- OpenVLA 默认 7-DoF delta，不是 absolute joint
- 输入图像 resize 到 224×224，比例错了精度大跌
- 训练时 action token 必须用相同的 tokenizer

---

## Day 35 — 周报 + VLA 知识树整理

### Daily Tasks
1. 写一篇 weekly recap：「VLA 的 5 个核心模块」
2. 画一张「VLM 家族 → VLA 家族」演化图
3. 整理「LoRA 微调 7B 模型的硬件 / 数据 / 时长」估算表
4. 准备 Week 6 的 SOTA VLA 论文（π0 / GR00T / RDT）

### 自检：你应该能回答
1. 为什么 OpenVLA 用 DINOv2 + SigLIP 双 backbone？
2. Action discretization 的瓶颈是什么？为什么有 FAST？
3. LoRA 和 full FT 在 7B 模型上效果差多少？
4. VLA 推理慢，工业界怎么解决？（量化 / 蒸馏 / chunk / 并行）
5. 给你 100 条真机数据，你怎么微调一个 VLA？

---

## 本周关键概念图（背下来）

```
                    ┌─────────────────┐
                    │   VLA = VLM + Action Head    │
                    └────────┬────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
  ┌────▼────┐          ┌─────▼────┐         ┌─────▼─────┐
  │  Vision │          │   LLM    │         │  Action   │
  │ Encoder │          │ Backbone │         │   Head    │
  └─────────┘          └──────────┘         └───────────┘
   DINOv2/CLIP          Llama/Qwen           Bin/VQ/FAST/
   /SigLIP                                   Diffusion/FM
```

---

## 本周总投入预估

| 活动 | 小时 |
|------|------|
| 论文阅读 | 14h |
| 代码（VLM / LoRA） | 22h |
| 训练等待 | 12h |
| 周报 | 5h |
| **合计** | **53h** |

→ [Week06.md](./Week06.md)
