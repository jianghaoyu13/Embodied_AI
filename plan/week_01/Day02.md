# Day 02 — 从零实现 Mini-GPT + Tiny Shakespeare 训练

> **日期**：2026-06-24（Week 1, Day 2）
>
> **本日定位**：把 Transformer 从 paper 公式 → PyTorch 代码 → 能采样的语言模型，全链路打通一遍。这是 Week 5/6 VLA 模型理解的地基。
>
> **总投入**：约 8 小时

---

## 0. 出发前自检（5 分钟）

```bash
# 昨天的环境还活着
conda activate embai
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# wandb 还登着
wandb status

# repo 还在
cd ~/embodied-ai-bootcamp-8w
git status                # 期望 clean
git log --oneline | head  # 期望看到 Day 1 的 commit

# GPU 干净（避免昨晚 Isaac Sim 残留）
nvidia-smi                # 显存占用应 < 1GB；不是的话 kill 掉残留进程
```

**今天主战场**：`week1-fundamentals/day2_transformer/`

```bash
mkdir -p ~/embodied-ai-bootcamp-8w/week1-fundamentals/day2_transformer/{src,configs,data,checkpoints,samples}
cd ~/embodied-ai-bootcamp-8w/week1-fundamentals/day2_transformer
```

---

## 1. 今日时间表（hour by hour）

| 时段 | 时长 | 任务 | 产出 |
|------|------|------|------|
| 07:00 - 08:30 | 1.5h | Vaswani 2017 第 3 节深读 + nanoGPT 速览 | 笔记 1 页，能默写 Attention |
| 09:00 - 10:00 | 1h   | Tiny Shakespeare 数据 + 字符级 tokenizer | `data.py` |
| 10:00 - 12:00 | 2h   | 手写 `CausalSelfAttention` + 单测 | 通过 mask / shape 测试 |
| 14:00 - 15:30 | 1.5h | 拼 `Block` + `MiniGPT`，跑通 forward | 一次前向无报错 |
| 15:30 - 17:00 | 1.5h | 训练循环 + wandb，启动训练 | loss 从 ~4.2 降到 < 2.0 |
| 19:30 - 20:30 | 1h   | 采样 + Flash Attention 替换 | 生成 500 字莎翁文 + 速度对比 |
| 20:30 - 21:00 | 0.5h | 复盘 + commit + push | Day 2 commit |

---

## 2. 早晨 (07:00 - 08:30) — 论文 + 概念

### 任务
- 重读：*Attention Is All You Need* **第 3 节** + Figure 1, 2
- 浏览：[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 的 `model.py`（300 行，**先看一遍但不抄**）
- Karpathy 视频可选：*Let's build GPT: from scratch* 的前 30 分钟

### 笔记产出 — `day_plan/notes/Day02_transformer.md`

必须能默写并解释：

1. **Multi-Head Attention 公式**
   ```
   head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
   MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W^O
   Attention(Q,K,V) = softmax(QK^T / √d_k) V
   ```

2. **三件事必须能口头答**
   - 为什么要 Multi-Head 而不是单 Head 加宽？→ 不同 head 学不同子空间（语法 / 共指 / 位置 ...）
   - Causal mask 长什么样？为什么 decoder 用上三角 mask？
   - Pre-LN vs Post-LN 区别？为什么现代 GPT 都用 Pre-LN？

3. **预算估算（背下来）**
   ```
   self-attention FLOPs   = O(T² · d)
   FFN FLOPs              = O(T · d²)
   T 长时 attention 占大头，所以 long-context 才需要 FlashAttention / 线性注意力
   ```

### 自检
- [ ] 闭眼默写 attention 公式 + multi-head 拆分
- [ ] 能解释 `√d_k` 来自 dot product 方差控制（`Var(q·k) = d_k`，所以除以 `√d_k`）
- [ ] 能讲清楚为什么 Pre-LN 训练更稳

---

## 3. 上午 09:00 - 10:00 — 数据 + Tokenizer

### 3.1 下载 Tiny Shakespeare

```bash
cd ~/embodied-ai-bootcamp-8w/week1-fundamentals/day2_transformer/data
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
wc -c input.txt           # ~1.1MB
head -20 input.txt
```

### 3.2 字符级 Tokenizer + train/val split

```python
# src/data.py
import torch
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "input.txt"

def load_data():
    text = DATA_PATH.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join(itos[i] for i in l)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], stoi, itos, encode, decode

def get_batch(split_data, block_size, batch_size, device):
    ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
    x = torch.stack([split_data[i:i+block_size]   for i in ix])
    y = torch.stack([split_data[i+1:i+block_size+1] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

if __name__ == "__main__":
    train, val, stoi, itos, enc, dec = load_data()
    print("vocab size:", len(stoi))    # 期望 65
    print("train tokens:", len(train), "| val tokens:", len(val))
    x, y = get_batch(train, block_size=128, batch_size=4, device="cpu")
    print("x:", x.shape, "y:", y.shape)
    print("decode x[0]:", dec(x[0].tolist())[:80])
```

```bash
cd ..
python src/data.py
```

期望输出：
```
vocab size: 65
train tokens: ~1003854 | val tokens: ~111540
x: torch.Size([4, 128]) y: torch.Size([4, 128])
```

### 自检
- [ ] vocab = 65（26+26+数字+标点+换行）
- [ ] `y` 是 `x` 整体右移一位（next-token prediction）

---

## 4. 上午 10:00 - 12:00 — `CausalSelfAttention` + 单测

### 4.1 写实现

```python
# src/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, max_len, dropout=0.0):
        super().__init__()
        assert dim % n_head == 0
        self.n_head, self.head_dim = n_head, dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # [B, h, T, d]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)     # [B, h, T, T]
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))
```

### 4.2 写单测（必须跑过再继续）

```python
# src/test_attn.py
import torch
from model import CausalSelfAttention

torch.manual_seed(0)

def test_shape():
    attn = CausalSelfAttention(dim=64, n_head=4, max_len=32)
    x = torch.randn(2, 16, 64)
    y = attn(x)
    assert y.shape == x.shape, y.shape
    print("✓ shape ok")

def test_causality():
    """改 t 之后的 token，t 处的输出不能变。"""
    attn = CausalSelfAttention(dim=32, n_head=2, max_len=16).eval()
    x = torch.randn(1, 8, 32)
    y1 = attn(x)
    x2 = x.clone()
    x2[:, 5:] = torch.randn_like(x2[:, 5:])     # 改第 5 位之后的所有
    y2 = attn(x2)
    diff_before = (y1[:, :5] - y2[:, :5]).abs().max().item()
    diff_after  = (y1[:, 5:] - y2[:, 5:]).abs().max().item()
    print(f"diff before t=5: {diff_before:.2e}  (应为 0)")
    print(f"diff after  t=5: {diff_after:.2e}  (应 > 0)")
    assert diff_before < 1e-6, "因果性被破坏！mask 写错了"
    assert diff_after  > 1e-3
    print("✓ causality ok")

if __name__ == "__main__":
    test_shape()
    test_causality()
```

```bash
cd src && python test_attn.py && cd ..
```

期望输出：
```
✓ shape ok
diff before t=5: 0.00e+00  (应为 0)
diff after  t=5: 5.xxe-01  (应 > 0)
✓ causality ok
```

> **如果 causality 测试失败**：99% 是 mask 维度切错。检查 `self.mask[:, :, :T, :T]` 那一行。

### 自检
- [ ] 两个单测都过
- [ ] 你能用 5x5 的小矩阵手画 causal mask（上三角全 -inf）
- [ ] 能解释为什么 `transpose(1, 2)` 把 head 维度提到第二维

---

## 5. 下午 14:00 - 15:30 — `Block` + `MiniGPT` + 前向 sanity

### 5.1 拼 Block 和 MiniGPT

继续 `src/model.py`，加 Block 和 MiniGPT：

```python
# src/model.py 续
class Block(nn.Module):
    def __init__(self, dim, n_head, max_len, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head, max_len, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))     # Pre-LN
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, dim=384, n_head=6, n_layer=6,
                 max_len=256, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(dim, n_head, max_len, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # 权重共享（GPT 标配，省参数）
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_len
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks: x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
```

### 5.2 前向 + 参数量 sanity check

```python
# src/sanity.py
import torch
from model import MiniGPT

model = MiniGPT(vocab_size=65, dim=384, n_head=6, n_layer=6, max_len=256).cuda()
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params/1e6:.2f}M")        # 期望 ~10-11M

x = torch.randint(0, 65, (4, 128), device="cuda")
y = torch.randint(0, 65, (4, 128), device="cuda")
logits, loss = model(x, y)
print("logits:", logits.shape)               # [4, 128, 65]
print("init loss:", loss.item())             # 期望 ~4.17 = log(65)，没训练前的随机基线
```

> **解读 `loss ≈ log(vocab) ≈ 4.17`**：
> 一个完全没学过的模型每个 token 都是均匀分布，cross-entropy 就是 log(V)。
> 训练后 < 2.0 才算学到东西。

```bash
cd src && python sanity.py && cd ..
```

### 自检
- [ ] 参数量 10-11M
- [ ] 初始 loss ≈ 4.17（说明 init 没炸）
- [ ] forward 不报错

---

## 6. 下午 15:30 - 17:00 — 训练循环 + 启动训练

### 6.1 写训练脚本

```python
# src/train.py
import math, time, json
from pathlib import Path
import torch
import wandb
from data import load_data, get_batch
from model import MiniGPT

# ---------- config ----------
CFG = dict(
    dim=384, n_head=6, n_layer=6, max_len=256, dropout=0.1,
    batch_size=64, lr=3e-4, min_lr=3e-5,
    max_iters=5000, warmup_iters=200,
    eval_interval=250, eval_iters=50,
    grad_clip=1.0, weight_decay=0.1, betas=(0.9, 0.95),
    device="cuda", compile=True, seed=42,
)

def get_lr(it, cfg):
    if it < cfg["warmup_iters"]:
        return cfg["lr"] * it / cfg["warmup_iters"]
    progress = (it - cfg["warmup_iters"]) / (cfg["max_iters"] - cfg["warmup_iters"])
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def estimate_loss(model, train, val, cfg):
    out = {}
    model.eval()
    for split, data in [("train", train), ("val", val)]:
        losses = torch.zeros(cfg["eval_iters"])
        for k in range(cfg["eval_iters"]):
            x, y = get_batch(data, cfg["max_len"], cfg["batch_size"], cfg["device"])
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def main():
    cfg = CFG
    torch.manual_seed(cfg["seed"])
    train, val, stoi, itos, enc, dec = load_data()
    Path("checkpoints").mkdir(exist_ok=True)
    Path("samples").mkdir(exist_ok=True)

    model = MiniGPT(vocab_size=len(stoi), dim=cfg["dim"], n_head=cfg["n_head"],
                    n_layer=cfg["n_layer"], max_len=cfg["max_len"], dropout=cfg["dropout"]
                    ).to(cfg["device"])
    if cfg["compile"]:
        model = torch.compile(model)

    # 分组：weight decay 不应用到 LayerNorm / bias / embedding
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (decay if p.dim() >= 2 else nodecay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=cfg["betas"], fused=True)

    wandb.init(project="embai-day02", name="mini-gpt", config=cfg)

    t0 = time.time()
    for it in range(cfg["max_iters"] + 1):
        lr = get_lr(it, cfg)
        for g in opt.param_groups: g["lr"] = lr

        if it % cfg["eval_interval"] == 0:
            losses = estimate_loss(model, train, val, cfg)
            print(f"iter {it} | lr {lr:.2e} | train {losses['train']:.4f} | val {losses['val']:.4f} | dt {time.time()-t0:.1f}s")
            wandb.log({"train/loss": losses["train"], "val/loss": losses["val"], "lr": lr}, step=it)
            if it == cfg["max_iters"]: break

        x, y = get_batch(train, cfg["max_len"], cfg["batch_size"], cfg["device"])
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()

        if it % 50 == 0 and it > 0:
            wandb.log({"train/loss_running": loss.item(), "lr": lr}, step=it)

    # 保存
    torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos, "cfg": cfg},
               "checkpoints/mini_gpt.pt")

    # 采样一次
    ctx = torch.tensor([[stoi["\n"]]], device=cfg["device"])
    out = (model.generate if not hasattr(model, "_orig_mod") else model._orig_mod.generate)(
        ctx, max_new_tokens=500, temperature=0.9, top_k=50)
    sample = dec(out[0].tolist())
    Path("samples/final.txt").write_text(sample)
    print("---- sample ----\n", sample[:300])
    wandb.finish()

if __name__ == "__main__":
    main()
```

### 6.2 启动训练

```bash
cd ~/embodied-ai-bootcamp-8w/week1-fundamentals/day2_transformer
python src/train.py
```

期望（4090 24GB）：
- 启动后 30s 内开始稳定迭代（首轮 `torch.compile` 编译略慢，正常）
- ~2500 token/s 起步，每 1000 iter 约 3-5 分钟
- 5000 iters 大约 **20-30 分钟**
- val loss 从 ~4.17 → < 1.6 才算训出来

### 6.3 训练时干嘛

打开新终端：

```bash
nvitop                            # 显存应 6-9GB，util > 90%
# 或
watch -n 1 nvidia-smi
```

并发干 Day 6 的 PPO 论文阅读（提前预习），别干等。

### Tuning Checklist
- [ ] 显存爆了：`batch_size=32` 或 `dim=256`
- [ ] `torch.compile` 报错：先 `compile=False` 跑通
- [ ] 迭代速度 < 1000 tok/s：检查 `num_workers` / `pin_memory`，或者 fused AdamW 没开
- [ ] val loss 不降：`lr` 调到 `1e-4`、`warmup` 调到 500

### Common Pitfalls
- **`torch.compile` + `model.generate`**：compiled model 的属性在 `model._orig_mod` 上，采样时记得绕过
- **AdamW 的 fused 参数**：CUDA 才支持，CPU 训练要去掉
- **embedding 共享权重**：`self.head.weight = self.tok_emb.weight` 这一行不能少（少了参数量多 25K，无害但不规范）

---

## 7. 晚上 19:30 - 20:30 — 采样 + Flash Attention 替换

### 7.1 采样几段对比 temperature

```python
# src/sample.py
import torch
from model import MiniGPT

ckpt = torch.load("checkpoints/mini_gpt.pt", map_location="cuda", weights_only=False)
cfg, stoi, itos = ckpt["cfg"], ckpt["stoi"], ckpt["itos"]
dec = lambda l: "".join(itos[i] for i in l)

model = MiniGPT(vocab_size=len(stoi), dim=cfg["dim"], n_head=cfg["n_head"],
                n_layer=cfg["n_layer"], max_len=cfg["max_len"], dropout=0.0).cuda()
sd = ckpt["model"]
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}   # 去掉 compile 前缀
model.load_state_dict(sd)
model.eval()

ctx = torch.tensor([[stoi["\n"]]], device="cuda")
for T in [0.5, 0.8, 1.0, 1.2]:
    out = model.generate(ctx, 300, temperature=T, top_k=50)
    print(f"\n=== temperature={T} ===\n{dec(out[0].tolist())}")
```

```bash
python src/sample.py | tee samples/temperature_sweep.txt
```

观察：
- T=0.5：保守，重复多
- T=0.8-1.0：莎翁味最浓（正常段落）
- T=1.2：开始胡言乱语

### 7.2 用 PyTorch SDPA / Flash Attention 替换手写 attention

PyTorch 2.x 自带 `F.scaled_dot_product_attention`，会自动选 Flash / Memory-Efficient kernel。

新建 `src/model_sdpa.py`，**只改 attention 内层 12 行**：

```python
# src/model_sdpa.py — 复制 model.py 的所有内容，把 forward 改成下面这样
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, max_len, dropout=0.0):
        super().__init__()
        assert dim % n_head == 0
        self.n_head, self.head_dim = n_head, dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout_p = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)
```

> 注意：用了 `is_causal=True` 之后**不再需要手动 mask**，PyTorch 内部处理。

写一个简单的速度对比：

```python
# src/bench_attn.py
import time, torch
from model      import MiniGPT as MGPT_naive
from model_sdpa import MiniGPT as MGPT_sdpa

def bench(cls, name, T=512):
    m = cls(vocab_size=65, dim=512, n_head=8, n_layer=8, max_len=T, dropout=0.0).cuda()
    x = torch.randint(0, 65, (16, T), device="cuda")
    for _ in range(5): m(x)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(50): m(x)
    torch.cuda.synchronize(); dt = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    print(f"{name:10s} | {dt:.3f}s | peak mem {mem:.0f} MB")

bench(MGPT_naive, "naive")
bench(MGPT_sdpa,  "sdpa")
```

```bash
python src/bench_attn.py
```

期望（4090，T=512）：
```
naive | 1.x s | peak mem ~1500 MB
sdpa  | 0.x s | peak mem ~600  MB     # 1.5-3x 速度，2-3x 显存节省
```

### 自检
- [ ] 4 个 temperature 的样本都生成成功
- [ ] SDPA 比手写快 ≥ 1.3x
- [ ] 你能口头答：Flash Attention 为啥更省显存（不显式 materialize T×T attention 矩阵）

---

## 8. 晚上 20:30 - 21:00 — 复盘 + commit + push

### 8.1 写复盘

```bash
cd ~/embodied-ai-bootcamp-8w/week1-fundamentals/day2_transformer
cat > NOTES.md <<'EOF'
# Day 2 Recap (2026-06-20)

## Done
- [x] CausalSelfAttention 单测通过（shape + causality）
- [x] Mini-GPT (~10M params) Tiny Shakespeare 训练
- [x] val loss 从 ~4.17 降到 ___（填你的）
- [x] 采样温度对比 0.5/0.8/1.0/1.2，效果可读
- [x] SDPA 替换 naive attention，速度 ___x，显存 ___x

## Insights
- causal mask 必须单测，肉眼检查不可靠
- Pre-LN 比 Post-LN 更稳的原因 = 残差路径不过 LN，梯度直通
- `√d_k` 是来自 dot product 方差控制：`Var(q·k) = d_k`
- 4090 24GB 上 dim=384, n_layer=6 是甜点配置，再大显存吃紧

## Pitfalls hit
- ...

## Tomorrow (Day 3 - SE(3) batched in PyTorch)
- 早晨：Solà 2020 Lie theory §1-3
- 上午：so3_exp / se3_exp 实现
- 下午：log(exp(x)) ≈ x 单测 + 和 cuRobo / pytorch3d 对比
EOF
```

把生成的样本贴一段进 NOTES：

```bash
echo "" >> NOTES.md
echo "## Sample (T=0.9)" >> NOTES.md
echo '```' >> NOTES.md
head -30 samples/final.txt >> NOTES.md
echo '```' >> NOTES.md
```

### 8.2 commit & push

```bash
cd ~/embodied-ai-bootcamp-8w
git add week1-fundamentals/day2_transformer
git commit -m "day2: mini-gpt from scratch + tiny shakespeare + sdpa benchmark"
git push origin main
```

### 8.3 Milestones 打勾

打开 `~/Documents/Embodied-AI/Milestones.md`，Week 1 那块勾掉「从零写 Transformer」「在 Tiny Shakespeare 上训练」两项。

---

## 9. 今日完成确认（必达 / 加分）

### 必达（5/6 才算合格）
- [ ] `CausalSelfAttention` 两个单测都过
- [ ] Mini-GPT forward + 初始 loss ≈ log(vocab)
- [ ] 训练 5000 iter 完成，val loss < 2.0
- [ ] 至少一个 temperature 下能采样出莎翁味
- [ ] wandb 上有完整训练曲线（loss + lr + val/loss）
- [ ] commit + push

### 加分
- [ ] SDPA 替换并跑过速度 / 显存对比
- [ ] 写完复盘并贴样本
- [ ] 4 个 temperature 全跑过

---

## 10. 明天 (Day 3) 预告 — Batched SE(3) in PyTorch

明早 7 点打开 `Week01.md` 的 Day 3 部分。要点：
- 实现 `hat`、`so3_exp`、`se3_exp`，要支持任意 batch shape
- 单测 `log(exp(x)) ≈ x`、`exp(0) = I`
- 和 cuRobo / pytorch3d 的实现对比，理解约定差异（spatial vs body twist 等）
- 把这套工具放到一个 mini-package（后面 Diffusion Policy / VLA 都会用到）

**别今晚提前看**——明早再读。

---

## 11. 反向提醒（容易忘）

1. **训练完关 Python 进程**：`pkill -f train.py`，否则显存还占着
2. **wandb 起名要描述配置**：`mini-gpt-d384-l6-h6` > `mini-gpt`
3. **commit 信息写动作 + 数字**：`day2: mini-gpt val_loss 1.52`，便于以后复盘
4. **明早起来先 `nvidia-smi`**：4090 显存应 < 1GB，不是的话杀进程
5. **wandb video / sample 上传**：今天的样本如果好玩，截一段往 wandb 上挂，求职博客以后能直接拿

---

## 12. 如果今天彻底崩了 — 救火方案

| 症状 | 兜底动作 |
|------|---------|
| `torch.compile` 报错 | 改 `compile=False`，先跑通 |
| OOM | `batch_size=32`，`dim=256`，`n_layer=4` |
| val loss 不降（卡在 3.x） | 检查 lr（用 1e-4 重跑）+ warmup（500） |
| SDPA 那段没时间 | 跳过，明天有空再补 |
| 单测过不了 | 不要继续往下，先修 mask（最常见 bug） |
| 训练时间超 1h | 改 `max_iters=3000`，能跑通就行 |

---

> Day 2 完成 = 你已经有从零搭 Transformer 的肌肉记忆。
> Week 5/6 看 OpenVLA / π0 / GR00T 源码时，你会在 30 秒内认出 attention block 长什么样。
