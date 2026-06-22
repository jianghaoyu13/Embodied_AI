# Week 3 — Diffusion Policy + 视觉表征学习

> **本周定位**：本周是「能不能进入头部具身团队」的分水岭。Diffusion Policy 是 2023-2026 模仿学习的事实标准，几乎所有 VLA 模型（π0、RDT、CogACT）都在用它的变体。

---

## 本周目标

完成本周后，你应该能：
- [ ] 从零推导 DDPM / DDIM / Flow Matching 的公式
- [ ] 独立实现 Diffusion Policy，并在 Push-T 达到 ≥85% 成功率
- [ ] 解释为什么 DP 比 BC/ACT 更鲁棒（multimodal action distribution）
- [ ] 在 DP 上做至少 3 组消融（backbone / horizon / scheduler）
- [ ] 体验 3D 表征（DP3）和加速变体（Consistency Policy）

---

## 本周可交付物

```
week3-diffusion/
├── day15_ddpm/         # 1D toy DDPM 复现
├── day16_dp_paper/     # DP 论文笔记 + 架构图
├── day17_dp_pusht/     # DP 在 Push-T 复现
├── day18_dp_ablation/  # backbone / horizon / scheduler 消融
├── day19_dp3/          # 3D 点云版本 (DP3)
├── day20_flow/         # Flow Matching / Consistency Policy
└── day21_recap/        # 周报 + 博客草稿
```

---

## Day 15 — Diffusion Models 数学基础

### 早晨论文 (2h)
- *Denoising Diffusion Probabilistic Models* (Ho 2020)
- *Elucidating the Design Space of Diffusion-Based Generative Models* (Karras 2022, EDM)

**自己推一遍**：
- Forward: `q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t) I)`
- Reverse: `p_θ(x_{t-1} | x_t) = N(μ_θ, σ²I)`，其中 `μ_θ` 由 `ε_θ` 推出
- 训练 loss: `||ε - ε_θ(x_t, t)||²`

### Daily Tasks
1. 在 1D 数据上（mixture of gaussians）实现 DDPM，看采样效果
2. 实现 DDIM（确定性快采样）
3. 对比 DDPM 1000 步 vs DDIM 50 步采样质量
4. 把噪声调度（linear / cosine）切换，看影响

### Code Template (1D DDPM Toy)

```python
# day15_ddpm/toy_ddpm.py
import torch, torch.nn as nn
import numpy as np

class NoiseNet(nn.Module):
    def __init__(self, dim=1, hidden=128, T=1000):
        super().__init__()
        self.t_embed = nn.Embedding(T, hidden)
        self.net = nn.Sequential(
            nn.Linear(dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
    def forward(self, x, t):
        h = torch.cat([x, self.t_embed(t)], dim=-1)
        return self.net(h)

T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
abar = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, noise):
    return torch.sqrt(abar[t])[:,None] * x0 + torch.sqrt(1 - abar[t])[:,None] * noise

def train_step(net, x0, opt):
    B = x0.shape[0]
    t = torch.randint(0, T, (B,))
    noise = torch.randn_like(x0)
    xt = q_sample(x0, t, noise)
    pred = net(xt, t)
    loss = ((pred - noise) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

@torch.no_grad()
def ddpm_sample(net, shape):
    x = torch.randn(shape)
    for t in reversed(range(T)):
        eps = net(x, torch.full((shape[0],), t, dtype=torch.long))
        a, ab = alphas[t], abar[t]
        mean = (x - (1 - a) / torch.sqrt(1 - ab) * eps) / torch.sqrt(a)
        if t > 0:
            x = mean + torch.sqrt(betas[t]) * torch.randn_like(x)
        else:
            x = mean
    return x
```

### Tuning Checklist
- [ ] 用 cosine schedule 替换 linear，看 sample 质量
- [ ] T=1000 vs T=100 vs T=50，DDPM 步数影响
- [ ] DDIM η=0 (deterministic) vs η=1（=DDPM），对比

### Common Pitfalls
- 公式里 `√ᾱ_t` 而不是 `√α_t`，前者是累积
- 训练时一定要 sample 各个 t（uniform），不能只用大 t
- noise 必须和 x0 同形状

---

## Day 16 — Diffusion Policy 论文精读

### 早晨论文 (3h，必须深读)
- *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (Chi 2023)
- 重点：
  1. 为什么用 diffusion 而不是 GMM / energy-based？（**multimodal action**）
  2. 为什么 score over **action sequence** 而不是单步？
  3. CNN backbone (UNet1D) vs Transformer，何时选哪个？
  4. Receding Horizon Control：predict 16 steps, execute 8 steps

### Daily Tasks
1. 整理一份 5 页的论文笔记（含核心图复制）
2. 画一张 DP 数据流图（observation → encoder → diffusion → action）
3. 列出 DP 与 ACT 的 5 个核心差异
4. 不写代码，今天专注于「想清楚」

### DP 关键设计选择

| 维度 | DP-CNN (UNet1D) | DP-Transformer |
|------|-----------------|----------------|
| 适用 | 短序列 (To=2, Tp=16) | 长序列、多模态 |
| 速度 | 快 | 慢 |
| 调参 | 简单 | 难 |
| 默认 | ✅ | 仅长任务 |

### 核心超参速查

| 参数 | 推荐值 |
|------|--------|
| `obs_horizon (To)` | 2 |
| `pred_horizon (Tp)` | 16 |
| `action_horizon (Ta)` | 8 |
| `num_diffusion_steps` | 100 (train) / 16 (DDIM infer) |
| `noise_scheduler` | DDPM 训练 / DDIM 推理 |
| `lr` | `1e-4` |
| `lr_backbone` | `1e-4` 或 freeze |
| `batch_size` | 64 |
| `epochs` | 200-500 |

---

## Day 17 — DP 在 Push-T 复现

### Daily Tasks
1. clone `real-stanford/diffusion_policy` 官方 repo
2. 跑通 Push-T (state-based)
3. 加上 image-based 版本，对比性能
4. 同时跑你 Day 11 的 BC baseline 在同一任务，把数字放一起

### 训练命令

```bash
# state-based
python train.py --config-name=train_diffusion_unet_lowdim_workspace \
    task=pusht_lowdim training.seed=42

# image-based
python train.py --config-name=train_diffusion_unet_image_workspace \
    task=pusht_image training.seed=42
```

### 期望结果

| 方法 | Push-T 成功率 |
|------|--------------|
| BC (你的 Day 11) | ~30-45% |
| ACT | ~50-65% |
| DP-CNN (state) | ~85-92% |
| DP-CNN (image) | ~78-88% |
| DP-Transformer | 视任务 |

### Tuning Checklist
- [ ] EMA（exponential moving average）on weights，几乎免费 +5%
- [ ] 数据归一化必须正确，看 wandb action histogram
- [ ] DDIM 推理步数 8 → 16 → 32，看精度提升

### Common Pitfalls
- **Action normalization**：min-max scaling 到 [-1, 1]，模型才好学
- **Observation normalization**：用 dataset 统计量
- **Receding horizon 推理时机**：每 Ta 步重新预测一次，不要每步都预测
- **训练时 randomly drop conditioning**（classifier-free guidance）：DP 默认不开，但可试

---

## Day 18 — DP 消融实验

### Daily Tasks (今天就是「跑实验」)
跑 5-7 组消融，每组训 50-100 epoch（够看趋势）：

| 实验 | 修改 | 预期 |
|------|------|------|
| A. baseline | 默认 | 参考 |
| B. obs horizon | To=1 vs 2 vs 4 | To=2 最优 |
| C. pred horizon | Tp=8 vs 16 vs 32 | 视任务 |
| D. action horizon | Ta=1 vs 4 vs 8 | Ta=8 抖动小 |
| E. backbone | UNet1D vs Transformer | UNet 更稳 |
| F. infer steps | DDIM 4/8/16/64 | 16 是甜点 |
| G. EMA on/off | | EMA on 更好 |

### 输出
- 一张消融柱状图（matplotlib）
- 一段文字：哪些参数最敏感

### Code Template (Hydra Sweep)

```yaml
# configs/sweep.yaml
defaults: [_self_]
hydra:
  sweeper:
    params:
      pred_horizon: 8,16,32
      obs_horizon: 1,2,4
```

```bash
python train.py -m  # multirun
```

---

## Day 19 — 3D Diffusion Policy (DP3)

### 早晨论文 (1.5h)
- *3D Diffusion Policy* (Ze 2024)
- 核心：用稀疏点云替换图像，用 DP3 (sparse encoder) 而非 ResNet

### Daily Tasks
1. 安装 `dp3` 官方 repo
2. 在 MetaWorld 或 Adroit 任务跑通 DP3
3. 对比 2D DP 和 3D DP3 性能（DP3 通常 +10-20% on 3D 任务）
4. 理解为什么点云 robust to viewpoint

### DP3 vs DP 关键差异

| 维度 | DP (2D image) | DP3 (3D point cloud) |
|------|---------------|---------------------|
| 输入 | RGB(D) image | sparse point cloud |
| 编码器 | ResNet18 | MLP + max-pooling |
| 视角鲁棒性 | ❌ 强依赖固定视角 | ✅ 天然 |
| 适用 | 桌面操作 | 6DoF / 移动操作 |
| 数据效率 | 中 | 高 |

### Code Template (Point Cloud Encoder)

```python
# day19_dp3/encoder.py
class PointCloudEncoder(nn.Module):
    def __init__(self, in_dim=3, hidden=128, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, pcd):  # [B, N, 3]
        feat = self.mlp(pcd)        # [B, N, out_dim]
        return feat.max(dim=1).values  # [B, out_dim]
```

### Tuning Checklist
- [ ] 点云下采样到 1024 或 2048 点（FPS）
- [ ] 输入归一化（中心化 + scale 到单位球）
- [ ] 用 colored point cloud (XYZRGB) 略好于 XYZ

---

## Day 20 — Flow Matching & Consistency Policy

### 早晨论文 (2h)
- *Flow Matching for Generative Modeling* (Lipman 2023)
- *Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation* (Prasad 2024)
- 关键：π0 用的是 Flow Matching，不是 DDPM

### Daily Tasks
1. 在 1D toy 数据上实现 Flow Matching（参考 Day 15 框架）
2. 对比 DDPM 和 FM 的训练 loss / 收敛速度
3. 把 DP 的 noise scheduler 换成 FM，看在 Push-T 是否能更快推理（< 5 步）
4. 阅读 Consistency Policy 论文，理解 distillation 思路

### Flow Matching 核心

DDPM 学 `ε_θ(x_t, t)`（噪声预测），FM 学 `v_θ(x_t, t)`（向量场）：

```
x_t = (1-t) x_0 + t ε
v_target = ε - x_0
loss = ||v_θ(x_t, t) - v_target||²
```

推理就是 ODE solver：`dx/dt = v_θ(x_t, t)`，欧拉 4-8 步够。

### Code Template

```python
# day20_flow/flow_matching.py
def train_step_fm(net, x0, opt):
    B = x0.shape[0]
    t = torch.rand(B, 1)        # uniform in [0, 1]
    eps = torch.randn_like(x0)
    xt = (1 - t) * x0 + t * eps
    v_target = eps - x0
    v_pred = net(xt, t)
    loss = ((v_pred - v_target) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def fm_sample(net, shape, steps=4):
    x = torch.randn(shape)
    dt = -1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0], 1), 1.0 + i * dt)
        v = net(x, t)
        x = x + v * dt
    return x
```

### 推理速度对比（参考）

| 方法 | 推理步数 | 速度 (Hz) |
|------|---------|----------|
| DDPM | 100 | 1-2 |
| DDIM | 16 | 10-15 |
| Flow Matching | 4-8 | 30-50 |
| Consistency Policy | 1-2 | 100+ |

---

## Day 21 — 周报 + 博客草稿

### Daily Tasks
1. 把 Day 17-18 的所有结果整理成一篇博客（约 2000 字）
2. 标题建议：「从零复现 Diffusion Policy：消融、踩坑与 Push-T 87% 成功率」
3. 包含：
   - 1 张架构图
   - 1 张消融柱状图
   - 2-3 段视频 GIF
   - 完整超参表
   - GitHub 链接
4. 发到知乎 / Medium

### 自检：你应该能回答
1. 为什么 DP 在 multimodal action 数据上比 BC 强？
2. UNet1D 的 1D 是什么维度？（**时间维度**，不是空间）
3. 训练 1000 步 + 推理 16 步，效果会差多少？
4. 如果让你把 DP 部署到 30Hz 控制频率，你会做什么？
5. DP 的失败模式有哪些？（OOD / 长尾 / 慢）

---

## 本周关键 takeaway（把这页贴显示器）

1. **Diffusion Policy 不是新算法**，是 DDPM 在 action sequence 上的应用
2. **Multimodal** 才是 DP 的核心优势，单峰任务 BC 已够
3. **Receding Horizon** 是工程关键，单步推理会抖
4. **EMA + DDIM** 是必开的两个 trick
5. **2026 年 SOTA 用 Flow Matching** 而非 DDPM（π0 / RDT 的选择）

---

## 本周总投入预估

| 活动 | 小时 |
|------|------|
| 论文阅读 | 14h |
| 代码实现（含调试） | 22h |
| 跑消融实验 | 12h |
| 博客 / 周报 | 8h |
| **合计** | **56h** |

→ [Week04.md](./Week04.md)
