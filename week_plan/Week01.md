# Week 1 — PyTorch 实战 + SE(3) 张量化 + RL 入门

> **本周定位**：把你的「概念级 DL/RL」推到「能从零手敲」的水平。
> 你已经懂运动学和 cuRobo，所以本周机器人学部分**只补强 SE(3) 在 PyTorch 中的 batched 实现**（VLA / DP 经常需要 EE pose 在 SE(3) 上做 loss / sample）。
> 本周不重复 IK/FK，**机器人动力学的补强放在 Week 4 Day 22**。

---

## 本周目标

完成本周后，你应该能：
- [ ] 从零写一个 Transformer + 训练循环，无需查文档
- [ ] 解释 PPO 损失函数每一项的来源
- [ ] 在 CartPole / Pendulum / HalfCheetah 上独立训出收敛策略
- [ ] 用 PyTorch 写出 batched SE(3) 操作（torch tensor，非 numpy）

---

## 本周可交付物（GitHub Repo `week1-fundamentals/`）

```
week1-fundamentals/
├── day1_pytorch/         # MNIST 训练 + Lightning 重构
├── day2_transformer/     # 从零 GPT-2 mini
├── day3_se3/             # batched SE(3) 工具库 + 单测
├── day4_mdp/             # tabular Q-learning + value iteration
├── day5_pg/              # 手写 REINFORCE / A2C
├── day6_ppo/             # PPO + GAE，CartPole 收敛
└── day7_recap/           # 周报 + wandb 截图
```

---

## Day 1 — PyTorch 工程基本功

### 早晨论文 (1.5h)
- 论文：*Attention Is All You Need* 第 3 节（自注意力）
- 任务：手写自注意力公式：`Attention(Q,K,V) = softmax(QK^T / √d_k) V`，能解释为什么除以 `√d_k`。

### Daily Tasks
1. 安装 PyTorch 2.4 + 验证 CUDA 可用
2. **手写 MNIST 训练循环**（不用 Lightning，纯 PyTorch）
3. 重构成 PyTorch Lightning，加上 wandb logging
4. 跑一个 batch_size 扫描实验，对比 16/32/64/128/256 的速度

### Code Template

```python
# day1_pytorch/train_mnist.py
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

class MLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 10),
        )
    def forward(self, x): return self.net(x)

def main():
    wandb.init(project="week1-mnist", config={"lr": 1e-3, "batch": 128, "epochs": 5})
    cfg = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(".", train=True, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=4)

    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            wandb.log({"loss": loss.item(), "acc": (logits.argmax(-1) == y).float().mean().item()})

if __name__ == "__main__":
    main()
```

### Tuning Checklist
- [ ] `lr` 在 `1e-4 ~ 3e-3` 之间扫，记录 loss 曲线
- [ ] `batch_size` 越大不一定越快（受 IO 限制），观察 GPU 利用率
- [ ] 用 `torch.compile(model)` 测速度提升（PyTorch 2.x 特性）
- [ ] `pin_memory=True` + `num_workers=4` 是标配

### Common Pitfalls
- `model.train()` 和 `model.eval()` 别忘了切，BN/Dropout 行为不同
- `loss.backward()` 前一定 `opt.zero_grad()`，否则梯度累加
- wandb log 频率太高会拖慢训练，每 N 步 log 一次

---

## Day 2 — Transformer 从零

### 早晨论文 (1.5h)
- *The Illustrated Transformer*（Jay Alammar 博客）
- *nanoGPT* repo (Karpathy) README，理解整体结构

### Daily Tasks
1. 手写多头注意力 `MultiHeadAttention`
2. 手写 `TransformerBlock`（含 LayerNorm + 残差）
3. 拼成一个 50M 参数的 mini-GPT
4. 在 Tiny Shakespeare 数据集训练 1 epoch，能采样出像样的莎士比亚风格文本

### Code Template

```python
# day2_transformer/mini_gpt.py
import torch, torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, max_len=256):
        super().__init__()
        assert dim % n_head == 0
        self.n_head, self.head_dim = n_head, dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        mask = torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab, dim=384, n_head=6, n_layer=6, max_len=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([Block(dim, n_head) for _ in range(n_layer)])
        self.ln_f, self.head = nn.LayerNorm(dim), nn.Linear(dim, vocab)
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for blk in self.blocks: x = blk(x)
        return self.head(self.ln_f(x))
```

### Tuning Checklist
- [ ] `lr` 用 cosine schedule，warmup 1000 步，peak `3e-4`
- [ ] AdamW，`betas=(0.9, 0.95)`，`weight_decay=0.1`（GPT 标配）
- [ ] gradient clipping 1.0，防止 loss 突然爆炸
- [ ] 训练时 loss 应从 ~10 (log(50000)) 降到 ~1.5 才算正常

### Common Pitfalls
- causal mask 写错是最常见 bug，自己用小例子验证 `att` 矩阵形状
- positional embedding 别忘了

---

## Day 3 — Batched SE(3) in PyTorch

> 你熟悉 cuRobo，本日**重点是把那套思路搬到 PyTorch tensor 上**。VLA / DP 模型经常需要把动作表示在 SE(3) 空间，且要 batch 处理。
> Week 4 Day 22 会补机器人动力学（M, C, G, OSC），本日先不涉及。

### 早晨论文 (1h)
- *A micro Lie theory for state estimation in robotics* (Solà 2020) §1-3

### Daily Tasks
1. 写 `quat_to_rotmat`, `rotmat_to_quat`（batch 维度任意）
2. 写 `se3_exp`（se(3) → SE(3)），`se3_log`（反之）
3. 写 `compose(T1, T2)`, `inverse(T)`，全部 batched
4. 跑 1000 个随机 SE(3)，验证 `log(exp(x)) ≈ x`

### Code Template

```python
# day3_se3/lie.py
import torch

def hat(omega):  # [B, 3] -> [B, 3, 3]
    O = torch.zeros(*omega.shape[:-1], 3, 3, device=omega.device)
    O[..., 0, 1] = -omega[..., 2]; O[..., 0, 2] =  omega[..., 1]
    O[..., 1, 0] =  omega[..., 2]; O[..., 1, 2] = -omega[..., 0]
    O[..., 2, 0] = -omega[..., 1]; O[..., 2, 1] =  omega[..., 0]
    return O

def so3_exp(omega):  # [B, 3] -> [B, 3, 3]
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    K = hat(omega / theta)
    I = torch.eye(3, device=omega.device).expand(*omega.shape[:-1], 3, 3)
    return I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta)).unsqueeze(-1) * (K @ K)

def se3_exp(xi):  # [B, 6] (rho, omega) -> [B, 4, 4]
    rho, omega = xi[..., :3], xi[..., 3:]
    R = so3_exp(omega)
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    K = hat(omega / theta)
    V = torch.eye(3, device=xi.device) \
        + ((1 - torch.cos(theta)) / theta).unsqueeze(-1) * K \
        + ((theta - torch.sin(theta)) / theta).unsqueeze(-1) * (K @ K)
    t = (V @ rho.unsqueeze(-1)).squeeze(-1)
    T = torch.eye(4, device=xi.device).expand(*xi.shape[:-1], 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T
```

### Tuning Checklist
- [ ] 单测：`torch.allclose(se3_log(se3_exp(xi)), xi, atol=1e-5)`
- [ ] batch 维度 `[B, N, 6]` 要能正常广播
- [ ] 数值稳定：`theta` 接近 0 时用 Taylor 展开

### Common Pitfalls
- 直接用四元数乘法时小心 `(w, x, y, z)` 还是 `(x, y, z, w)` 顺序
- 不同库（pytorch3d / scipy / pin / curobo）的约定不一样，写注释

---

## Day 4 — MDP & Tabular RL

### 早晨论文 (1.5h)
- Sutton & Barto《Reinforcement Learning》第 3-4 章（MDP / DP）
- 自己推导 Bellman 方程：`V*(s) = max_a [r + γ Σ p(s'|s,a) V*(s')]`

### Daily Tasks
1. 实现 FrozenLake-v1（4x4 + 8x8）的 Value Iteration
2. 实现 Policy Iteration，对比收敛速度
3. 实现 Tabular Q-Learning（ε-greedy）
4. 画收敛曲线，理解 γ 对策略行为的影响

### Code Template

```python
# day4_mdp/value_iteration.py
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=True)
nS, nA = env.observation_space.n, env.action_space.n
P = env.unwrapped.P  # P[s][a] = [(prob, s', r, done), ...]

def value_iteration(gamma=0.99, theta=1e-8):
    V = np.zeros(nS)
    while True:
        delta = 0
        for s in range(nS):
            v_old = V[s]
            V[s] = max(sum(p * (r + gamma * V[s2]) for p, s2, r, _ in P[s][a]) for a in range(nA))
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta: break
    pi = np.array([np.argmax([sum(p * (r + gamma * V[s2]) for p, s2, r, _ in P[s][a]) for a in range(nA)]) for s in range(nS)])
    return V, pi
```

### Tuning Checklist
- [ ] γ=0.9 / 0.95 / 0.99，看策略保守/激进度变化
- [ ] FrozenLake 8x8 上 VI 应能 100 步内收敛
- [ ] Q-Learning 的 ε 用线性衰减（1.0 → 0.05 over 10k 步）

---

## Day 5 — REINFORCE & A2C

### 早晨论文 (1.5h)
- Sutton & Barto 第 13 章（Policy Gradient）
- 自己推导 PG 定理：`∇J(θ) = E_τ[Σ ∇log π(a|s) · G_t]`

### Daily Tasks
1. 在 CartPole-v1 实现 REINFORCE（无 baseline）
2. 加 baseline（用 V(s) 网络），变成 A2C
3. 加 entropy bonus，对比探索效果
4. 把训练曲线发到 wandb

### Code Template

```python
# day5_pg/a2c.py
import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, act_dim)
        self.v  = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.shared(x)
        return self.pi(h), self.v(h).squeeze(-1)

def train(env_id="CartPole-v1", episodes=1000, gamma=0.99, lr=3e-4, ent_coef=0.01):
    env = gym.make(env_id)
    net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(episodes):
        obs, _ = env.reset()
        log_probs, values, rewards, entropies = [], [], [], []
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            logits, v = net(obs_t)
            dist = Categorical(logits=logits)
            a = dist.sample()
            obs, r, term, trunc, _ = env.step(a.item())
            done = term or trunc
            log_probs.append(dist.log_prob(a)); values.append(v); rewards.append(r); entropies.append(dist.entropy())
        # compute returns
        R, returns = 0, []
        for r in reversed(rewards):
            R = r + gamma * R; returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        log_probs = torch.stack(log_probs); values = torch.stack(values); entropies = torch.stack(entropies)
        adv = returns - values.detach()
        loss = -(log_probs * adv).mean() + 0.5 * (returns - values).pow(2).mean() - ent_coef * entropies.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0: print(f"ep {ep}, return {sum(rewards):.1f}")
```

### Tuning Checklist
- [ ] CartPole 应在 ~500 episode 内稳定到 reward = 500
- [ ] 没 baseline 时方差大，曲线抖动明显
- [ ] entropy_coef 从 0.01 调到 0，看探索性变化

---

## Day 6 — PPO + GAE

> 这是本周最重要的一天。PPO 是后面所有 RL 训练的基础。

### 早晨论文 (1.5h)
- *Proximal Policy Optimization Algorithms* (Schulman 2017)
- *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (Schulman 2016)

### Daily Tasks
1. 实现 GAE 计算函数
2. 实现 PPO clipped surrogate loss
3. 加上 value clipping、orthogonal init、advantage normalization
4. 在 CartPole 收敛，再上 LunarLander，最后 HalfCheetah-v4 (mujoco)

### Code Template (PPO 核心)

```python
# day6_ppo/ppo.py — 关键片段
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advs, last = [], 0
    for t in reversed(range(len(rewards))):
        next_v = values[t+1] if t+1 < len(values) else 0
        delta = rewards[t] + gamma * next_v * (1 - dones[t]) - values[t]
        last = delta + gamma * lam * (1 - dones[t]) * last
        advs.insert(0, last)
    return advs

def ppo_update(net, opt, batch, clip=0.2, vf_coef=0.5, ent_coef=0.0, epochs=10):
    obs, acts, old_logp, advs, returns = batch
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    for _ in range(epochs):
        logits, values = net(obs)
        dist = Categorical(logits=logits)
        new_logp = dist.log_prob(acts)
        ratio = (new_logp - old_logp).exp()
        surr1 = ratio * advs
        surr2 = ratio.clamp(1 - clip, 1 + clip) * advs
        pi_loss = -torch.min(surr1, surr2).mean()
        v_loss  = (returns - values).pow(2).mean()
        ent     = dist.entropy().mean()
        loss = pi_loss + vf_coef * v_loss - ent_coef * ent
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()
```

### Tuning Checklist (PPO 黄金超参)
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `lr` | `3e-4` | 离散动作 |
| `clip_eps` | 0.2 | 经典值，调到 0.1 更保守 |
| `gae_lambda` | 0.95 | |
| `gamma` | 0.99 | |
| `epochs` per update | 10 | 多了会过拟合采样数据 |
| `n_steps` (rollout) | 2048 | 越大越稳但慢 |
| `n_envs` (vec env) | 8-32 | 加速采样 |
| `ent_coef` | 0.0 (离散) / 0.0 (连续) | 看任务，探索难时 0.01 |
| `vf_coef` | 0.5 | |
| `max_grad_norm` | 0.5 | |

### Common Pitfalls
- **GAE 计算时 `dones` 处理错**：episode 结尾要把 advantage truncate 掉
- **没做 advantage normalization**：方差大，不收敛
- **每个 update 用太多 epoch**：旧策略和新策略偏差大，clip 频繁触发
- **vectorized env 的 obs 形状**：`[n_envs, obs_dim]` vs `[obs_dim]`

---

## Day 7 — 周报 + 复盘

### Daily Tasks
1. 写一篇 500-800 字 weekly recap，包含：
   - 学了什么（论文 / 代码）
   - 卡了哪里 / 怎么解的
   - 下周计划
2. 整理 GitHub repo，写好每个 day 的 README
3. 把 wandb 上漂亮的曲线截图放进 recap
4. **重点回顾**：手画一张图，说明 PPO 的数据流（rollout → GAE → update → 重复）

### Recap 模板

```markdown
# Week 1 Recap — PyTorch + RL Foundations

## ✅ Done
- 手写 MNIST / mini-GPT / SE(3) batched ops
- 跑通 Value Iteration / REINFORCE / A2C / PPO
- PPO 在 CartPole 100% 成功，HalfCheetah reward ~3000

## 🔥 Insights
- 原本以为 PPO 难在算法本身，实际难在工程细节（GAE / advantage norm / vec env）
- transformer 的 mask 必须 unit test，不能靠肉眼检查

## ⚠️ Pitfalls Hit
- ...

## ➡️ Next Week (Isaac Lab)
- ...
```

---

## 本周总投入预估

| 活动 | 小时 |
|------|------|
| 论文阅读 | 10h |
| 代码实现 | 25h |
| 调参 / 跑实验 | 15h |
| 周报 / 整理 | 5h |
| **合计** | **55h** |

---

## 进入 Week 2 前的准备

- [ ] 注册 NVIDIA NGC 账号（Isaac 系列要用）
- [ ] 预留 100GB 磁盘空间（数据集会膨胀）
- [ ] 把 PPO 代码模板化，下周 RL 训练直接复用

→ [Week02.md](./Week02.md)
