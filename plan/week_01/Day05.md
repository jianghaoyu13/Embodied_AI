# Day 05 — REINFORCE & A2C（Policy Gradient 入门）

> **日期**：2026-06-25（Week 1, Day 5）
> **本日定位**：从 value-based（Day 4 的 VI / Q-Learning）切换到 **policy-based**。手推 PG 定理 → REINFORCE → 加 baseline 变 A2C → 加 entropy bonus。这是明天 PPO 的"前置肌肉记忆"，也是 Week 5/6 VLA action head（ACT / DP / RDT）的根。
> **总投入**：约 8 小时

---

## 0. 出发前自检（5 分钟）

```bash
[embai]$ conda activate embai
[embai]$ python -c "
import torch, gymnasium as gym
print('torch:', torch.__version__)
env = gym.make('CartPole-v1')
print('CartPole obs:', env.observation_space.shape, 'act:', env.action_space.n)
"
# 期望：torch 2.12.0+cu130, obs (4,), act 2

[embai]$ wandb status
[embai]$ cd ~/embodied-ai-bootcamp-8w
[embai]$ git status                # clean
[embai]$ git log --oneline | head  # 看到 Day 4 commit
```

**今天主战场**：`week1-fundamentals/day5_pg/`

```bash
[embai]$ mkdir -p ~/embodied-ai-bootcamp-8w/week1-fundamentals/day5_pg/{src,configs,plots}
[embai]$ cd ~/embodied-ai-bootcamp-8w/week1-fundamentals/day5_pg
```

---

## 1. 今日时间表（hour by hour）

| 时段 | 时长 | 任务 | 产出 |
|------|------|------|------|
| 07:00 - 08:30 | 1.5h | Sutton & Barto §13 + 手推 PG 定理 | 笔记 1 页，能默写 ∇J 公式 |
| 09:00 - 10:30 | 1.5h | REINFORCE on CartPole（无 baseline） | 训练曲线（抖得厉害是正常的） |
| 10:30 - 12:00 | 1.5h | 加 V(s) baseline → A2C | 方差对比图 |
| 14:00 - 15:30 | 1.5h | 加 entropy bonus + reward-to-go + 标准化 | CartPole 稳定 500 reward |
| 15:30 - 17:00 | 1.5h | LunarLander-v2 试水 + 超参扫描 | 5 个 wandb run |
| 19:30 - 20:30 | 1h   | value-based vs policy-based 对比表 | 一图说清楚区别 |
| 20:30 - 21:00 | 0.5h | 复盘 + commit + push | Day 5 commit |

---

## 2. 早晨 (07:00 - 08:30) — 论文 + 概念

### 任务
- 精读：*Reinforcement Learning: An Introduction* (Sutton & Barto, 2nd ed.) **§13.1 - 13.5**
- 重点小节：13.1 (Policy Approximation)、13.2 (PG Theorem 推导)、13.3 (REINFORCE)、13.4 (REINFORCE with Baseline)、13.5 (Actor-Critic)
- 可选视频：Pieter Abbeel CS287 Policy Gradient lecture

### 笔记产出 — `day_plan/notes/Day05_pg.md`

必须能默写并解释：

1. **Policy Gradient 定理（必须闭眼默写）**
   ```
   J(θ) = E_τ~π_θ [R(τ)]                            (target)

   ∇J(θ) = E_τ [Σ_t ∇log π_θ(a_t|s_t) · G_t]        (REINFORCE form)
         = E_(s,a) [∇log π_θ(a|s) · Q_π(s, a)]       (Q form)
         = E_(s,a) [∇log π_θ(a|s) · A_π(s, a)]       (advantage form)
   其中 G_t = Σ_{k≥t} γ^{k-t} r_k     (reward-to-go)
        A_π(s,a) = Q_π(s,a) - V_π(s)   (advantage)
   ```

2. **手推 PG 定理（必做，写在白纸上）**
   - 起点：`J(θ) = Σ_s d_π(s) Σ_a π_θ(a|s) Q_π(s,a)`
   - log-derivative trick：`∇π = π · ∇log π`
   - 关键技巧：忽略 `∇d_π(s)` 项 —— 这是 PG 定理的精华，证明依赖 ergodicity
   - 推到 expectation 形式

3. **三件事必须能口头答**
   - **为什么用 reward-to-go `G_t` 而不是整个 episode 的 R？** —— `t` 之前的 reward 跟 `a_t` 因果无关，期望意义下不影响梯度，但**减小方差**（去掉了不相关的随机项）
   - **baseline 为什么不改 unbiasedness？** —— `E[∇log π · b(s)] = b(s) · E[∇log π] = b(s) · 0 = 0`（log 概率对参数的期望梯度恒为零），所以减任何只与 s 有关的 b(s) 都不偏，但能减方差
   - **Advantage A(s,a) = Q(s,a) - V(s) 是最优 baseline 吗？** —— 不严格是（最优是 `b*(s) = E_a[∇log π² · Q] / E_a[∇log π²]`），但 V(s) 实操中接近最优、最常用、最便宜

4. **REINFORCE → A2C → PPO 的递进逻辑**
   ```
   REINFORCE:   ∇log π · G_t                    (Monte Carlo, 高方差)
   + baseline:  ∇log π · (G_t - V(s_t))         (减方差)
   = A2C:       ∇log π · A(s_t, a_t),  A = G_t - V(s_t)   (Advantage Actor-Critic)
   + bootstrap: A = r + γ V(s') - V(s)          (TD baseline, 引入偏差换方差)
   + GAE:       λ-weighted 平均不同 n-step       (Day 6 内容)
   + PPO clip:  ratio · A 加裁剪                 (Day 6 内容)
   ```

### 自检
- [ ] 闭眼能默写 ∇J(θ) 三种形式（G / Q / Advantage）
- [ ] 能解释 baseline 不改 unbiasedness 的证明（一行）
- [ ] 能口头说清 REINFORCE → A2C → PPO 三步加了什么

---

## 3. 上午 09:00 - 10:30 — REINFORCE（裸版本，先看到方差）

> **本节目的**：先实现最朴素的 REINFORCE，**故意不加 baseline / 不归一化**，亲眼看曲线抖动有多狠。这样后面加 baseline / norm / entropy 时你能感受到每一步带来的改善。

### 3.1 实现

```python
# day5_pg/src/reinforce.py
import torch, torch.nn as nn
import gymnasium as gym
import wandb
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
    def forward(self, x):
        return self.net(x)

def reinforce(env_id="CartPole-v1", episodes=1000, gamma=0.99, lr=1e-3,
              normalize=False, run_name="reinforce_naive"):
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    net = PolicyNet(obs_dim, act_dim)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    wandb.init(project="embai-day05", name=run_name,
               config=dict(episodes=episodes, gamma=gamma, lr=lr,
                           normalize=normalize, env_id=env_id))

    for ep in range(episodes):
        obs, _ = env.reset()
        log_probs, rewards = [], []
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            logits = net(obs_t)
            dist = Categorical(logits=logits)
            a = dist.sample()
            obs, r, term, trunc, _ = env.step(a.item())
            done = term or trunc
            log_probs.append(dist.log_prob(a))
            rewards.append(r)

        # reward-to-go
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(log_probs)
        loss = -(log_probs * returns).mean()        # 注意负号:max J = min -J

        opt.zero_grad(); loss.backward(); opt.step()

        ep_return = sum(rewards)
        wandb.log({"ep": ep, "ep_return": ep_return,
                   "ep_len": len(rewards), "loss": loss.item()}, step=ep)
        if ep % 50 == 0:
            print(f"ep {ep:4d}  return {ep_return:6.1f}  loss {loss.item():+.3f}")

    wandb.finish()

if __name__ == "__main__":
    # 跑两个对比:无标准化 vs 有标准化
    reinforce(normalize=False, run_name="A_reinforce_no_norm")
    reinforce(normalize=True,  run_name="B_reinforce_with_norm")
```

```bash
[embai]$ python src/reinforce.py
```

### 3.2 期望观察

| 实验 | 收敛 episode | 抖动程度 | 解读 |
|------|--------------|---------|------|
| **A** no norm  | ~600-800 | 极抖（return 在 50-500 间反复横跳）| baseline 也没、norm 也没，方差爆炸 |
| **B** with norm | ~400-600 | 中度抖 | 简单 z-score 已显著降方差 |

### 自检
- [ ] A 跑出来曲线明显比 B 抖
- [ ] 没归一化时 loss 数值很大（几百），归一化后稳定在 0.1-1.0
- [ ] 知道 `loss = -(log_probs * returns).mean()` 那个负号怎么来的

---

## 4. 上午 10:30 - 12:00 — 加 V(s) baseline → A2C

### 4.1 实现 ActorCritic

```python
# day5_pg/src/a2c.py
import torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym
import wandb
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, act_dim)
        self.v  = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.shared(x)
        return self.pi(h), self.v(h).squeeze(-1)

def a2c(env_id="CartPole-v1", episodes=1000, gamma=0.99, lr=3e-4,
        ent_coef=0.0, vf_coef=0.5, run_name="a2c"):
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    net = ActorCritic(obs_dim, act_dim)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    wandb.init(project="embai-day05", name=run_name,
               config=dict(episodes=episodes, gamma=gamma, lr=lr,
                           ent_coef=ent_coef, vf_coef=vf_coef, env_id=env_id))

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
            log_probs.append(dist.log_prob(a))
            values.append(v)
            rewards.append(r)
            entropies.append(dist.entropy())

        # returns (Monte Carlo)
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + gamma * G; returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        log_probs = torch.stack(log_probs)
        values    = torch.stack(values)
        entropies = torch.stack(entropies)

        # advantage = G - V(s),detach value 不让 critic 梯度从 actor loss 流回
        adv = (returns - values.detach())
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pi_loss = -(log_probs * adv).mean()
        v_loss  = F.mse_loss(values, returns)
        ent     = entropies.mean()
        loss = pi_loss + vf_coef * v_loss - ent_coef * ent

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()

        wandb.log({"ep": ep, "ep_return": sum(rewards), "ep_len": len(rewards),
                   "pi_loss": pi_loss.item(), "v_loss": v_loss.item(),
                   "entropy": ent.item()}, step=ep)
        if ep % 50 == 0:
            print(f"ep {ep:4d}  return {sum(rewards):6.1f}  pi {pi_loss.item():+.3f}  v {v_loss.item():.3f}  H {ent.item():.3f}")

    wandb.finish()

if __name__ == "__main__":
    # 三组对比:加 baseline / 加 entropy / 都加
    a2c(ent_coef=0.0,  run_name="C_a2c_no_ent")
    a2c(ent_coef=0.01, run_name="D_a2c_ent_001")
    a2c(ent_coef=0.05, run_name="E_a2c_ent_005")
```

```bash
[embai]$ python src/a2c.py
```

### 4.2 期望观察（vs Day 5 § 3 的 REINFORCE）

| 实验 | 收敛 episode | 抖动 | 备注 |
|------|--------------|------|------|
| REINFORCE + norm | ~500 | 中 | baseline 隐含在 norm 里 |
| **A2C ent=0**     | ~300 | 低 | V baseline 显著降方差 |
| **A2C ent=0.01**  | ~250 | 低 | 探索略好 |
| **A2C ent=0.05**  | 500+ | 中 | 探索过头反而慢 |

### 自检
- [ ] A2C 比 REINFORCE 收敛快 1.5-2x
- [ ] critic 的 v_loss 应稳定下降（不抖）
- [ ] entropy 自然下降（从 ln(2)≈0.69 降到 0.05 左右）—— 策略变 deterministic 是好事
- [ ] **关键**：写代码时一定 `values.detach()` 当 baseline，**否则 actor 的梯度会试图调 V 来"骗"自己**

---

## 5. 下午 14:00 - 15:30 — 完整 A2C（reward-to-go + entropy + 标准化）

### 5.1 把上节代码集大成

上节 A2C 已经把 reward-to-go、advantage、entropy bonus 全用上了。本节做两件：

1. **跑一组 baseline run**（A2C, ent=0.01）作为 Day 5 的 best 配置
2. **画方差对比图**：把 REINFORCE / REINFORCE+norm / A2C 三条曲线放一起，**展示每一步带来的方差降低**

```python
# day5_pg/src/plot_variance.py
import wandb, numpy as np, matplotlib.pyplot as plt
api = wandb.Api()
runs = {
    "REINFORCE (no norm)": "A_reinforce_no_norm",
    "REINFORCE (norm)":    "B_reinforce_with_norm",
    "A2C":                 "C_a2c_no_ent",
    "A2C + entropy":       "D_a2c_ent_001",
}
fig, ax = plt.subplots(figsize=(8, 5))
for label, run_name in runs.items():
    run = api.runs("embai-day05", filters={"display_name": run_name})[0]
    h = run.history(keys=["ep_return"])
    ret = h["ep_return"].values
    smooth = np.convolve(ret, np.ones(20)/20, mode="valid")
    ax.plot(smooth, label=label)
ax.set_xlabel("episode"); ax.set_ylabel("return (smoothed, w=20)")
ax.legend(); ax.set_title("CartPole-v1: REINFORCE → A2C 方差对比")
plt.savefig("plots/variance_compare.png", dpi=120, bbox_inches="tight")
```

### 5.2 自检
- [ ] CartPole 至少有 1 个 run **稳定在 500 reward**（CartPole-v1 上限）
- [ ] 方差对比图能看出 A2C 比 REINFORCE 显著更平
- [ ] 知道为什么 CartPole 收敛快但 LunarLander 慢（state 维度高 + reward dense → 高 / sparse → ?）

---

## 6. 下午 15:30 - 17:00 — LunarLander-v2 + 超参扫描

> CartPole 太简单（4-dim state, 2 action），看不出 PG 真正的 trade-off。LunarLander 是 8-dim state, 4 action, reward 较 dense 但 episode 长达 1000，能压出 entropy 调参的效果。

### 6.1 跑 LunarLander

```bash
[embai]$ pip install gymnasium[box2d]   # 若没装
```

```python
# day5_pg/src/run_lunar.py
from a2c import a2c
# LunarLander-v3 在 gymnasium 1.0+ 已是 v3,以你装的版本为准
# 老版本: LunarLander-v2
import gymnasium as gym
ENV = "LunarLander-v3"
try:
    gym.make(ENV)
except gym.error.NameNotFound:
    ENV = "LunarLander-v2"

# baseline
a2c(env_id=ENV, episodes=2000, lr=3e-4, ent_coef=0.01, run_name="F_lunar_a2c")

# lr 扫描
for lr in [1e-4, 3e-4, 1e-3]:
    a2c(env_id=ENV, episodes=2000, lr=lr, ent_coef=0.01,
        run_name=f"G_lunar_lr{lr}")

# entropy 扫描
for ec in [0.0, 0.01, 0.05]:
    a2c(env_id=ENV, episodes=2000, lr=3e-4, ent_coef=ec,
        run_name=f"H_lunar_ent{ec}")
```

### 6.2 期望观察

```
LunarLander-v3:
- A2C 单线程版本 2000 episode 大概到 reward 80-150（onlineCritic + Monte Carlo return 局限）
- 真正的 SOTA 要等 PPO + GAE + 多环境（Day 6）才能稳定 200+
- 所以今天的 LunarLander 是"压力测试"而非"复现 SOTA"
```

### 6.3 必须记录的洞察（写到笔记里）

- **lr 太大（1e-3）** → policy 崩溃，return 卡 -100 不动（飞船坠毁）
- **lr 太小（1e-4）** → 收敛太慢，2000 episode 才到 reward 50
- **lr=3e-4** → 经典甜点（Adam 默认值），下周 PPO 也会用
- **ent=0** → 早期就 deterministic，陷入 sub-optimal
- **ent=0.05** → 一直乱探索，永远飞不稳

### 自检
- [ ] LunarLander 至少有一组 run 跑出 reward > 100
- [ ] 知道为什么 lr=1e-3 会崩（PG loss 没有 PPO 的 clip 保护，一步走太大）
- [ ] **能口头说清明天 PPO 解决了哪些今天的痛**：①步长不可控（→ clip） ② Monte Carlo 高方差（→ GAE） ③ 单环境采样慢（→ vec env）

---

## 7. 晚上 19:30 - 20:30 — value-based vs policy-based

### 7.1 写一张对比表（贴墙上）

| 维度 | Value-based（DQN, Q-Learning） | Policy-based（REINFORCE, A2C） | Actor-Critic（PPO, SAC） |
|------|--------------------------------|-------------------------------|--------------------------|
| **学的对象** | Q(s, a)，策略隐式 = argmax Q | 直接学 π(a\|s) | 同时学 π 和 V/Q |
| **动作空间** | 离散好，连续要 arg-max（NAF / DDPG） | 离散 / 连续都行 | 离散 / 连续都行 |
| **on/off-policy** | off-policy（exp replay 友好）| on-policy（每次更新都要新数据） | on / off 都有 |
| **样本效率** | 高（off-policy + replay） | 低（on-policy） | 中（PPO 是 on-policy 但多 epoch） |
| **稳定性** | 中（DQN 容易 overestim）| 中（高方差）| 高（PPO clip） |
| **真机部署** | exploration 风险大（ε-greedy） | 自然探索（stochastic π） | PPO 推荐 |
| **多模态行为** | 难（argmax 是 unimodal） | 中（看分布族） | 中 |
| **典型用途** | Atari / 离散动作游戏 | 机器人控制（连续 + on-policy 安全）| 大规模 RL（Isaac Lab、ChatGPT-RLHF） |

### 7.2 写一段（笔记里必带）

> 我现在能口头说清三件事:
> 1. **CartPole 我会用什么？** —— A2C/PPO 即可，SAC overkill
> 2. **Franka Lift 我会用什么？** —— PPO + Isaac Lab vec env（Week 4）
> 3. **真机机械臂 first try？** —— BC/ACT/DP（imitation），等 RL 在仿真打稳了再 sim2real（Week 4-7 思路）

### 自检
- [ ] 上面这张表能默写出 4-5 行
- [ ] 知道为什么 Diffusion Policy 是 imitation 而不是 RL（学多模态分布的 capacity 不在 PG 范畴）

---

## 8. 晚上 20:30 - 21:00 — 复盘 + commit + push

### 8.1 写复盘

```bash
cd ~/embodied-ai-bootcamp-8w/week1-fundamentals/day5_pg
cat > NOTES.md <<'EOF'
# Day 5 Recap (2026-06-25)

## Done
- [x] PG 定理三种形式 (G / Q / Advantage) 闭眼默写
- [x] REINFORCE 朴素版本，CartPole 收敛 episode = ___
- [x] REINFORCE + return 标准化，收敛 episode = ___
- [x] A2C (V baseline)，收敛 episode = ___
- [x] A2C + entropy bonus，最稳 ent_coef = ___
- [x] LunarLander 上 A2C 最高 return = ___
- [x] lr / entropy 双扫描，画出曲线
- [x] 方差对比图：REINFORCE → A2C 显著降方差

## Insights
- baseline 不改 unbiasedness,只降方差,这是 advantage 概念的根
- on-policy 的死穴在样本效率,所以明天 PPO 用 K-epoch 重用 rollout
- lr=3e-4 + ent_coef=0.01 + Adam 是 PG 的 default 甜点
- detach value 当 baseline 是必须的(不 detach actor 会去骗 critic)

## Pitfalls hit
- 第一次 loss 没加负号,return 越来越低
- 没 detach baseline,critic 抖得厉害
- LunarLander lr=1e-3 直接坠毁,降到 3e-4 才稳

## Tomorrow (Day 6 - PPO + GAE)
- 早晨: Schulman 2017 (PPO) + Schulman 2016 (GAE)
- 上午: 实现 GAE + PPO clipped loss
- 下午: 加 vec env, CartPole / LunarLander / HalfCheetah
- 晚上: PPO 黄金超参表整理
EOF
```

### 8.2 commit & push

```bash
cd ~/embodied-ai-bootcamp-8w
git add week1-fundamentals/day5_pg/
git commit -m "Day 5: REINFORCE + A2C with baseline / entropy on CartPole + LunarLander"
git push origin main
```

### 8.3 Milestones 打勾

```
[x] 手写 MNIST 训练循环                                  ✅ Day 1
[x] 手写 Mini-GPT, Tiny Shakespeare                      ✅ Day 2
[x] Batched SE(3) log(exp(x)) 单测                       ✅ Day 3
[x] 实现 Value Iteration / Q-Learning                    ✅ Day 4
[ ] 实现 PPO + GAE                                        ← 明天打勾
[ ] 5 个实验全部上 wandb（已 5+ 个,周末统计）
```

---

## 9. 今日完成确认（必达 / 加分）

### 必达（5/6 才算合格）
- [ ] PG 定理三种形式默写
- [ ] REINFORCE + 标准化 在 CartPole 收敛到 reward 500
- [ ] A2C 在 CartPole 收敛速度 > REINFORCE
- [ ] A2C 在 LunarLander reward > 100
- [ ] entropy / lr 至少一组扫描，wandb 出图
- [ ] value-based vs policy-based 对比表

### 加分
- [ ] 实现 n-step return（介于 MC 和 TD 之间，预热 GAE）
- [ ] 加 GAE（明天的内容今天先写一版），看效果
- [ ] HalfCheetah-v4 (mujoco) A2C 跑出 reward > 500
- [ ] 用 `torch.distributions.Normal` 实现连续动作版的 A2C

---

## 10. 明天 (Day 6) 预告 — PPO + GAE

- 早晨：Schulman 2017 (PPO) + Schulman 2016 (GAE)
- 上午：实现 GAE 函数 + PPO clipped surrogate loss
- 下午：加 vec env、advantage normalization、value clipping
- 晚上：CartPole → LunarLander → HalfCheetah 三连，整理黄金超参表

预读：
- *Proximal Policy Optimization Algorithms* (Schulman 2017)
- 推一遍：`L_CLIP = E[min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t)]`，知道 clip 在抑制什么

---

## 11. 反向提醒（容易忘）

- ❗ **`loss = -(log_probs * returns).mean()` 那个负号** —— PG 是 max J，loss 是 min -J
- ❗ **baseline 必须 `values.detach()`**，否则 actor loss 倒灌进 critic
- ❗ **on-policy 不能用旧 rollout 重训** —— 每次 update 完旧的 (s, a, log_prob) 都失效，Day 6 PPO 才用 importance ratio 解决
- ❗ **CartPole 上限 reward=500（v1）**，看到 500 就别期待更高
- ❗ **`action_space.n` 用于 Discrete，连续动作走 `Box` + `Normal` 分布**
- ❗ **`entropy` 自动下降是正常的**（策略变确定），不要看到下降就以为坏了；除非降到 0 还没收敛
- ❗ 今天**不写 vec env**（VectorEnv），明天 PPO 才用 —— 单环境 + 1k episode 足够展示 PG 性质

---

## 12. 如果今天彻底崩了 — 救火方案

如果 14:00 还没把 REINFORCE 跑通：

1. **降目标**：今天只跑通 REINFORCE + 标准化（CartPole reward → 200+ 就算过），A2C / LunarLander / 扫描 推到 Day 7 周末
2. **抄一段**：[`pytorch/examples/reinforcement_learning`](https://github.com/pytorch/examples/tree/main/reinforcement_learning) 的 `reinforce.py` 是官方 60 行版本，比自己从零写快
3. **保 Day 6 不延期**：PPO 是 Week 1 的"主菜"，明天必须按时启动 —— REINFORCE / A2C 算 PPO 的预热，不行就明天补

---

→ [Day06.md](./Day06.md)（待生成） | [Week01.md](../week_plan/Week01.md)
