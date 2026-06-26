# Day 06 — PPO + GAE（Week 1 主菜）

> **日期**：2026-06-26（Week 1, Day 6）
> **本日定位**：**这是 Week 1 最重要的一天**。PPO 是后面所有 RL 训练的基础 —— Week 4 Franka RL、Week 7 轮式双臂 residual RL、甚至 RLHF 风格的 VLA 训练都基于它。今天必须把 PPO + GAE + vec env + 完整 trick 打通，让它成为你"夹紧调出来"的通用工具。
> **总投入**：约 8.5 小时（重要日，可放宽 0.5h）

---

## 0. 出发前自检（5 分钟）

```bash
[embai]$ conda activate embai
[embai]$ python -c "
import gymnasium as gym, torch
print('torch:', torch.__version__)
env = gym.make('HalfCheetah-v5')
print('HalfCheetah obs:', env.observation_space.shape, 'act:', env.action_space.shape)
"
# 期望：obs (17,) act (6,) — 连续动作空间
# 若 mujoco 没装：pip install 'gymnasium[mujoco]'

[embai]$ python -c "from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv; print('vec ok')"
[embai]$ wandb status
[embai]$ cd ~/embodied-ai-bootcamp-8w
[embai]$ git status                # clean
[embai]$ git log --oneline | head  # 看到 Day 5 commit
```

**今天主战场**：`week1-fundamentals/day6_ppo/`

```bash
[embai]$ mkdir -p ~/embodied-ai-bootcamp-8w/week1-fundamentals/day6_ppo/{src,configs,plots}
[embai]$ cd ~/embodied-ai-bootcamp-8w/week1-fundamentals/day6_ppo
```

---

## 1. 今日时间表（hour by hour）

| 时段 | 时长 | 任务 | 产出 |
|------|------|------|------|
| 07:00 - 08:30 | 1.5h | Schulman 2017 PPO + Schulman 2016 GAE | 笔记 1 页，能默写 GAE / clip 公式 |
| 09:00 - 10:30 | 1.5h | 实现 GAE + PPO loss + vec env | rollout buffer 跑通 |
| 10:30 - 12:00 | 1.5h | 接上 actor-critic 网络 + 完整训练循环 | CartPole 收敛 |
| 14:00 - 15:30 | 1.5h | 加齐 11 件 trick（orth init / value clip / norm） | LunarLander 收敛 |
| 15:30 - 17:00 | 1.5h | HalfCheetah-v5（连续动作 PPO） | reward > 3000 |
| 19:30 - 20:30 | 1h   | PPO 黄金超参表 + clip 消融 | 一图说清 clip 作用 |
| 20:30 - 21:30 | 1h   | 复盘 + commit + push（多花 0.5h 整理） | Day 6 commit + PPO 模板 |

---

## 2. 早晨 (07:00 - 08:30) — 论文 + 概念

### 任务
- 精读：*Proximal Policy Optimization Algorithms* (Schulman et al., 2017)
- 精读：*High-Dimensional Continuous Control Using GAE* (Schulman et al., 2016)
- 必读博客：[The 37 Implementation Details of PPO (Huang 2022)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) —— **本日的实战 bible**
- 浏览：CleanRL 的 `ppo.py` 和 `ppo_continuous_action.py` 源码（< 200 行很值得读）

### 笔记产出 — `day_plan/notes/Day06_ppo.md`

必须能默写并解释：

1. **GAE 公式（必须闭眼默写）**
   ```
   δ_t = r_t + γ V(s_{t+1}) (1 - done_t) - V(s_t)            (TD error)
   A_t^{GAE(γ,λ)} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}                (geometric weighted)
                  = δ_t + γλ (1 - done_t) A_{t+1}              (recursion)
   返回 R_t = A_t + V(s_t)                                    (作为 critic target)
   ```
   - `λ=0`: GAE = TD(0)，低方差高偏差
   - `λ=1`: GAE = Monte Carlo，高方差零偏差
   - `λ=0.95`: 经典甜点

2. **PPO Clipped Surrogate Loss（必须闭眼默写）**
   ```
   r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)        (probability ratio)

   L^CLIP = E_t [min( r_t · A_t,  clip(r_t, 1-ε, 1+ε) · A_t )]

   总 loss = -L^CLIP + c1 · L^VF - c2 · H[π]
           = pi_loss + 0.5 * v_loss - 0.0 * entropy   (c1=0.5, c2 看任务)
   ```

3. **三件事必须能口头答**
   - **clip 在抑制什么？** —— 抑制 ratio · A 在「ratio 偏离 1 太远」时继续推梯度。当 A>0 且 ratio>1+ε 时梯度被截断（不让你"越来越偏"）；A<0 且 ratio<1-ε 时同理。**直觉**：信任域的便宜近似 —— 不让一步走太大。
   - **为什么 PPO 是 on-policy 但能 K-epoch 重用 rollout？** —— 重用旧数据时用 importance ratio `r_t` 修正分布漂移；clip 保证漂移不会失控。所以 PPO 是 "approximately on-policy"，K-epoch 不能太大（默认 4-10），否则 ratio 偏离过大频繁触发 clip 等于浪费。
   - **GAE 的 bias-variance trade-off？** —— 用 V_φ 估的 bootstrap 引偏差但降方差；MC return 没偏差但方差大；λ 是连续旋钮。**实操**：连续动作高维任务（HalfCheetah）λ=0.95 几乎是 default。

4. **手算一个 GAE（5 分钟练习）**
   - 假设 rewards = [1, 1, 1, 1] dones = [0, 0, 0, 1] V = [10, 9, 8, 7] γ=0.99 λ=0.95
   - 倒推算 δ_3 ... δ_0，再算 A_3 ... A_0
   - 验证：A_t + V_t ≈ rewards-to-go + bootstrap

### 自检
- [ ] 闭眼能默写 GAE 递推式
- [ ] 闭眼能默写 L^CLIP
- [ ] 能解释 ratio 偏离 1 ± ε 时梯度行为
- [ ] 手算 GAE 例子能对上

---

## 3. 上午 09:00 - 10:30 — GAE + Rollout Buffer + Vec Env

### 3.1 RolloutBuffer

```python
# day6_ppo/src/buffer.py
import numpy as np, torch

class RolloutBuffer:
    """固定容量 buffer:n_steps × n_envs × dim,采集满后一次性算 GAE。"""
    def __init__(self, n_steps, n_envs, obs_shape, act_shape, device="cpu"):
        self.n_steps, self.n_envs = n_steps, n_envs
        self.device = device
        T, N = n_steps, n_envs
        self.obs       = np.zeros((T, N, *obs_shape), dtype=np.float32)
        self.actions   = np.zeros((T, N, *act_shape), dtype=np.float32)
        self.log_probs = np.zeros((T, N),             dtype=np.float32)
        self.rewards   = np.zeros((T, N),             dtype=np.float32)
        self.dones     = np.zeros((T, N),             dtype=np.float32)
        self.values    = np.zeros((T, N),             dtype=np.float32)
        self.advs      = np.zeros((T, N),             dtype=np.float32)
        self.returns   = np.zeros((T, N),             dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, done, value):
        i = self.ptr
        self.obs[i] = obs; self.actions[i] = action
        self.log_probs[i] = log_prob; self.rewards[i] = reward
        self.dones[i] = done; self.values[i] = value
        self.ptr += 1

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        """T 时刻之后的 bootstrap 用 last_value([n_envs])"""
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            self.advs[t] = last_gae
        self.returns = self.advs + self.values

    def get_minibatches(self, batch_size, shuffle=True):
        T, N = self.n_steps, self.n_envs
        flat = lambda x: x.reshape(T * N, *x.shape[2:])
        data = dict(obs=flat(self.obs), actions=flat(self.actions),
                    log_probs=flat(self.log_probs), advs=flat(self.advs),
                    returns=flat(self.returns), values=flat(self.values))
        idx = np.arange(T * N)
        if shuffle: np.random.shuffle(idx)
        for start in range(0, T * N, batch_size):
            mb = idx[start:start + batch_size]
            yield {k: torch.tensor(v[mb], device=self.device) for k, v in data.items()}

    def reset(self):
        self.ptr = 0
```

### 3.2 Vector Env

```python
# day6_ppo/src/envs.py
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

def make_env(env_id, seed, idx):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)   # 自动统计 ep_return / ep_len
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env
    return thunk

def make_vec_env(env_id, n_envs=8, seed=0, async_=False):
    fns = [make_env(env_id, seed, i) for i in range(n_envs)]
    return AsyncVectorEnv(fns) if async_ else SyncVectorEnv(fns)
```

### 自检
- [ ] `RolloutBuffer` 单测通过（`compute_gae` 输出 shape 对得上）
- [ ] vec env 能 `step` 返回 `[n_envs, obs_dim]`
- [ ] 知道 `RecordEpisodeStatistics` 包装后 `info["episode"]` 字段会自动有 `r/l`

---

## 4. 上午 10:30 - 12:00 — Actor-Critic + 完整 PPO 循环

### 4.1 ActorCritic（同时支持离散和连续）

```python
# day6_ppo/src/network.py
import torch, torch.nn as nn
from torch.distributions import Categorical, Normal

def layer_init(layer, std=2.0**0.5, bias=0.0):
    """orthogonal init - PPO 标配"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, continuous=False, hidden=64):
        super().__init__()
        self.continuous = continuous
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),  # 小 std → 初始策略接近确定
        )
        if continuous:
            # state-independent log_std（CleanRL 风），适合 MuJoCo
            self.actor_log_std = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        v = self.get_value(x)
        if self.continuous:
            mean = self.actor_mean(x)
            log_std = self.actor_log_std.expand_as(mean)
            std = log_std.exp()
            dist = Normal(mean, std)
            if action is None: action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            entropy  = dist.entropy().sum(-1)
        else:
            logits = self.actor_mean(x)
            dist = Categorical(logits=logits)
            if action is None: action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy  = dist.entropy()
        return action, log_prob, entropy, v
```

### 4.2 PPO 训练主循环

```python
# day6_ppo/src/ppo.py
import torch, torch.nn as nn
import numpy as np, wandb
from envs import make_vec_env
from network import ActorCritic
from buffer import RolloutBuffer

def train_ppo(env_id="CartPole-v1", continuous=False,
              total_timesteps=500_000, n_envs=8, n_steps=128,
              n_epochs=10, mb_size=256, lr=3e-4,
              gamma=0.99, lam=0.95, clip_eps=0.2,
              vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5,
              clip_vloss=True, anneal_lr=True, target_kl=None,
              run_name="ppo_cartpole", seed=0, device="cuda"):

    torch.manual_seed(seed); np.random.seed(seed)
    envs = make_vec_env(env_id, n_envs=n_envs, seed=seed)
    obs_shape = envs.single_observation_space.shape
    if continuous:
        act_shape = envs.single_action_space.shape
        act_dim = act_shape[0]
    else:
        act_shape = ()
        act_dim = envs.single_action_space.n

    net = ActorCritic(obs_shape[0], act_dim, continuous=continuous).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    buf = RolloutBuffer(n_steps, n_envs, obs_shape, act_shape, device)

    wandb.init(project="embai-day06", name=run_name,
               config=dict(env_id=env_id, lr=lr, gamma=gamma, lam=lam,
                           clip_eps=clip_eps, n_envs=n_envs, n_steps=n_steps,
                           n_epochs=n_epochs, mb_size=mb_size,
                           ent_coef=ent_coef, vf_coef=vf_coef))

    obs, _ = envs.reset(seed=seed)
    n_updates = total_timesteps // (n_envs * n_steps)
    global_step = 0

    for upd in range(1, n_updates + 1):
        if anneal_lr:
            frac = 1.0 - (upd - 1) / n_updates
            for g in opt.param_groups: g["lr"] = frac * lr

        # 1) ROLLOUT
        buf.reset()
        for t in range(n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, log_prob, _, value = net.get_action_and_value(obs_t)
            a_np = action.cpu().numpy()
            next_obs, reward, term, trunc, info = envs.step(a_np)
            done = np.logical_or(term, trunc).astype(np.float32)
            buf.add(obs, a_np, log_prob.cpu().numpy(), reward,
                    done, value.cpu().numpy())
            obs = next_obs
            global_step += n_envs
            # episode return
            if "episode" in info:
                # gymnasium vector 1.0+: info["episode"]["r"] 是 [n_envs] 的数组,带 mask
                ep_r = info["episode"].get("r")
                if ep_r is not None:
                    mask = info.get("_episode", np.zeros_like(ep_r, dtype=bool))
                    if mask.any():
                        wandb.log({"global_step": global_step,
                                   "ep_return": ep_r[mask].mean()})

        # 2) GAE
        with torch.no_grad():
            last_val = net.get_value(torch.tensor(obs, dtype=torch.float32, device=device)).cpu().numpy()
        buf.compute_gae(last_val, gamma=gamma, lam=lam)

        # 3) UPDATE (K epoch)
        for ep_idx in range(n_epochs):
            for mb in buf.get_minibatches(mb_size):
                _, new_logp, entropy, new_v = net.get_action_and_value(
                    mb["obs"], mb["actions"].long() if not continuous else mb["actions"])
                # advantage normalization (per minibatch)
                adv = mb["advs"]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                # ratio
                ratio = (new_logp - mb["log_probs"]).exp()
                # clipped surrogate
                surr1 = ratio * adv
                surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
                pi_loss = -torch.min(surr1, surr2).mean()
                # value loss (with optional clip)
                if clip_vloss:
                    v_pred_clip = mb["values"] + (new_v - mb["values"]).clamp(-clip_eps, clip_eps)
                    v_loss1 = (new_v - mb["returns"]).pow(2)
                    v_loss2 = (v_pred_clip - mb["returns"]).pow(2)
                    v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    v_loss = 0.5 * (new_v - mb["returns"]).pow(2).mean()
                ent = entropy.mean()
                loss = pi_loss + vf_coef * v_loss - ent_coef * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                opt.step()

                # 监控:approx KL,触发 early stop
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (new_logp - mb["log_probs"])).mean().item()
                    clipfrac  = ((ratio - 1).abs() > clip_eps).float().mean().item()
            if target_kl is not None and approx_kl > target_kl:
                break

        wandb.log({"global_step": global_step, "update": upd,
                   "pi_loss": pi_loss.item(), "v_loss": v_loss.item(),
                   "entropy": ent.item(), "approx_kl": approx_kl,
                   "clip_frac": clipfrac, "lr": opt.param_groups[0]["lr"]})

    wandb.finish()
    envs.close()

if __name__ == "__main__":
    # 1) CartPole sanity
    train_ppo(env_id="CartPole-v1", continuous=False,
              total_timesteps=200_000, n_envs=8, n_steps=128,
              run_name="A_ppo_cartpole")
```

```bash
[embai]$ python src/ppo.py
```

### 4.3 期望结果（CartPole-v1）

```
~30 update（约 200k step）后 ep_return 稳定在 500
approx_kl 通常在 0.005 ~ 0.03 之间，clip_frac 5-15%
```

### 自检
- [ ] CartPole 上 ep_return 收敛到 500
- [ ] approx_kl 不会突然飙到 > 0.05
- [ ] `clip_frac` 不会一直接近 0（说明 clip 起了作用）也不会接近 1（说明 ratio 没失控）
- [ ] 整套代码能在 1 分钟内跑完一次 CartPole

---

## 5. 下午 14:00 - 15:30 — 11 件 PPO Trick + LunarLander

> 来自 *The 37 Implementation Details* 博客，必装的 11 件（其余 26 件是次要 / 任务相关）。

### 5.1 11 件 trick 验收清单

```
✅ 1. orthogonal init + final layer std=0.01 (actor) / std=1.0 (critic)
✅ 2. Adam eps=1e-5（默认 1e-8 在 RL 上数值不稳）
✅ 3. lr 线性退火到 0
✅ 4. GAE (γ=0.99, λ=0.95)
✅ 5. minibatch advantage normalization
✅ 6. clipped surrogate loss
✅ 7. clipped value loss（可选,争议小但加无害）
✅ 8. entropy bonus（离散建议加 0.01,连续可 0）
✅ 9. global gradient clipping (max_norm=0.5)
✅ 10. value coef = 0.5
✅ 11. (vec env) reward / obs running normalization（连续动作必备,见下）
```

### 5.2 加 obs / reward normalization（连续任务必备）

```python
# day6_ppo/src/envs.py（追加）
def make_vec_env_norm(env_id, n_envs=8, seed=0):
    fns = [make_env(env_id, seed, i) for i in range(n_envs)]
    envs = SyncVectorEnv(fns)
    envs = gym.wrappers.vector.NormalizeObservation(envs)
    envs = gym.wrappers.vector.TransformObservation(
        envs, lambda o: np.clip(o, -10, 10), envs.observation_space)
    envs = gym.wrappers.vector.NormalizeReward(envs, gamma=0.99)
    envs = gym.wrappers.vector.TransformReward(
        envs, lambda r: np.clip(r, -10, 10))
    return envs
```

> CartPole / LunarLander 离散版本 obs / reward norm 加不加都行；HalfCheetah 等 MuJoCo **必加**，否则 reward 量级跨过 100 时 critic 不稳。

### 5.3 跑 LunarLander

```python
# day6_ppo/src/run_all.py（追加）
train_ppo(env_id="LunarLander-v3",  # 或 v2 看你装的版本
          continuous=False,
          total_timesteps=1_000_000, n_envs=8, n_steps=128,
          ent_coef=0.01,
          run_name="B_ppo_lunar")
```

### 5.4 期望结果

```
~80-120 update 后 ep_return 进入 200+ 区间（landed 成功）
1M step 内稳定到 250-280（v3 的 SB3-zoo baseline 也大致这个区间）
```

### 自检
- [ ] 11 件 trick 全部勾上
- [ ] LunarLander 1M step 稳定在 reward > 200
- [ ] 知道哪些 trick 是 "no-op" 安全可加，哪些是任务相关

---

## 6. 下午 15:30 - 17:00 — HalfCheetah-v5（连续动作 PPO）

> **本节是 Day 6 的硬核**：连续动作 + Normal 分布 + obs/reward norm。Week 4 Franka RL 用的就是这套。

### 6.1 跑 HalfCheetah

```python
# day6_ppo/src/run_cheetah.py
from ppo import train_ppo

train_ppo(env_id="HalfCheetah-v5",
          continuous=True,
          total_timesteps=2_000_000, n_envs=8, n_steps=2048,
          mb_size=64, n_epochs=10,
          lr=3e-4, gamma=0.99, lam=0.95,
          clip_eps=0.2, ent_coef=0.0, vf_coef=0.5,
          run_name="C_ppo_halfcheetah")
```

> 注意：连续任务 `n_steps=2048` 是 PPO 经典配置（2048 × 8 envs = 16k step / update），minibatch 64，10 epoch；这跟离散小任务的 (128, 256, 4) 完全不同。

> 如果用 `train_ppo` 时已经把 obs/reward norm 包进 envs，记得在 vec env 构建时切到 `make_vec_env_norm`：

```python
# 在 ppo.py 里:
from envs import make_vec_env, make_vec_env_norm
envs = make_vec_env_norm(env_id, n_envs=n_envs, seed=seed) if continuous else \
       make_vec_env(env_id, n_envs=n_envs, seed=seed)
```

### 6.2 期望结果

```
2M step 后 ep_return > 3000 算合格
4M step 可上 5000+（CleanRL baseline 大致这个量级）
4090 上单卡跑 2M step 约 15-25 分钟（CPU 瓶颈在 MuJoCo step）
```

### 6.3 自检
- [ ] HalfCheetah 2M step reward > 3000
- [ ] 训练曲线没有"崩塌段"（突然回到 0 之类的）
- [ ] `actor_log_std` 自然下降（从 0 → -0.5 ~ -1.0），动作变 deterministic

---

## 7. 晚上 19:30 - 20:30 — clip 消融 + 黄金超参表

### 7.1 clip 消融实验

```python
# day6_ppo/src/clip_ablation.py
from ppo import train_ppo
for clip in [0.1, 0.2, 0.4, 1.0]:   # 1.0 ≈ 不 clip
    train_ppo(env_id="LunarLander-v3", continuous=False,
              total_timesteps=500_000, clip_eps=clip,
              run_name=f"E_lunar_clip{clip}")
```

期望：
- `clip=0.1`：保守、收敛慢但稳
- `clip=0.2`：经典甜点
- `clip=0.4`：偶尔崩
- `clip=1.0`：策略大跨步，时灵时不灵（这就是为什么没 clip 的 vanilla PG 不稳）

### 7.2 PPO 黄金超参表（贴墙上）

| 参数 | 离散 (CartPole/LunarLander) | 连续 (HalfCheetah/Franka) | 说明 |
|------|----------------------------|--------------------------|------|
| `lr` | `3e-4` | `3e-4` | Adam 默认；跟 Day 5 PG 一致 |
| `gamma` | 0.99 | 0.99 | 视野长就降到 0.95 |
| `gae_lambda` | 0.95 | 0.95 | bias-variance 甜点 |
| `clip_eps` | 0.2 | 0.2 | 保守取 0.1 |
| `n_envs` | 8 | 8 | 多了反而 obs norm 抖 |
| `n_steps` | 128 | 2048 | 长 rollout 对连续任务关键 |
| `n_epochs` | 4-10 | 10 | 太多会过拟合 rollout |
| `minibatch_size` | 256 | 64 | 让 minibatch 数 ≈ 4-32 |
| `ent_coef` | 0.01 | 0.0 | 探索难加 0.001-0.01 |
| `vf_coef` | 0.5 | 0.5 | 共享 backbone 时调到 1.0 |
| `max_grad_norm` | 0.5 | 0.5 | 梯度爆炸保护 |
| `clip_vloss` | True | True | 加无害 |
| `anneal_lr` | True | True | 训练后期更稳 |
| `obs/reward norm` | False | True | 连续任务 reward 量级大必加 |
| `target_kl` (early stop) | None | None or 0.015 | 大模型 RLHF 才常用 |

### 自检
- [ ] 4 条 clip 消融曲线全在 wandb
- [ ] 黄金超参表抄到 `day6_ppo/src/configs/ppo_default.yaml`，下周 Isaac Lab 直接复用
- [ ] 知道每个超参的物理含义和调参方向

---

## 8. 晚上 20:30 - 21:30 — 复盘 + 模板化 + commit

### 8.1 写复盘

```bash
cd ~/embodied-ai-bootcamp-8w/week1-fundamentals/day6_ppo
cat > NOTES.md <<'EOF'
# Day 6 Recap (2026-06-26)

## Done
- [x] GAE 公式默写,手算例子对得上
- [x] PPO L^CLIP 默写
- [x] RolloutBuffer / vec env / ActorCritic / 完整 PPO 全部从零写
- [x] CartPole 200k step 收敛 reward=500
- [x] LunarLander 1M step reward=___
- [x] HalfCheetah 2M step reward=___
- [x] 11 件 PPO trick 全部勾上
- [x] clip 消融 4 组,清楚 clip 的作用

## Insights
- clip 不是越小越好,0.2 是甜点;太小学不动,太大不稳
- approx_kl 是诊断神器,飙到 > 0.05 说明 step 太大需要降 lr 或加 target_kl
- HalfCheetah obs/reward norm 不加直接学不出来,reward 量级 100+ critic 炸
- n_steps × n_envs 决定 rollout 数据量;离散任务 1k 量级,连续任务 16k 量级

## Pitfalls hit
- 一开始忘了 advantage normalization,LunarLander 训练曲线乱跳
- clip_vloss 不开时 v_loss 偶尔爆炸到 100+
- 没用 orth init + final layer std=0.01,初期策略动作太大,马上掉崖
- gymnasium 1.0 后 vector env 的 info["episode"] 形状变了,要看 mask

## Tomorrow (Day 7 - Week 1 Recap)
- 早晨: 写 weekly recap 800 字
- 上午: 整理 GitHub repo,每个 day 写 README
- 下午: PPO 模板化,下周 Isaac Lab 直接复用
- 晚上: 准备 Week 2 (Isaac Sim + LeRobot)
EOF
```

### 8.2 PPO 模板化（关键）

```bash
# 把 day6_ppo/src/ 整体打包成可复用模板
cp -r src ~/embodied-ai-bootcamp-8w/templates/ppo_clean
# 下周 Isaac Lab 训练时直接 import 而不是复制粘贴
```

```bash
# templates/ppo_clean/README.md
cat > ~/embodied-ai-bootcamp-8w/templates/ppo_clean/README.md <<'EOF'
# Clean PPO Template
- 来源: Embodied-AI bootcamp Day 6
- 已验证: CartPole / LunarLander / HalfCheetah-v5
- 用法:
    from ppo import train_ppo
    train_ppo(env_id=..., continuous=True/False, ...)
- 黄金超参在 configs/ppo_default.yaml
EOF
```

### 8.3 commit & push

```bash
cd ~/embodied-ai-bootcamp-8w
git add week1-fundamentals/day6_ppo/ templates/ppo_clean/
git commit -m "Day 6: PPO + GAE + 11 tricks (CartPole/Lunar/HalfCheetah) + reusable template"
git push origin main
```

### 8.4 Milestones 打勾

```
[x] 手写 MNIST 训练循环                                  ✅ Day 1
[x] 手写 Mini-GPT, Tiny Shakespeare                      ✅ Day 2
[x] Batched SE(3) log(exp(x)) 单测                       ✅ Day 3
[x] 实现 Value Iteration / Q-Learning                    ✅ Day 4
[x] 实现 PPO + GAE                                        ✅ 今天打勾
[ ] 5 个实验全部上 wandb（明天周末统计）
```

---

## 9. 今日完成确认（必达 / 加分）

### 必达（5/6 才算合格）
- [ ] GAE 递推式 + L^CLIP 闭眼默写
- [ ] PPO 在 CartPole 收敛到 reward=500
- [ ] PPO 在 LunarLander reward > 200
- [ ] PPO 在 HalfCheetah-v5 reward > 3000
- [ ] 11 件 trick 全部勾上
- [ ] 黄金超参表整理 + PPO 模板化（下周复用）

### 加分
- [ ] clip 消融 4 组完整 wandb 曲线
- [ ] 实现 SAC（off-policy 对照），HalfCheetah reward 对比
- [ ] 用 `target_kl=0.015` 做 early-stop，看影响
- [ ] Atari (BreakoutNoFrameskip-v4) PPO with CNN（加分中的加分）

---

## 10. 明天 (Day 7) 预告 — Week 1 Recap

- 早晨：写 800 字 weekly recap，发到知乎 / 个人博客
- 上午：整理 GitHub repo，每个 day 写 README
- 下午：手画一张 PPO 数据流图（rollout → GAE → update）
- 晚上：准备 Week 2（Isaac Sim 5.1 启动确认 + 6.0 容器 sanity check + LeRobot 装）

---

## 11. 反向提醒（容易忘）

- ❗ **`approx_kl` 飙高（> 0.05）通常意味着 lr 太大或 n_epochs 太多**，先降 lr，不行再降 n_epochs
- ❗ **`clip_frac` 长期 < 1% = clip 没起作用 = ratio 几乎没动 = 几乎是 supervised learning**；长期 > 30% = ratio 一直在被截 = step 太大
- ❗ **连续动作必加 `obs/reward norm`**，离散动作可以不加
- ❗ **`actor_log_std` 用 `nn.Parameter`（state-independent）比 head 输出更稳**，CleanRL 经验
- ❗ **`final layer std=0.01` 是 PPO 的"silent magic"** —— 初始策略接近确定动作但有微小随机性，让训练前期不会乱炸
- ❗ **vec env 的 `info["episode"]` 在 gymnasium 1.0 + 是数组带 mask** —— 老版本是 list of dict，注意兼容
- ❗ **PPO 是 on-policy**，**不能**用旧 update 的 rollout 训新 update —— 每次 rollout 用完即弃；要重用就得 off-policy（SAC / IMPALA）
- ❗ Week 4 Franka RL **要在 Isaac Lab 容器内** 跑（`[isaac6]$`），今天的代码是 host `[embai]$` —— 接 Isaac Lab 时 vec env 会换成 IsaacLab 的 GPU vec env，**RolloutBuffer / 网络可直接复用**

---

## 12. 如果今天彻底崩了 — 救火方案

如果 14:00 还没把 CartPole PPO 跑通：

1. **降目标**：今天只跑通 CartPole（reward → 200+ 即过），LunarLander / HalfCheetah / clip 消融 全部推到 Day 7 周末补
2. **抄一段**：[CleanRL 的 `ppo.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) 是 200 行的最小实现，**直接 fork 改 wandb project name 就能跑**，比从零写快 5x
3. **保 Week 2 不延期**：Day 7 周末写 recap 时不要拖时间，PPO 没跑透就把它列为 Week 1 补作业，**别把 Week 2 Isaac Sim 启动推迟**

---

→ [Day07.md](./Day07.md) | [Week01.md](../week_plan/Week01.md)
