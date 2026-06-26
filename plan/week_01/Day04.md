# Day 04 — MDP & Tabular RL（Bellman / VI / PI / Q-Learning）

> **日期**：2026-06-24（Week 1, Day 4）
>
> **本日定位**：把 RL 的「**根**」打牢 —— Bellman 方程、Value Iteration、Policy Iteration、Tabular Q-Learning。这是后面 Day 5（PG/A2C）、Day 6（PPO/GAE）、Week 4（Franka RL）、Week 7（轮式双臂 residual RL）的共同地基。**今天不碰深度网络**，全部 numpy + Gymnasium。
>
> **总投入**：约 8 小时

---

## 0. 出发前自检（5 分钟）

```bash
[embai]$ conda activate embai
[embai]$ python -c "import gymnasium as gym; e=gym.make('FrozenLake-v1'); print(e.observation_space, e.action_space)"
# 期望：Discrete(16) Discrete(4)

[embai]$ python -c "import numpy, matplotlib, wandb; print('ok')"
[embai]$ wandb status
[embai]$ cd ~/embodied-ai-bootcamp-8w
[embai]$ git status                # clean
[embai]$ git log --oneline | head  # 看到 Day 3 commit
```

**今天主战场**：`week1-fundamentals/day4_mdp/`

```bash
[embai]$ mkdir -p ~/embodied-ai-bootcamp-8w/week1-fundamentals/day4_mdp/{src,tests,configs,plots}
[embai]$ cd ~/embodied-ai-bootcamp-8w/week1-fundamentals/day4_mdp
```

---

## 1. 今日时间表（hour by hour）

| 时段 | 时长 | 任务 | 产出 |
|------|------|------|------|
| 07:00 - 08:30 | 1.5h | Sutton & Barto §3-4 + 手推 Bellman | 笔记 1 页，能默写 4 个 Bellman 方程 |
| 09:00 - 10:30 | 1.5h | FrozenLake-v1 4x4 Value Iteration | VI 收敛曲线 + 最优策略可视化 |
| 10:30 - 12:00 | 1.5h | Policy Iteration（PE + PI） | PI vs VI 收敛步数对比 |
| 14:00 - 15:30 | 1.5h | Tabular Q-Learning（ε-greedy） | 4x4 上 success rate ≥ 0.7 |
| 15:30 - 17:00 | 1.5h | 8x8 + γ 扫描 + ε 衰减实验 | 4 张 wandb 曲线 |
| 19:30 - 20:30 | 1h   | SARSA vs Q-Learning + on/off-policy 对比 | 一图说清楚区别 |
| 20:30 - 21:00 | 0.5h | 复盘 + commit + push | Day 4 commit |

---

## 2. 早晨 (07:00 - 08:30) — 论文 + 概念

### 任务
- 精读：*Reinforcement Learning: An Introduction* (Sutton & Barto, 2nd ed.) **§3 (Finite MDPs) + §4 (DP)**
- 重点小节：3.5 (Policies and Value Functions)、3.6 (Optimal Policies)、4.3 (Policy Iteration)、4.4 (Value Iteration)
- 可选视频：David Silver UCL RL Lecture 2 (MDP) + Lecture 3 (DP)

### 笔记产出 — `day_plan/notes/Day04_mdp.md`

必须能默写并解释：

1. **MDP 五元组** `(S, A, P, R, γ)`
   ```
   S — 状态空间        A — 动作空间
   P(s'|s,a) — 转移概率 R(s,a,s') — 奖励
   γ ∈ [0, 1) — 折扣因子（γ=1 + 无穷 horizon 会发散，必须 < 1 或 episodic）
   ```

2. **4 个 Bellman 方程（必须闭眼默写）**
   ```
   ① V_π(s)  = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [r + γ V_π(s')]            （评估）
   ② Q_π(s,a)= Σ_{s',r} p(s',r|s,a) [r + γ Σ_{a'} π(a'|s') Q_π(s',a')]
   ③ V*(s)   = max_a Σ_{s',r} p(s',r|s,a) [r + γ V*(s')]                  （最优）
   ④ Q*(s,a) = Σ_{s',r} p(s',r|s,a) [r + γ max_{a'} Q*(s',a')]
   ```

3. **三件事必须能口头答**
   - **VI vs PI 谁快？** —— PI 单次迭代贵（要 Policy Evaluation 内循环至收敛），但**外层迭代次数极少**（FrozenLake 通常 3-5 次）；VI 单次迭代便宜（只 1 次 backup），但**外层迭代多**（几十到上百次）。**总 wall-clock 通常 PI 略快或持平**。
   - **γ 的物理意义？** —— γ→1：长视野，策略保守，看重远期 reward；γ→0：短视，只盯下一步。FrozenLake 上 γ=0.99 比 γ=0.9 更愿意绕远路避开冰窟。
   - **on-policy (SARSA) vs off-policy (Q-Learning)？** —— Q-Learning 用 `max_{a'} Q(s',a')` 更新（不管你实际怎么选），所以学的是**贪心策略下**的 Q；SARSA 用 `Q(s', a'_actual)` 更新，所以学的是**当前 ε-greedy 行为策略下**的 Q。Q-Learning 在 cliff-walking 上学得"激进且最优"，SARSA 学得"保守但安全"。

4. **手推 Bellman optimality（必做）**
   - 拿一张白纸，从 `V*(s) = max_π V_π(s)` 出发
   - 推到 `V*(s) = max_a [r + γ Σ p(s'|s,a) V*(s')]`
   - 关键步：交换 max 和 Σ_π 的顺序需要 deterministic optimal policy 的存在性证明
   - 写在笔记里，看到就能对着 Bellman backup 一行行解释

### 自检
- [ ] 闭眼默写 ① - ④ 四个 Bellman 方程
- [ ] 能解释为什么 contraction mapping → VI 一定收敛（γ < 1 是关键）
- [ ] 能口头说清 SARSA / Q-Learning / Expected SARSA 的 target 各是什么

---

## 3. 上午 09:00 - 10:30 — Value Iteration

> **本日不用 Bellman expectation 推导新东西**，直接把上面 ③ 写成代码。Gymnasium 的 FrozenLake 把 `P[s][a]` 直接给你，**不需要自己模拟环境**，纯 numpy 算就行。

### 3.1 FrozenLake 环境探索（5 分钟）

```bash
[embai]$ python -c "
import gymnasium as gym
env = gym.make('FrozenLake-v1', is_slippery=True, map_name='4x4')
print('nS =', env.observation_space.n)
print('nA =', env.action_space.n)
P = env.unwrapped.P
# P[s][a] = [(prob, next_state, reward, done), ...]
print('P[0][0] =', P[0][0])     # 从 state 0 走 LEFT
print('P[14][2] =', P[14][2])   # 从倒数第二格走 RIGHT，看会不会到 G
"
```

> `is_slippery=True` —— 走的方向有 1/3 概率执行，2/3 概率滑到两侧（这才是有意思的 stochastic MDP）。先用 slippery 训出来才算合格。

### 3.2 实现 Value Iteration

```python
# day4_mdp/src/value_iteration.py
import numpy as np
import gymnasium as gym

def value_iteration(env, gamma=0.99, theta=1e-8, max_iter=10000):
    """
    Returns:
        V: [nS] state value
        pi: [nS] greedy policy
        history: list[float] 每次迭代的 max delta，用来画收敛曲线
    """
    nS, nA = env.observation_space.n, env.action_space.n
    P = env.unwrapped.P
    V = np.zeros(nS)
    history = []

    for it in range(max_iter):
        delta = 0.0
        V_new = V.copy()                     # synchronous update
        for s in range(nS):
            q_sa = np.zeros(nA)
            for a in range(nA):
                for prob, s2, r, done in P[s][a]:
                    q_sa[a] += prob * (r + gamma * V[s2] * (not done))
            V_new[s] = q_sa.max()
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        history.append(delta)
        if delta < theta:
            break

    # 一次 greedy policy extraction
    pi = np.zeros(nS, dtype=int)
    for s in range(nS):
        q_sa = np.zeros(nA)
        for a in range(nA):
            for prob, s2, r, done in P[s][a]:
                q_sa[a] += prob * (r + gamma * V[s2] * (not done))
        pi[s] = q_sa.argmax()

    return V, pi, history


def evaluate_policy(env, pi, episodes=1000, max_steps=200):
    """跑实际 episode 测 success rate（FrozenLake 到 G 才 reward=1）"""
    successes = 0
    for _ in range(episodes):
        s, _ = env.reset()
        for _ in range(max_steps):
            s, r, term, trunc, _ = env.step(int(pi[s]))
            if term or trunc:
                successes += int(r > 0)
                break
    return successes / episodes


if __name__ == "__main__":
    for slippery in [False, True]:
        env = gym.make("FrozenLake-v1", is_slippery=slippery, map_name="4x4")
        V, pi, hist = value_iteration(env, gamma=0.99)
        sr = evaluate_policy(env, pi, episodes=1000)
        print(f"slippery={slippery}: VI 收敛 {len(hist)} 步, success={sr:.3f}")
        print("V:", V.reshape(4, 4).round(3))
        print("π:", pi.reshape(4, 4))   # 0=L, 1=D, 2=R, 3=U
```

### 3.3 期望结果

```
slippery=False: VI 收敛 ~7 步,  success=1.000
slippery=True:  VI 收敛 ~30-60 步, success≈0.74 (slippery 上限本就 ~0.82,有运气成分)
```

### 3.4 策略可视化（必做）

```python
# day4_mdp/src/viz.py
import numpy as np, matplotlib.pyplot as plt

ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}
TILES  = "SFFF FHFH FFFH HFFG"            # FrozenLake-v1 4x4 默认地图

def plot_policy(V, pi, shape, title, save):
    H, W = shape
    fig, ax = plt.subplots(figsize=(W, H))
    ax.imshow(V.reshape(H, W), cmap="viridis")
    for i in range(H):
        for j in range(W):
            ax.text(j, i, ARROWS[pi[i*W + j]], ha="center", va="center",
                    fontsize=20, color="white")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(save, dpi=120, bbox_inches="tight")
```

### 自检
- [ ] `slippery=False` 上 VI 在 10 步内收敛，success=1.0
- [ ] `slippery=True` 上 VI 收敛，success > 0.7
- [ ] 画出来的箭头图能看出策略「绕开 H」的意图
- [ ] `done=True` 的 transition 一定要把 `gamma * V[s2]` 截断（用 `* (not done)`），否则 terminal 之外的"幻觉收益"会污染 V

---

## 4. 上午 10:30 - 12:00 — Policy Iteration

### 4.1 实现

```python
# day4_mdp/src/policy_iteration.py
import numpy as np
import gymnasium as gym

def policy_evaluation(P, pi, nS, nA, gamma, theta=1e-8, max_iter=10000):
    V = np.zeros(nS)
    for _ in range(max_iter):
        delta = 0.0
        V_new = V.copy()
        for s in range(nS):
            a = pi[s]
            v = 0.0
            for prob, s2, r, done in P[s][a]:
                v += prob * (r + gamma * V[s2] * (not done))
            V_new[s] = v
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < theta:
            break
    return V

def policy_improvement(P, V, nS, nA, gamma):
    pi = np.zeros(nS, dtype=int)
    for s in range(nS):
        q_sa = np.zeros(nA)
        for a in range(nA):
            for prob, s2, r, done in P[s][a]:
                q_sa[a] += prob * (r + gamma * V[s2] * (not done))
        pi[s] = q_sa.argmax()
    return pi

def policy_iteration(env, gamma=0.99):
    nS, nA = env.observation_space.n, env.action_space.n
    P = env.unwrapped.P
    pi = np.zeros(nS, dtype=int)
    pe_steps = []                                 # 每轮 PE 内部迭代步数

    for it in range(1000):
        V = policy_evaluation(P, pi, nS, nA, gamma)
        pi_new = policy_improvement(P, V, nS, nA, gamma)
        pe_steps.append(it)
        if np.array_equal(pi_new, pi):
            return V, pi, it + 1, pe_steps
        pi = pi_new
    return V, pi, 1000, pe_steps
```

### 4.2 VI vs PI 对比

```python
# day4_mdp/src/compare_vi_pi.py
import time
import gymnasium as gym
from value_iteration import value_iteration, evaluate_policy
from policy_iteration import policy_iteration

env = gym.make("FrozenLake-v1", is_slippery=True, map_name="8x8")

t0 = time.time()
V_vi, pi_vi, hist_vi = value_iteration(env, gamma=0.99)
t_vi = time.time() - t0

t0 = time.time()
V_pi, pi_pi, n_outer, _ = policy_iteration(env, gamma=0.99)
t_pi = time.time() - t0

print(f"VI: {len(hist_vi)} 步, {t_vi*1000:.1f}ms, success={evaluate_policy(env, pi_vi):.3f}")
print(f"PI: {n_outer} 外层步, {t_pi*1000:.1f}ms, success={evaluate_policy(env, pi_pi):.3f}")
import numpy as np
print("两者策略一致性:", (pi_vi == pi_pi).mean())
```

### 4.3 期望结果（8x8 slippery）

```
VI: ~80 步, ~30ms, success ≈ 0.45
PI: ~5  外层步, ~25ms, success ≈ 0.45
策略一致性 > 0.9（不等于 1 是因为 V 在 1e-8 阈值内可能多个 a 等效）
```

### 自检
- [ ] PI 外层迭代次数 < 10
- [ ] PI 和 VI 的最终 V 接近（差距 < 1e-5），策略大体一致
- [ ] 知道 PI 的"真正贵"在 PE 内层 —— 它是另一种 VI（固定 π 的 VI），所以单步成本高
- [ ] **能解释为什么 Modified PI（PE 只跑 5-10 步而不是收敛）通常最快**

---

## 5. 下午 14:00 - 15:30 — Tabular Q-Learning

> **VI/PI 假设你知道 P** —— 这是上帝视角。真世界不知道 P，得通过和环境交互学。Q-Learning 是从这跨进 model-free 的第一步。

### 5.1 实现

```python
# day4_mdp/src/q_learning.py
import numpy as np
import gymnasium as gym
import wandb

def epsilon_schedule(step, total, eps_start=1.0, eps_end=0.05):
    """linear decay over `total` steps"""
    frac = min(step / total, 1.0)
    return eps_start + frac * (eps_end - eps_start)

def q_learning(env_id="FrozenLake-v1", map_name="4x4", is_slippery=True,
               total_steps=50_000, alpha=0.1, gamma=0.99,
               eps_decay_steps=10_000, eval_every=1000, run_name="q_4x4"):
    env = gym.make(env_id, is_slippery=is_slippery, map_name=map_name)
    eval_env = gym.make(env_id, is_slippery=is_slippery, map_name=map_name)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    rng = np.random.default_rng(0)

    wandb.init(project="embai-day04", name=run_name,
               config=dict(alpha=alpha, gamma=gamma, total_steps=total_steps,
                           eps_decay_steps=eps_decay_steps, map_name=map_name,
                           is_slippery=is_slippery))

    s, _ = env.reset()
    ep_return, ep_returns = 0.0, []

    for t in range(total_steps):
        eps = epsilon_schedule(t, eps_decay_steps)
        if rng.random() < eps:
            a = rng.integers(nA)
        else:
            a = int(Q[s].argmax())

        s2, r, term, trunc, _ = env.step(a)
        # Q-Learning target: r + γ max_a' Q(s', a')，terminal 时 future 项为 0
        target = r + gamma * Q[s2].max() * (not term)
        Q[s, a] += alpha * (target - Q[s, a])

        ep_return += r
        s = s2
        if term or trunc:
            ep_returns.append(ep_return); ep_return = 0.0
            s, _ = env.reset()

        if (t + 1) % eval_every == 0:
            sr = eval_greedy(eval_env, Q, episodes=200)
            recent = np.mean(ep_returns[-100:]) if ep_returns else 0.0
            wandb.log({"step": t, "epsilon": eps, "success_rate": sr,
                       "ep_return_recent": recent})

    wandb.finish()
    return Q

def eval_greedy(env, Q, episodes=200, max_steps=200):
    suc = 0
    for _ in range(episodes):
        s, _ = env.reset()
        for _ in range(max_steps):
            s, r, term, trunc, _ = env.step(int(Q[s].argmax()))
            if term or trunc:
                suc += int(r > 0); break
    return suc / episodes

if __name__ == "__main__":
    Q = q_learning(map_name="4x4", total_steps=50_000, run_name="q_4x4_slippery")
```

```bash
[embai]$ python src/q_learning.py
```

### 5.2 期望结果（4x4 slippery, 50k 步）

```
最终 success rate ≈ 0.70 ~ 0.78（VI 上限 ~0.82，Q-Learning 因为 sample-based 略低）
ep_return_recent 应在 25k 步附近爬升到稳定区
```

### 自检
- [ ] 4x4 slippery 上 success > 0.7
- [ ] wandb 曲线能看到 epsilon 线性下降 + success_rate 阶梯上升
- [ ] 知道为什么 `done=True` 时不能加 `gamma * Q[s2].max()`（terminal 之外没有未来）

---

## 6. 下午 15:30 - 17:00 — 8x8 + γ 扫描 + ε 衰减消融

### 6.1 跑 4 组实验（必做）

```python
# day4_mdp/src/sweep.py
from q_learning import q_learning

# A. 4x4 baseline
q_learning(map_name="4x4", total_steps=50_000, run_name="A_4x4_g099_eps1to005")

# B. 8x8（更难，更多步）
q_learning(map_name="8x8", total_steps=300_000, eps_decay_steps=100_000,
           run_name="B_8x8_g099_eps1to005")

# C. γ 扫描（4x4）
for g in [0.90, 0.95, 0.99]:
    q_learning(map_name="4x4", total_steps=50_000, gamma=g,
               run_name=f"C_4x4_gamma{g}")

# D. ε 衰减时长扫描（4x4）
for ed in [2_000, 10_000, 30_000]:
    q_learning(map_name="4x4", total_steps=50_000, eps_decay_steps=ed,
               run_name=f"D_4x4_epsdecay{ed}")
```

### 6.2 期望观察

| 实验 | 预期 | 解读 |
|------|------|------|
| **A** 4x4 baseline | success → 0.7+ | sanity 通过 |
| **B** 8x8 | 300k 步 success → 0.3-0.5 | tabular 在大状态空间显著吃力 |
| **C γ=0.90** | success 显著低 | γ 太小 → reward 信号传不远 → 不知道往 G 走 |
| **C γ=0.99** | 收敛慢但最终最好 | γ 大 → V 的 horizon 长 → 更耐心绕路 |
| **D ε 衰减 2k** | success 偶尔卡 0.5 不动 | 探索不够，陷入 sub-optimal |
| **D ε 衰减 30k** | 后期还在乱探索 | 利用不够，但最终更稳 |

> **写到笔记**：γ 和 ε-schedule 是 RL 两个**最常被新手忽视、但杀伤力最大**的超参。今天的 C / D 两组实验是肌肉记忆。

### 6.3 wandb 出图

```bash
# 在 wandb 网页 → embai-day04 项目 → 把 success_rate 曲线分组（group_by run_name 前缀），
# 截图存到 day4_mdp/plots/sweep.png
```

### 自检
- [ ] 4 组实验全在 wandb 上有曲线
- [ ] γ=0.9 vs 0.99 的对比图能一眼看出差距
- [ ] ε 衰减的 trade-off 你能口头说清

---

## 7. 晚上 19:30 - 20:30 — SARSA vs Q-Learning（on/off-policy）

> 课本上反复讲的「Cliff Walking」例子今晚跑一次，**亲眼看到 SARSA 学得保守、Q-Learning 学得激进且最优**。

### 7.1 实现 SARSA

```python
# day4_mdp/src/sarsa.py
import numpy as np
import gymnasium as gym

def sarsa(env_id="CliffWalking-v0", total_episodes=500, alpha=0.5, gamma=1.0,
          eps=0.1):
    env = gym.make(env_id)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    rng = np.random.default_rng(0)
    returns = []

    def pick(s):
        return rng.integers(nA) if rng.random() < eps else int(Q[s].argmax())

    for ep in range(total_episodes):
        s, _ = env.reset()
        a = pick(s)
        ep_ret = 0.0
        while True:
            s2, r, term, trunc, _ = env.step(a)
            ep_ret += r
            a2 = pick(s2)
            target = r + gamma * Q[s2, a2] * (not term)   # SARSA: 用实际选的 a2
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s2, a2
            if term or trunc:
                break
        returns.append(ep_ret)
    return Q, returns

def q_learning_cliff(total_episodes=500, alpha=0.5, gamma=1.0, eps=0.1):
    env = gym.make("CliffWalking-v0")
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    rng = np.random.default_rng(0)
    returns = []
    for ep in range(total_episodes):
        s, _ = env.reset()
        ep_ret = 0.0
        while True:
            a = rng.integers(nA) if rng.random() < eps else int(Q[s].argmax())
            s2, r, term, trunc, _ = env.step(a)
            ep_ret += r
            target = r + gamma * Q[s2].max() * (not term)  # Q-Learning: max
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
            if term or trunc:
                break
        returns.append(ep_ret)
    return Q, returns
```

### 7.2 对比图

```python
# day4_mdp/src/compare_on_off.py
import numpy as np, matplotlib.pyplot as plt
from sarsa import sarsa, q_learning_cliff

_, sarsa_ret = sarsa()
_, ql_ret    = q_learning_cliff()

def smooth(x, w=20):
    return np.convolve(x, np.ones(w)/w, mode="valid")

plt.plot(smooth(sarsa_ret), label="SARSA (on-policy)")
plt.plot(smooth(ql_ret),    label="Q-Learning (off-policy)")
plt.xlabel("episode"); plt.ylabel("return (smoothed)")
plt.legend(); plt.title("Cliff Walking: SARSA vs Q-Learning")
plt.savefig("plots/cliff_sarsa_vs_ql.png", dpi=120, bbox_inches="tight")
```

### 7.3 期望观察

```
Q-Learning return 稳定线 ≈ -13（最优，沿悬崖边走）
SARSA      return 稳定线 ≈ -17（保守，绕开悬崖）

两条曲线在前 200 episode 都剧烈震荡，是因为 ε=0.1 探索时 Q-Learning 偶尔掉崖（return -100）。
```

### 自检（必须能口答）
- [ ] **为什么 Q-Learning 学最优、SARSA 学保守？** —— Q-Learning 用 `max_a' Q(s',a')` 学的是「假设我之后都贪心」的 Q，所以它推荐沿悬崖边走（贪心下确实最优）；SARSA 用 `Q(s', a'_actual)` 学的是「我有 ε 概率乱走」的 Q，悬崖边一旦乱走就 -100，所以它学会绕远。
- [ ] **真机部署该选哪个？** —— 通常 on-policy 系（PPO / SARSA）更安全，因为「学到的策略 = 部署的策略」；off-policy 系（DQN / SAC / Q-Learning）样本效率高但 deployment gap 风险大。

---

## 8. 晚上 20:30 - 21:00 — 复盘 + commit + push

### 8.1 写复盘

```bash
cd ~/embodied-ai-bootcamp-8w/week1-fundamentals/day4_mdp
cat > NOTES.md <<'EOF'
# Day 4 Recap (2026-06-24)

## Done
- [x] Bellman 方程 ①②③④ 闭眼默写
- [x] FrozenLake 4x4 VI: ___ 步收敛, success=___
- [x] FrozenLake 8x8 VI: ___ 步收敛, success=___
- [x] Policy Iteration 外层 ___ 步收敛, 与 VI 一致性 ___
- [x] Q-Learning 4x4: success=___ @ 50k steps
- [x] Q-Learning 8x8: success=___ @ 300k steps
- [x] γ 扫描: γ=0.9 success=___, 0.95=___, 0.99=___
- [x] ε 衰减扫描: 完成 4 组 wandb run
- [x] Cliff Walking 上 SARSA(-___) vs Q-Learning(-___) 对比图

## Insights
- VI 单步便宜但外层多, PI 单步贵但外层只 5 步左右, Modified PI 是甜点
- γ=0.99 比 0.9 在 FrozenLake 上明显更优, 因为 reward 稀疏
- Q-Learning 学最优 Cliff Walking 是因为 max 算子等于"假设之后贪心"
- SARSA 受 ε 影响, 部署时切到 ε=0 行为会和训练时不一致

## Pitfalls hit
- 一开始 done=True 时没截断 future 项, V 出现幻觉收益, success 停在 0.4
- ε 衰减太快(2k)在 4x4 卡 sub-optimal, 改 10k 后过

## Tomorrow (Day 5 - REINFORCE & A2C)
- 早晨: Sutton & Barto §13 (Policy Gradient)
- 上午: REINFORCE on CartPole-v1
- 下午: 加 baseline → A2C, 加 entropy bonus
- 晚上: 把今天 Q-Learning 和明天 PG 做一次对比写进 NOTES
EOF
```

### 8.2 commit & push

```bash
cd ~/embodied-ai-bootcamp-8w
git add week1-fundamentals/day4_mdp/
git commit -m "Day 4: tabular RL — VI / PI / Q-Learning / SARSA on FrozenLake & Cliff"
git push origin main
```

### 8.3 Milestones 打勾

```
[x] 手写 MNIST 训练循环                                  ✅ Day 1
[x] 手写 Mini-GPT, Tiny Shakespeare 训练 1 epoch 能采样   ✅ Day 2
[x] Batched SE(3) 操作通过 log(exp(x)) ≈ x 单测           ✅ Day 3
[x] 实现 Value Iteration / Q-Learning                     ✅ 今天打勾
[ ] 实现 PPO + GAE
[ ] 5 个实验全部上 wandb（Day 1 ~ Day 4 应已 4 个）
```

---

## 9. 今日完成确认（必达 / 加分）

### 必达（5/6 才算合格）
- [ ] FrozenLake 4x4 / 8x8 VI 实现 + 收敛 + success > 0.7（4x4）
- [ ] Policy Iteration 实现 + 与 VI 策略一致性 > 0.9
- [ ] Q-Learning 4x4 success > 0.7
- [ ] γ 扫描 + ε 衰减扫描，4 组 wandb run
- [ ] Cliff Walking 上 SARSA vs Q-Learning 对比图
- [ ] 4 个 Bellman 方程闭眼默写

### 加分
- [ ] 8x8 Q-Learning success > 0.5（tabular 在 8x8 是 hard mode）
- [ ] 实现 Expected SARSA，跟 SARSA / Q-Learning 三者对比
- [ ] 实现 Modified PI（PE 只跑 5 步），证实是 VI/PI 之间的甜点
- [ ] 把 V/π 的 4x4 / 8x8 热力图 + 箭头图存到 `plots/`

---

## 10. 明天 (Day 5) 预告 — REINFORCE & A2C

- 早晨：Sutton & Barto §13（Policy Gradient）+ 推 PG 定理
- 上午：CartPole-v1 上手写 REINFORCE（无 baseline）
- 下午：加 V(s) baseline → A2C，加 entropy bonus
- 晚上：跟 Day 4 Q-Learning 做一次「value-based vs policy-based」对比

预读：
- *Sutton & Barto §13.1-13.5*
- 自己推：`∇J(θ) = E_τ[Σ_t ∇log π(a_t|s_t) · G_t]`，知道 baseline 不改 unbiasedness 但减 variance

---

## 11. 反向提醒（容易忘）

- ❗ **`done=True` 一定要把 `gamma * V[s2]`（或 `gamma * Q[s2].max()`）截断**，否则 terminal 之外幻觉收益污染 V/Q —— 这是新手最常翻车点
- ❗ **VI 用 synchronous update**（用 `V_new` 数组）比 in-place 慢但和教科书对得上；in-place（Gauss-Seidel）实战更快但收敛性证明要重写
- ❗ **γ < 1 是 contraction mapping 的关键**，γ=1 + 非 episodic 不收敛
- ❗ **FrozenLake 默认 reward 稀疏**（只有到 G 才 +1），所以 γ=0.99 比 0.9 好；其他 dense reward 任务可能反过来
- ❗ **Q-Learning 的 max 是 off-policy 的来源**，把它换成 `Q[s2, a2_actual]` 就是 SARSA
- ❗ **今天不写 deep network，所有 Q 都是 numpy 数组** —— DQN（Q + neural network）是 Day 6 之前不要碰
- ❗ **wandb run 命名要规整**（`A_4x4_g099_eps1to005`），不然一周后回来看一片 `run-1 / run-2`

---

## 12. 如果今天彻底崩了 — 救火方案

如果 14:00 还没把 VI 跑通：

1. **降目标**：今天只完成 VI（4x4 + 8x8）+ Q-Learning（4x4），PI / SARSA / γ 扫描 / Cliff 推到 Day 7 周末补
2. **抄一段**：把 [`dennybritz/reinforcement-learning`](https://github.com/dennybritz/reinforcement-learning) 的 `DP/` 和 `TD/` 直接抄来读懂，而不是从零写
3. **保 Day 5 不延期**：PG 比 tabular 更重要 —— tabular 后面除了帮你理解 baseline / advantage 几乎不再用，PG 是 PPO / VLA action head 的根

---

→ [Day05.md](./Day05.md)（待生成） | [Week01.md](../week_plan/Week01.md)
