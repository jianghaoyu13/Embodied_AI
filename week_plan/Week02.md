# Week 2 — Isaac Lab + LeRobot + 模仿学习入门

> **本周定位**：从「单环境 Gym 玩具」过渡到「真实机器人仿真」，并在仿真器上完成第一个 Behavior Cloning / ACT 项目。

---

## 本周目标

完成本周后，你应该能：
- [ ] 独立安装并运行 Isaac Lab，理解 Asset / Sensor / Manager / Env 抽象
- [ ] 在 MuJoCo 和 Isaac Lab 之间自由切换，知道各自适用场景
- [ ] 录制并加载 LeRobot 格式的数据集
- [ ] 复现 ACT (Action Chunking Transformer)，在 ALOHA Cube Transfer 任务达到 ≥80% 成功率
- [ ] 写一个 Behavior Cloning baseline，作为后续 Diffusion Policy 的对照

---

## 本周可交付物

```
week2-imitation/
├── day8_isaaclab/        # IsaacLab 安装记录 + 跑通 Lift Cube
├── day9_mujoco_genesis/  # MuJoCo MJX + Genesis 体验
├── day10_lerobot/        # LeRobot 数据格式 + 自录 Push-T
├── day11_bc/             # Behavior Cloning baseline
├── day12_act/            # ACT 复现
├── day13_act_tune/       # ACT 调参 + 视频可视化
└── day14_recap/          # 周报
```

---

## Day 8 — Isaac Lab 入门

### 早晨论文 (1h)
- *Orbit / Isaac Lab* 官方 paper（Mittal 2023）
- 浏览 Isaac Lab 文档 Architecture 章节

### Daily Tasks
1. 安装 Isaac Sim 4.5+ 和 Isaac Lab（约 50GB，预留时间 2-3 小时）
2. 跑通官方 tutorial：
   - `Isaac-Cartpole-v0`
   - `Isaac-Lift-Cube-Franka-v0`（这是你后面所有操作任务的起点）
3. 阅读 `omni.isaac.lab.envs` 模块的源码
4. 用随机策略跑 100 个 episode，画成功率分布

### 安装命令（基于你已有的 Isaac Sim 6.0）

```bash
# 激活 isaaclab 环境
conda activate isaaclab

# 如果你已经通过 pip 装好了 Isaac Sim 6.0（见 README Day 0 步骤 4a），直接装 Isaac Lab：
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install

# 如果你是通过 Omniverse Launcher 装的 Isaac Sim 6.0：
# 需要先 export ISAACSIM_PATH 指向你的安装目录，比如：
# export ISAACSIM_PATH=~/.local/share/ov/pkg/isaac-sim-6.0.0
# 然后再 ./isaaclab.sh --install

# 验证 Isaac Lab 安装
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Reach-Franka-v0 --headless --num_envs 4096

# 如果要看可视化（非 headless）：
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Lift-Cube-Franka-v0 --num_envs 16
```

> **注意**：Isaac Lab 的 `./isaaclab.sh` 是入口脚本，它会自动设置 Isaac Sim 的 Python 路径。
> 你**不需要手动启动 Isaac Sim GUI**来跑 RL 训练——全部通过命令行 + `--headless` 搞定。

### ROS2 Jazzy 桥接（你已装好 Jazzy，可直接利用）

Isaac Sim 6.0 内置 `ros2_bridge` 扩展，原生支持 ROS2 Jazzy。在 Week 2 你可以暂不深入，但建议**Day 8 跑一个 hello world 验证 bridge**：

```bash
# 启动 Isaac Sim（需要 ROS2 环境已 source）
source /opt/ros/jazzy/setup.bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --enable_ros2

# 在另一终端订阅默认 topic
ros2 topic list
```

> 用途预告：
> - Week 4：用 ROS2 publish Pinocchio 算的 `tau`，订阅 Isaac Sim 关节状态
> - Week 7：作为 sim2real bridge，同一套节点既能驱动 Isaac Sim，又能切到真机

### 关键概念速记

| 概念 | 类比 cuRobo / 传统机器人学 | 用途 |
|------|------------------------|------|
| `Articulation` | URDF 加载后的对象 | 控制关节 |
| `RigidObject` | 静态/动态物体 | 抓取目标 |
| `SceneEntityCfg` | 场景配置 | 声明谁是谁 |
| `ActionTerm` | 控制器接口 | joint pos / IK / OSC |
| `ObservationTerm` | 观测器 | 关节角 / 末端 pose / 图像 |
| `RewardTerm` | 奖励函数模块 | 奖励组合 |
| `EventTerm` | 随机化 | DR (domain randomization) |

### Tuning Checklist
- [ ] GPU 利用率：`Isaac-Lift-Cube-Franka-v0` 用 4096 envs 时应跑满 90%+
- [ ] FPS（steps/sec/env）：< 100k 说明哪里瓶颈
- [ ] 可视化只在 debug 时开（`--headless` 加速训练 5-10x）

### Common Pitfalls
- **NVIDIA 驱动版本**：必须 ≥535，否则 Isaac Sim 启动报错
- **首次启动慢**：会下载很多 USD asset，10-30 分钟，正常
- **Vulkan 错误**：装 `libvulkan1` `vulkan-utils`
- **不要把 `Isaac-` 前缀的 env 名拼错**，找不到环境时是这个原因

---

## Day 9 — MuJoCo MJX + Genesis 速览

### Daily Tasks
1. 安装 `mujoco` + `mujoco-mjx`，跑官方 cartpole demo
2. 体验 MJX 的 GPU 并行（10000 envs 同时跑）
3. 安装 Genesis（`pip install genesis-world`），跑 quick start
4. 写一个 markdown 表格对比四个仿真器（Isaac Lab / MuJoCo / MJX / Genesis）

### 对比表（你应该能填出来）

| 维度 | Isaac Lab | MuJoCo (CPU) | MuJoCo MJX | Genesis |
|------|-----------|--------------|------------|---------|
| GPU 加速 | ✅ 强 | ❌ | ✅ JAX | ✅ Taichi |
| 渲染质量 | 工业级 | 中 | 中 | 高 |
| 接触模拟 | PhysX | Soft | Soft | 多种 |
| 学习曲线 | 陡 | 平 | 中 | 中 |
| 适用场景 | locomotion / manip | RL benchmarks | locomotion | 全栈 |

### Code Template (MJX 并行 rollout)

```python
# day9_mujoco_genesis/mjx_parallel.py
import mujoco
import mujoco.mjx as mjx
import jax, jax.numpy as jnp

m = mujoco.MjModel.from_xml_path("cartpole.xml")
mjx_m = mjx.put_model(m)

@jax.jit
@jax.vmap
def rollout(key):
    d = mjx.make_data(mjx_m)
    def step(carry, _):
        d = carry
        d = mjx.step(mjx_m, d)
        return d, d.qpos
    _, traj = jax.lax.scan(step, d, None, length=200)
    return traj

keys = jax.random.split(jax.random.PRNGKey(0), 10000)
trajs = rollout(keys)  # [10000, 200, qpos_dim]
```

---

## Day 10 — LeRobot 数据格式 + 自录 Push-T

### 早晨论文 (1h)
- LeRobot 项目 README + 一篇 datasets 教程
- *Push-T* 任务原始 paper（Implicit BC, Florence 2021）

### Daily Tasks
1. `pip install lerobot` 并跑通示例
2. 下载 `lerobot/pusht` 数据集，看其 parquet 结构
3. 用键盘录制自己的 Push-T 数据集（30 条）
4. 把数据上传 HuggingFace Hub（练习数据集打包）

### Code Template (加载 LeRobot 数据)

```python
# day10_lerobot/load_dataset.py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("lerobot/pusht")
print(ds)
print(ds.features)
print(ds[0].keys())     # observation.image, observation.state, action, ...

# 关键：如何切片成训练序列
delta_timestamps = {
    "observation.image": [-0.1, 0.0],          # 当前 + 0.1s 前
    "observation.state": [-0.1, 0.0],
    "action": [t * 0.1 for t in range(16)],    # 未来 16 步动作
}
ds = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)
```

### Tuning Checklist
- [ ] 录制时帧率稳定 30Hz（用 `time.sleep` 校准）
- [ ] 每条 episode 长度 100-300 步
- [ ] 检查 action 分布无 spike（手抖会留 outlier）

---

## Day 11 — Behavior Cloning Baseline

### 早晨论文 (1h)
- Implicit BC (Florence 2021) 第 1-3 节
- 复习：BC = Supervised Learning on (s, a) pairs

### Daily Tasks
1. 在 LeRobot Push-T 上写 BC（MLP + image encoder）
2. 用 `tqdm` + `wandb` 训练 10 epoch
3. 在仿真中评估，记录 100 episode 成功率
4. 输出 baseline 数字（应 ~30-40% on Push-T，DP 后续要打到 ~90%）

### Code Template

```python
# day11_bc/bc_pusht.py
import torch, torch.nn as nn
from torchvision.models import resnet18
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

class BCPolicy(nn.Module):
    def __init__(self, act_dim=2, state_dim=2):
        super().__init__()
        self.vision = resnet18(weights="DEFAULT")
        self.vision.fc = nn.Identity()  # 512-d feature
        self.head = nn.Sequential(
            nn.Linear(512 + state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim),
        )
    def forward(self, image, state):
        feat = self.vision(image)
        return self.head(torch.cat([feat, state], dim=-1))

# train loop 略，按 Day 1 模板改
```

### Tuning Checklist
- [ ] `lr=1e-4`，AdamW
- [ ] 图像 normalize 用 ImageNet mean/std
- [ ] 数据增强：random crop / color jitter，提升 5-10%
- [ ] BC 在 Push-T 不会超过 50%（这就是 DP 存在的理由）

---

## Day 12 — ACT (Action Chunking Transformer) 复现

### 早晨论文 (2h)
- *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (Zhao 2023)
- 重点理解：
  1. Action Chunking 为什么 work（动作连续性 vs 单步贪心）
  2. CVAE 训练目标
  3. Temporal Ensemble 推理

### Daily Tasks
1. clone `tonyzhaozh/act` 官方 repo
2. 在 `aloha_sim_transfer_cube` 任务训练 + 评估
3. 把核心模块（CVAE encoder, Transformer decoder）画成图
4. 自己重写 inference loop，加上 temporal ensemble

### ACT 关键超参

| 参数 | 推荐值 | 含义 |
|------|--------|------|
| `chunk_size` | 100 | 每次预测未来 100 步 |
| `kl_weight` | 10 | CVAE KL 散度权重 |
| `hidden_dim` | 512 | Transformer width |
| `dim_feedforward` | 3200 | FFN 宽度 |
| `lr` | `1e-5`(backbone) / `1e-4`(其他) | |
| `batch_size` | 8 | 显存 24GB 上限 |
| `num_epochs` | 2000 | 长，但能看到平稳收敛 |

### Code Template (Inference with Temporal Ensemble)

```python
# day12_act/inference.py
class TemporalEnsembleAgent:
    def __init__(self, policy, chunk_size=100, k=0.01):
        self.policy = policy
        self.chunk_size = chunk_size
        self.k = k  # 指数权重衰减率
        self.history = []  # [(t, action_chunk), ...]

    def step(self, obs, t):
        chunk = self.policy(obs)  # [chunk_size, act_dim]
        self.history.append((t, chunk))
        # 收集所有覆盖时刻 t 的预测
        candidates = []
        weights = []
        for t_pred, c in self.history:
            offset = t - t_pred
            if 0 <= offset < self.chunk_size:
                candidates.append(c[offset])
                weights.append(np.exp(-self.k * offset))
        candidates = np.stack(candidates)
        weights = np.array(weights); weights /= weights.sum()
        return (weights[:, None] * candidates).sum(0)
```

### Tuning Checklist
- [ ] 没用 temporal ensemble 时成功率会差 10-20%
- [ ] 显存不够：`hidden_dim=384`、`batch_size=4`
- [ ] backbone（ResNet18）lr 一定要小，否则会破坏预训练特征

### Common Pitfalls
- **CVAE 不收敛**：检查 KL weight，太大会 collapse 到 prior
- **训练 loss 低但 eval 差**：典型 distribution shift，加更多 DR
- **action 抖动**：必须开 temporal ensemble

---

## Day 13 — ACT 调参 + 可视化

### Daily Tasks
1. 跑 3 组消融：
   - 不用 chunk（chunk_size=1）
   - 不用 CVAE（VAE encoder 关闭）
   - 不用 temporal ensemble
2. 把 rollout 视频上传 wandb
3. 画消融 bar chart，量化每个组件贡献
4. 试着把 ACT 应用到 LeRobot Push-T（迁移性练习）

### Wandb Video Logging

```python
import wandb
import imageio
frames = []  # [H, W, 3] uint8 list
for _ in range(200):
    obs = env.get_obs()
    a = agent.step(obs, t)
    obs, r, done, _ = env.step(a)
    frames.append(env.render())
imageio.mimsave("rollout.mp4", frames, fps=30)
wandb.log({"video/rollout": wandb.Video("rollout.mp4")})
```

### 消融结果（参考目标）

| 配置 | 成功率 |
|------|--------|
| Full ACT | 80-90% |
| no chunk | 30-40% |
| no CVAE | 60-70% |
| no temporal ensemble | 65-75% |

---

## Day 14 — 周报 + 第一阶段总结

### Daily Tasks
1. 写 weekly recap（800-1000 字）
2. 把 ACT 训练视频和成功率柱状图整理成一篇博客草稿
3. **重要**：画一张图，说明 BC vs ACT 的核心差异
4. 准备 Week 3 的 Diffusion Policy 论文，先读一遍

### 自检：你应该能回答
1. 为什么 ACT 用 CVAE 而不是普通 transformer？
2. Action chunking 的副作用是什么？
3. 在什么情况下你会选 BC 而不是 ACT？
4. 如果让你把 ACT 部署到真机，你会担心什么？

---

## 本周工程化要点（必须养成习惯）

1. **每个实验都用 hydra**：

```yaml
# configs/act_pusht.yaml
defaults: [_self_]
seed: 42
task: pusht
chunk_size: 100
hidden_dim: 512
lr_backbone: 1e-5
lr: 1e-4
batch_size: 8
epochs: 2000
```

```python
import hydra
@hydra.main(config_path="configs", config_name="act_pusht")
def main(cfg): ...
```

2. **每次跑实验前 git commit**：用 wandb 记录 git hash

```python
import subprocess
sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
wandb.init(project="...", config={"git_sha": sha, **cfg})
```

3. **用 tmux 跑长训练**：

```bash
tmux new -s act_train
# 跑训练
# Ctrl+B 然后 D 脱离
tmux attach -t act_train
```

---

## 本周总投入预估

| 活动 | 小时 |
|------|------|
| Isaac Lab 安装 + 摸索 | 8h |
| 论文阅读 | 12h |
| 代码实现 | 18h |
| 训练 / 调参 | 15h |
| 周报 | 5h |
| **合计** | **58h** |

→ [Week03.md](./Week03.md)
