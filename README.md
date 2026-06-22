# 具身智能算法 · 8 周速成路线图

> 个性化校准版本（已根据你最新背景更新）：
> - 在职实习中
> - 硬件：单卡 RTX 4090 24GB
> - 已有：机器人运动学 ✓、cuRobo 熟练 ✓、DL/RL 概念级、数学较强
> - 弱项：机器人**动力学**、DL 工程实战、端到端项目落地
> - 职业方向：**轮式双臂人形 + 7-DoF 机械臂**的运动规划 / 运动控制 / Pick & Place / 抓取 / 模仿学习 / 强化学习
> - **本计划已剔除 quadruped locomotion 等不相关内容**，全部围绕你的目标平台展开

---

## 一、给你的个性化校准

| 你的强项 | 你的弱项 | 校准策略 |
|---------|---------|---------|
| 机器人运动学、cuRobo | **机器人动力学**（M, C, G, OSC, impedance） | Week 4 Day 22 整天补强，配 Pinocchio + Franka |
| 数学功底 | DL 工程实战 | Week 1 加塞 PyTorch 工程肌肉记忆训练 |
| 概念理解 | RL 实操 | PPO 调参必须亲手跑透；并把 cuRobo 用作 residual baseline |
| 已熟 cuRobo | 端到端项目落地 | Week 7 项目允许「DP + cuRobo residual」，活用你的存量优势 |
| — | VLA 大模型（关键短板） | Week 5-6 全力攻坚 |

**核心原则**：你不缺概念，你缺 **「从零到能跑」的肌肉记忆**。所以这 8 周的设计是：**80% 时间手敲代码与调参，20% 看论文**。

**实验平台已锁定**：所有任务围绕 **Franka / Panda 7-DoF 机械臂、ALOHA 双臂、RoboCasa 厨房 mobile manipulator、自建轮式双臂**，**不训 Go2 / 人形 locomotion**。

---

## 二、文件总览

| 文件 | 主题 | 阶段产出 |
|------|------|---------|
| [Week01.md](./Week01.md) | PyTorch 实战 + SE(3) + RL 入门 | 手写 PPO，CartPole/HalfCheetah 收敛 |
| [Week02.md](./Week02.md) | Isaac Lab + LeRobot + BC/ACT | ALOHA 双臂 Cube Transfer 复现 |
| [Week03.md](./Week03.md) | Diffusion Policy + 表征学习 | Push-T 任务 SOTA 复现 |
| [Week04.md](./Week04.md) | **动力学补强 + 7-DoF RL + 双臂 + 抓取 + 移动操作** | Franka RL + 双臂 + RoboCasa |
| [Week05.md](./Week05.md) | VLM 基础 + LLaVA 微调 | 自录数据微调 Qwen2.5-VL |
| [Week06.md](./Week06.md) | VLA 主流模型深读（π0 / GR00T / RDT） | 在 LIBERO 上微调 OpenVLA |
| [Week07.md](./Week07.md) | **端到端项目（轮式双臂 / 7-DoF）** | 1 个亮眼 GitHub 项目 |
| [Week08.md](./Week08.md) | 论文输出 + 求职冲刺 | 简历、博客、面试准备 |
| [Resources.md](./Resources.md) | 完整资源清单（论文 / 仓库 / 视频） | 收藏夹 |
| [Milestones.md](./Milestones.md) | 8 周里程碑自检表 | 每周打勾 |
| [Day_Plan/](./Day_Plan/) | **逐日精细化执行计划**（hour-by-hour） | 每天直接打开当天的 DayXX.md |

---

## 三、硬件 / 环境前置（Day 0）

### 3.1 你的硬件 / 系统实况

| 配置 | 你的值 |
|------|--------|
| GPU | RTX 4090 (24GB) |
| Driver | **590.48.01** |
| CUDA (Driver API) | **13.1** |
| OS | Ubuntu 24.04|
| ROS2 | **Jazzy** (LTS, 2024-2029) |
| Isaac Sim | 5.1 + **6.0**（主力用 6.0） |
| Isaac Lab | 待装 v3.0+ / main |

> **关于 CUDA 13.1**：
> Driver CUDA Version 13.1 表示你的驱动**最高支持** CUDA 13.1 Runtime。
> 但 PyTorch 的 prebuilt wheels 还没有 `cu131`——最新的是 `cu124`。
> 这完全没问题：CUDA 向后兼容，`cu124` wheels 在 Driver 590 上可以正常运行。
> 如果你后续要编译自定义 CUDA kernel，可以装 CUDA Toolkit 12.4（别装 13.1 toolkit，PyTorch 不兼容）。
> 
> **关于 ROS2 Jazzy**：
> Week 4（动力学 + Pinocchio）和 Week 7（真机/仿真项目）可利用 ROS2 做 Isaac Sim ↔ 真机 bridge。
> Isaac Sim 6.0 内置了 `ros2_bridge` 扩展，直接支持 Jazzy topic 发布/订阅。

### 3.2 Day 0 一次性环境搭建（约 4 小时）

> **关于 Isaac Sim / Isaac Lab 版本搭配（2026-06）**
> - 你已下载 Isaac Sim **5.1 / 6.0**，建议**固定使用 Isaac Sim 6.0**（最新，含 NuRec / Gaussian Splatting）
> - **Isaac Sim ≠ Isaac Lab**：前者是底层仿真器，后者是上层 RL/IL 训练框架。两者**都要装**
> - Isaac Sim 6.0 → 配 **Isaac Lab v3.0+ / main**
> - 切忌混用版本：固定一个组合，别两个 Isaac Sim 来回切（USD schema 不兼容）

```bash
# 1. CUDA 12.1+ + cuDNN（系统级）
# 2. miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 3. 主环境（Week 1-3 用，纯 PyTorch / Gym）
conda create -n embai python=3.10 -y
conda activate embai
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install gymnasium[all] stable-baselines3 wandb hydra-core matplotlib einops tqdm
pip install lerobot diffusers transformers accelerate pinocchio

# 4. Isaac Sim 6.0 + Isaac Lab 环境（Week 2 起用，独立 conda 避免冲突）
conda create -n isaaclab python=3.10 -y
conda activate isaaclab

# 4a. 装 Isaac Sim 6.0（pip 版，最方便；如已用 launcher 装好可跳过）
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install isaacsim[all,extscache]==6.0.0 --extra-index-url https://pypi.nvidia.com

# 4b. 装 Isaac Lab（推荐 git clone 主分支，最新任务集合最全）
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install              # 自动检测当前 conda 中的 Isaac Sim

# 4c. 验证
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Cube-Franka-v0 --headless --num_envs 4096

# 5. 注册账号
# - wandb.ai （实验跟踪，必装）
# - HuggingFace （下载模型/数据集）
# - GitHub （存项目）
```

### 3.3 工程化工具（强烈建议第一周就上手）

| 工具 | 用途 | 优先级 |
|------|------|-------|
| `wandb` | 实验跟踪 / loss 曲线 / video log | P0 |
| `hydra-core` | 配置管理 | P0 |
| `tmux` | 长训练保活 | P0 |
| `pre-commit` + `ruff` | 代码风格 | P1 |
| `pytest` | 测试 | P1 |
| `nvitop` | GPU 监控 | P1 |

---

## 四、时间投入与节奏

| 时段 | 时长 | 内容 |
|------|------|------|
| 早 7:00–8:30 | 1.5h | 论文精读（一天 1 篇，带公式推导） |
| 上 9:00–12:00 | 3h | 代码实现 / 跑实验 |
| 下 14:00–17:00 | 3h | 调参 / debug / 阅读源码 |
| 晚 19:30–21:00 | 1.5h | 视频课 / 社区跟进 / 写笔记 |
| 周末 | — | 写 Weekly Recap，整理 GitHub |

**每周总投入约 60 小时 × 8 = 480 小时**。

---

## 五、阶段性里程碑

```
Week 1  ━━━━━━━━━━  PyTorch + RL 基本功打通
Week 2  ━━━━━━━━━━  会用主流仿真器，能跑 BC/ACT
Week 3  ━━━━━━━━━━  能从零写 Diffusion Policy
Week 4  ━━━━━━━━━━  动力学 + 7-DoF RL + 双臂 + 移动操作
Week 5  ━━━━━━━━━━  会微调 VLM
Week 6  ━━━━━━━━━━  能读懂并改动 SOTA VLA 代码
Week 7  ━━━━━━━━━━  有一个完整的轮式双臂端到端项目
Week 8  ━━━━━━━━━━  能进面试 / 投简历
```

---

## 六、使用方法

1. 每天早上打开当天对应的 `WeekXX.md`，按 **Day X** 标题找到任务。
2. 完成 **Daily Tasks** 后，对照 **Tuning Checklist** 检查实验质量。
3. 遇到坑先看 **Common Pitfalls**，再 Google / 问 GPT。
4. 每周末打开 [Milestones.md](./Milestones.md) 打勾。
5. 所有代码 push 到一个 GitHub repo（建议命名 `embodied-ai-bootcamp-8w`）。

---

## 七、最后三条嘱咐

1. **不要 Day 1 就装 Isaac Lab**。先把 PyTorch / Gym 摸熟，否则装 Isaac 会让你怀疑人生。
2. **每个实验都开 wandb**。不开 wandb 的实验等于没做，因为 3 天后你就忘了超参。
3. **每周末写一篇 500 字 Weekly Recap**，发到知乎或个人博客。8 篇连起来就是你的求职背书。

祝你 8 周后能在简历上写：「独立复现 Diffusion Policy / OpenVLA，并在 X 任务上达到 Y 成功率」。

开始吧 → [Week01.md](./Week01.md)
