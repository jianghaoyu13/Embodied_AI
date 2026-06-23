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

> **目录约定（已按你最新结构调整）**
> - `week_plan/` — 8 周宏观计划（每周一个 `WeekXX.md`）+ Resources / Milestones
> - `day_plan/` — 当天 hour-by-hour 执行计划（每天一个 `DayXX.md`）
> - 跨文件链接已全部更新为相对路径

| 文件 | 主题 | 阶段产出 |
|------|------|---------|
| [week_plan/Week01.md](./week_plan/Week01.md) | PyTorch 实战 + SE(3) + RL 入门 | 手写 PPO，CartPole/HalfCheetah 收敛 |
| [week_plan/Week02.md](./week_plan/Week02.md) | Isaac Lab + LeRobot + BC/ACT | ALOHA 双臂 Cube Transfer 复现 |
| [week_plan/Week03.md](./week_plan/Week03.md) | Diffusion Policy + 表征学习 | Push-T 任务 SOTA 复现 |
| [week_plan/Week04.md](./week_plan/Week04.md) | **动力学补强 + 7-DoF RL + 双臂 + 抓取 + 移动操作** | Franka RL + 双臂 + RoboCasa |
| [week_plan/Week05.md](./week_plan/Week05.md) | VLM 基础 + LLaVA 微调 | 自录数据微调 Qwen2.5-VL |
| [week_plan/Week06.md](./week_plan/Week06.md) | VLA 主流模型深读（π0 / GR00T / RDT） | 在 LIBERO 上微调 OpenVLA |
| [week_plan/Week07.md](./week_plan/Week07.md) | **端到端项目（轮式双臂 / 7-DoF）** | 1 个亮眼 GitHub 项目 |
| [week_plan/Week08.md](./week_plan/Week08.md) | 论文输出 + 求职冲刺 | 简历、博客、面试准备 |
| [week_plan/Resources.md](./week_plan/Resources.md) | 完整资源清单（论文 / 仓库 / 视频） | 收藏夹 |
| [week_plan/Milestones.md](./week_plan/Milestones.md) | 8 周里程碑自检表 | 每周打勾 |
| [day_plan/](./day_plan/) | **逐日精细化执行计划**（hour-by-hour） | 每天直接打开当天的 `DayXX.md` |

---

## 三、硬件 / 环境前置（Day 0）

### 3.1 你的硬件 / 系统实况

| 配置 | 你的值 | 备注 |
|------|--------|------|
| GPU | RTX 4090 (24GB) | |
| Driver | **590.48.01** | |
| CUDA (Driver API) | **13.1** | PyTorch 装 `cu130` wheels(PyTorch 2.11+ 已稳定) |
| OS | Ubuntu 24.04 | |
| ROS2 | **Jazzy** (LTS, 2024-2029) | 已装,可选用,不强制 |
| Isaac Sim 5.1 | **本地原生**（已装） | 备用 GUI / 调试 / 轻量场景 |
| Isaac Sim 6.0 | **Docker 容器**（已装） | **主力训练**(headless RL/IL) |
| Isaac Lab | 装在 Isaac Sim 6.0 容器内 | v3.0+ / main |

> **关于 CUDA 13.1**:
> Driver CUDA Version 13.1 表示驱动**最高支持** CUDA 13.1 Runtime。PyTorch 2.11 起 `cu130` 已是 PyPI stable wheel(2026-03 起),Driver 590 + cu130 + torch 2.12 实测稳定,cuRobo 也已验证可用。
>
> **关于 Isaac Sim 双装策略(已为你校准)**:
> - **5.1 本地** —— 适合「快速 GUI 看场景 / 写 USD / 单步调试 / 不需要训练的任务」。本地启动延迟低,文件系统直通方便。
> - **6.0 Docker** —— 适合「headless RL/IL 训练 / Isaac Lab 任务集合 / 多人共用机器时隔离环境」。版本固化,部署到云容易。
> - **Isaac Lab 装在 6.0 容器内**,不混到 5.1 本地。USD schema 不兼容,**别两边来回切**。
> - Week 2 Day 8 起的所有训练命令默认在 **6.0 容器**里跑;本地 conda 只跑 Pinocchio / cuRobo / 纯 PyTorch。

### 3.2 Day 0 一次性环境搭建(约 4 小时)

#### Step 1 — 本地 conda 主环境(Week 1-3 + 动力学 / cuRobo / Pinocchio 用)

```bash
# 1. miniconda(没装的话)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 2. embai 主环境(纯 PyTorch / Gym / Pinocchio / cuRobo)
# Python 3.11 — 与 Isaac Sim 6.0 容器内 Python 对齐,host↔container 共享 pickle/dataset/ckpt 不踩版本坑
conda create -n embai python=3.11 -y
conda activate embai
pip install --upgrade pip
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install gymnasium[all] stable-baselines3 wandb hydra-core matplotlib einops tqdm
pip install lerobot diffusers transformers accelerate
pip install pin                                    # Pinocchio,Week 4 动力学用
# cuRobo 你已熟练,按官方 README 装
```

#### Step 2 — Isaac Sim 5.1 本地启动验证(已装)

```bash
# 找到 5.1 安装位置
ls ~/.local/share/ov/pkg/ | grep "isaac-sim-5.1" || \
  find ~ -maxdepth 4 -name "isaac-sim.sh" 2>/dev/null

# 启动一次 GUI(确认能开)
cd ~/isaacsim    # 路径以你实际为准
./isaac-sim.sh
```

5.1 本地的用途:day_plan 里凡涉及「打开 USD 看一眼」「拖一个机器人到场景」这类轻量交互,都走 5.1。

#### Step 3 — Isaac Sim 6.0 Docker 启动验证(已装)

> 假设你已经通过 NGC 拉好镜像。常用 tag:
> `nvcr.io/nvidia/isaac-sim:6.0.0` 或你自定义的 tag(用 `docker images | grep isaac` 确认)。

**3a. 准备宿主机挂载目录**

```bash
mkdir -p ~/isaac_workspace/{IsaacLab,projects,nv_cache/main,nv_cache/computecache,nv_logs,nv_config,nv_data,nv_pkg,nv_hub,docs}

# Isaac Sim 6.0 容器以非 root 用户 UID 1234 运行(镜像内置 `ubuntu` 用户,home=/isaac-sim)
# 所以宿主机的挂载点必须能被 1234:1234 读写:
sudo chown -R 1234:1234\                         
    ~/isaac_workspace/nv_cache \
    ~/isaac_workspace/nv_logs \
    ~/isaac_workspace/nv_data \
    ~/isaac_workspace/nv_config \
    ~/isaac_workspace/nv_pkg \
    ~/isaac_workspace/nv_hub
# 如果你本地 UID 已经是 1234,可跳过(用 `id -u` 查一下)

# IsaacLab          — git clone 的 Isaac Lab 源码,容器内可写,宿主机也能编辑
# projects          — 你的训练代码 / Embodied-AI git repo 软链或 clone
# nv_cache/main     — Omniverse / Kit 通用缓存(/isaac-sim/.cache)
# nv_cache/computecache — NVIDIA shader 编译缓存(/isaac-sim/.nv/ComputeCache),首次启动慢的元凶,单独挂
# nv_logs           — Omniverse 日志(/isaac-sim/.nvidia-omniverse/logs)
# nv_config         — Omniverse 用户配置 / UI 设置(/isaac-sim/.nvidia-omniverse/config)
# nv_data           — OV 数据(/isaac-sim/.local/share/ov/data)
# nv_pkg            — OV 扩展包(/isaac-sim/.local/share/ov/pkg),不挂会丢扩展
# nv_hub            — Omniverse Hub 缓存(/var/cache/hub)
# docs              — 容器内 Documents 目录(/isaac-sim/Documents)
```

**3b. 一次性启动脚本**(写到 `~/isaac_workspace/run_isaac6.sh`)

```bash
#!/usr/bin/env bash
# ~/isaac_workspace/run_isaac6.sh
set -e

IMG=${IMG:-nvcr.io/nvidia/isaac-sim:6.0.0}
NAME=${NAME:-isaac6}

xhost +local: >/dev/null 2>&1 || true     # 允许容器内非 root 用户访问宿主机 X server

docker run --name "$NAME" --rm -it \
  --entrypoint bash \
  --gpus all \
  --runtime=nvidia \
  --network=host \
  -u 1234:1234 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e DISPLAY="$DISPLAY" \
  -v "$HOME/.Xauthority:/isaac-sim/.Xauthority:rw" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$HOME/isaac_workspace/IsaacLab:/workspace/IsaacLab:rw" \
  -v "$HOME/isaac_workspace/projects:/workspace/projects:rw" \
  -v "$HOME/isaac_workspace/nv_cache/main:/isaac-sim/.cache:rw" \
  -v "$HOME/isaac_workspace/nv_cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
  -v "$HOME/isaac_workspace/nv_logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
  -v "$HOME/isaac_workspace/nv_config:/isaac-sim/.nvidia-omniverse/config:rw" \
  -v "$HOME/isaac_workspace/nv_data:/isaac-sim/.local/share/ov/data:rw" \
  -v "$HOME/isaac_workspace/nv_pkg:/isaac-sim/.local/share/ov/pkg:rw" \
  -v "$HOME/isaac_workspace/nv_hub:/var/cache/hub:rw" \
  -v "$HOME/isaac_workspace/docs:/isaac-sim/Documents:rw" \
  "$IMG"
```

> **要点速记**(为什么这个脚本对 6.0 比"老版本 root 容器"稳):
> - `-u 1234:1234` —— Isaac Sim 6.0 镜像内置 `ubuntu` 用户(UID 1234, home `/isaac-sim`),非 root 跑才符合官方设计,避免和 OV Kit sandboxing 冲突
> - 所有挂载点用 `/isaac-sim/...` —— 6.0 的非 root home 在这,挂到 `/root/...` 不生效,会每次重下 shader
> - `nv_cache/computecache` 单独挂 —— shader 编译缓存(几 GB),分开挂避免被通用 `.cache` 清理误伤
> - `nv_config` / `nv_pkg` / `nv_hub` —— 6.0 把扩展、用户配置、Hub 缓存拆三处,缺一个就丢一种持久化状态
> - `.Xauthority` 显式挂载 + `xhost +local:` —— 非 root 用户访问宿主机 X server 必需

```bash
chmod +x ~/isaac_workspace/run_isaac6.sh
~/isaac_workspace/run_isaac6.sh
```

进容器后 sanity check:

```bash
# 在容器内
nvidia-smi                       # 期望看到 4090
which isaacsim || ls /isaac-sim  # 镜像内 Isaac Sim 路径,自查
./python.sh -c "
from isaacsim import SimulationApp
sim = SimulationApp({'headless': True})
print('Isaac Sim 6.0 OK')
sim.close()
"
```

**3c. Isaac Lab 装到 6.0 容器内**

```bash
# 在宿主机先 clone(挂载点)
cd ~/isaac_workspace/IsaacLab
git clone https://github.com/isaac-sim/IsaacLab.git . 

# 在容器内安装(Isaac Lab 会调用容器自带的 isaacsim)
cd /workspace/IsaacLab
./isaaclab.sh --install

# 验证
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Cube-Franka-v0 --headless --num_envs 4096
```

> **使用约定**(写在墙上):
> - 训练命令以 `[isaac6]$` 开头 → 在 6.0 容器内跑
> - 命令以 `[embai]$` 开头 → 本地 conda `embai` 跑
> - 命令以 `[5.1]$` 开头 → 本地 Isaac Sim 5.1 跑(GUI 调试)

#### Step 4 — 注册账号

- wandb.ai (实验跟踪,必装)
- HuggingFace (下载模型 / 数据集)
- GitHub (存项目)

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

1. 每天早上打开 `day_plan/DayXX.md`(精细 hour-by-hour),配合 `week_plan/WeekXX.md`(宏观任务)使用。
2. 完成 **Daily Tasks** 后,对照 **Tuning Checklist** 检查实验质量。
3. 遇到坑先看 **Common Pitfalls**,再 Google / 问 GPT。
4. 每周末打开 [week_plan/Milestones.md](./week_plan/Milestones.md) 打勾。
5. 所有代码 push 到 `git@github.com:jianghaoyu13/Embodied_AI.git`(本地工作目录建议 `~/embodied-ai-bootcamp-8w`,与本仓库的学习计划文档分开)。

---

## 七、最后三条嘱咐

1. **不要 Day 1 就装 Isaac Lab**。先把 PyTorch / Gym 摸熟，否则装 Isaac 会让你怀疑人生。
2. **每个实验都开 wandb**。不开 wandb 的实验等于没做，因为 3 天后你就忘了超参。
3. **每周末写一篇 500 字 Weekly Recap**，发到知乎或个人博客。8 篇连起来就是你的求职背书。

祝你 8 周后能在简历上写：「独立复现 Diffusion Policy / OpenVLA，并在 X 任务上达到 Y 成功率」。

开始吧 → [week_plan/Week01.md](./week_plan/Week01.md) / [day_plan/Day01.md](./day_plan/Day01.md)
