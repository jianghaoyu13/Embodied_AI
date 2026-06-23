# 资源大全 · 8 周用得上的全部链接

> 这份是「随用随查」，不要试图一次读完。建议用浏览器收藏夹按周分组。

---

## 一、必读论文 Top 30（按主题）

### A. 模仿学习 / 操作 (10)
| # | 论文 | 年份 | 关键词 |
|---|------|------|-------|
| 1 | [ACT — Learning Fine-Grained Bimanual Manipulation](https://arxiv.org/abs/2304.13705) | 2023 | Action Chunking, ALOHA |
| 2 | [Diffusion Policy](https://arxiv.org/abs/2303.04137) | 2023 | DDPM, multimodal |
| 3 | [DP3 — 3D Diffusion Policy](https://arxiv.org/abs/2403.03954) | 2024 | sparse 3D |
| 4 | [3D Diffuser Actor](https://arxiv.org/abs/2402.10885) | 2024 | 3D + diffusion |
| 5 | [Implicit Behavior Cloning](https://arxiv.org/abs/2109.00137) | 2021 | EBM |
| 6 | [Diffuser](https://arxiv.org/abs/2205.09991) | 2022 | trajectory diffusion |
| 7 | [Consistency Policy](https://arxiv.org/abs/2405.07503) | 2024 | distillation |
| 8 | [BC-Z](https://arxiv.org/abs/2202.02005) | 2022 | scaling BC |
| 9 | [Equibot / EquiAct](https://arxiv.org/abs/2407.01479) | 2024 | SE(3) equivariance |
| 10 | [HumanPlus](https://arxiv.org/abs/2406.10454) | 2024 | shadowing |

### B. VLA (10)
| # | 论文 | 年份 |
|---|------|------|
| 11 | [RT-1](https://arxiv.org/abs/2212.06817) | 2022 |
| 12 | [RT-2](https://arxiv.org/abs/2307.15818) | 2023 |
| 13 | [Open X-Embodiment / RT-X](https://arxiv.org/abs/2310.08864) | 2023 |
| 14 | [OpenVLA](https://arxiv.org/abs/2406.09246) | 2024 |
| 15 | [π0: Vision-Language-Action Flow Model](https://arxiv.org/abs/2410.24164) | 2024 |
| 16 | [π0.5](https://www.physicalintelligence.company/blog/pi05) | 2025 |
| 17 | [GR00T N1](https://research.nvidia.com/labs/gear/gr00t-n1) | 2025 |
| 18 | [RDT-1B](https://arxiv.org/abs/2410.07864) | 2024 |
| 19 | [CogACT](https://arxiv.org/abs/2411.19650) | 2024 |
| 20 | [TinyVLA](https://arxiv.org/abs/2409.12514) | 2024 |

### C. 强化学习 + Sim2Real + 控制 + 动力学 (8)
| # | 论文 | 年份 |
|---|------|------|
| 21 | [PPO](https://arxiv.org/abs/1707.06347) | 2017 |
| 22 | [Massively Parallel RL in Isaac Gym](https://arxiv.org/abs/2108.10470) | 2022 |
| 23 | [Operational Space Control: A Theoretical and Empirical Comparison](https://www.cs.cmu.edu/~cga/dynopt/readings/Nakanishi08a.pdf) | 2008 |
| 24 | [Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201) | 2018 |
| 25 | [DreamerV3](https://arxiv.org/abs/2301.04104) | 2023 |
| 26 | [TD-MPC2](https://arxiv.org/abs/2310.16828) | 2023 |
| 27 | [Contact-GraspNet](https://arxiv.org/abs/2103.14127) | 2021 |
| 28 | [AnyGrasp](https://arxiv.org/abs/2212.08333) | 2022 |

### D. 双臂 / 移动操作 (4)
| # | 论文 | 年份 |
|---|------|------|
| 29 | [Mobile ALOHA](https://arxiv.org/abs/2401.02117) | 2024 |
| 30 | [RoboCasa: Large-Scale Simulation for Generalist Robots](https://arxiv.org/abs/2406.02523) | 2024 |
| 31 | [Hi-Robot: Open-Ended Instruction Following with Hierarchical VLA](https://www.physicalintelligence.company/research/hirobot) | 2025 |
| 32 | [HumanPlus / OmniH2O / iDP3](https://arxiv.org/abs/2406.10454) | 2024 |

### E. 表征 / 大模型 (4)
| # | 论文 | 年份 |
|---|------|------|
| 33 | [DINOv2](https://arxiv.org/abs/2304.07193) | 2023 |
| 34 | [SigLIP](https://arxiv.org/abs/2303.15343) | 2023 |
| 35 | [LLaVA](https://arxiv.org/abs/2304.08485) | 2023 |
| 36 | [LoRA](https://arxiv.org/abs/2106.09685) | 2021 |

---

## 二、必 star 代码仓库

### 数据 / 训练框架
| 仓库 | 说明 |
|------|------|
| [`huggingface/lerobot`](https://github.com/huggingface/lerobot) | 数据 + IL 全家桶 |
| [`isaac-sim/IsaacLab`](https://github.com/isaac-sim/IsaacLab) | 主流仿真训练 |
| [`Genesis-Embodied-AI/Genesis`](https://github.com/Genesis-Embodied-AI/Genesis) | 新仿真器 |
| [`google-deepmind/mujoco`](https://github.com/google-deepmind/mujoco) | MuJoCo + MJX |

### 动力学 / 控制 / 抓取（你这周强化的部分）
| 仓库 | 说明 |
|------|------|
| [`stack-of-tasks/pinocchio`](https://github.com/stack-of-tasks/pinocchio) | 7-DoF 动力学 RNEA / CRBA / ABA |
| [`NVlabs/curobo`](https://github.com/NVlabs/curobo) | 你已熟练，规划 / IK / collision |
| [`elchun/contact_graspnet_pytorch`](https://github.com/elchun/contact_graspnet_pytorch) | Contact-GraspNet PyTorch 实现 |
| [`graspnet/graspnetAPI`](https://github.com/graspnet/graspnetAPI) | GraspNet-1Billion API |
| [`graspnet/anygrasp_sdk`](https://github.com/graspnet/anygrasp_sdk) | AnyGrasp |
| [`UM-ARM-Lab/pytorch_kinematics`](https://github.com/UM-ARM-Lab/pytorch_kinematics) | 可微 FK / Jacobian |

### 操作 / VLA
| 仓库 | 说明 |
|------|------|
| [`real-stanford/diffusion_policy`](https://github.com/real-stanford/diffusion_policy) | DP 官方实现 |
| [`YanjieZe/3D-Diffusion-Policy`](https://github.com/YanjieZe/3D-Diffusion-Policy) | DP3 |
| [`tonyzhaozh/act`](https://github.com/tonyzhaozh/act) | ACT — 双臂入门 |
| [`MarkFzp/mobile-aloha`](https://github.com/MarkFzp/mobile-aloha) | Mobile ALOHA — 移动双臂 |
| [`robocasa/robocasa`](https://github.com/robocasa/robocasa) | 厨房 mobile manipulation |
| [`openvla/openvla`](https://github.com/openvla/openvla) | OpenVLA |
| [`Physical-Intelligence/openpi`](https://github.com/Physical-Intelligence/openpi) | π0 开源版 |
| [`NVIDIA/Isaac-GR00T`](https://github.com/NVIDIA/Isaac-GR00T) | GR00T 推理 + 微调 |
| [`thu-ml/RoboticsDiffusionTransformer`](https://github.com/thu-ml/RoboticsDiffusionTransformer) | RDT-1B |

### Benchmarks（已剔除 locomotion-only）
| 仓库 | 说明 |
|------|------|
| [`haosulab/ManiSkill`](https://github.com/haosulab/ManiSkill) | 操作任务集合 |
| [`Lifelong-Robot-Learning/LIBERO`](https://github.com/Lifelong-Robot-Learning/LIBERO) | VLA 评测标准 |
| [`robocasa/robocasa`](https://github.com/robocasa/robocasa) | 厨房长程 mobile manipulation |
| [`ARISE-Initiative/robosuite`](https://github.com/ARISE-Initiative/robosuite) | 操作底座 |

### RL 基础 / 控制
| 仓库 | 说明 |
|------|------|
| [`DLR-RM/stable-baselines3`](https://github.com/DLR-RM/stable-baselines3) | RL 标准实现 |
| [`vwxyzjn/cleanrl`](https://github.com/vwxyzjn/cleanrl) | 单文件版 RL，超易读 |
| [`leggedrobotics/rsl_rl`](https://github.com/leggedrobotics/rsl_rl) | Isaac Lab 配套 PPO |
| [`nicklashansen/tdmpc2`](https://github.com/nicklashansen/tdmpc2) | TD-MPC2 |
| [`danijar/dreamerv3`](https://github.com/danijar/dreamerv3) | Dreamer V3 |
| [`loco-3d/crocoddyl`](https://github.com/loco-3d/crocoddyl) | 最优控制 (OC) |

### 表征 / 大模型
| 仓库 | 说明 |
|------|------|
| [`facebookresearch/dinov2`](https://github.com/facebookresearch/dinov2) | DINOv2 |
| [`mlfoundations/open_clip`](https://github.com/mlfoundations/open_clip) | CLIP / SigLIP |
| [`huggingface/peft`](https://github.com/huggingface/peft) | LoRA 全家桶 |
| [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention) | FlashAttention |

---

## 三、视频 / 课程 / 博客

### 系统课程
| 资源 | 风格 |
|------|------|
| [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/) | RL 入门最佳 |
| [Sergey Levine — CS285 Deep RL](https://rail.eecs.berkeley.edu/deeprlcourse/) | RL 高阶 |
| [Chelsea Finn — CS336 Deep Multi-Task and Meta Learning](https://cs330.stanford.edu/) | 元学习 / 多任务 |
| [李宏毅 — 机器学习 / RL](https://speech.ee.ntu.edu.tw/~hylee/index.php) | 中文，亲切 |
| [Modern Robotics (Kevin Lynch)](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) | 机器人学经典 |

### 顶会演讲
- CoRL 2024 / 2025 keynotes
- RSS 2024 / 2025 best papers
- ICRA 2024 / 2025 plenary
- NeurIPS Robotics workshops

### 必关注个人博客
| 作者 | 链接 |
|------|------|
| Lilian Weng | https://lilianweng.github.io/ |
| Sergey Levine | https://people.eecs.berkeley.edu/~svlevine/ |
| Chelsea Finn | https://ai.stanford.edu/~cbfinn/ |
| Pieter Abbeel | https://people.eecs.berkeley.edu/~pabbeel/ |
| Pieter Abbeel — Robot Brains podcast | https://www.therobotbrains.ai/ |

### 行业新闻 / 趋势
- The Humanoid Hub（Substack）
- The Batch (Andrew Ng)
- Import AI（Jack Clark）
- The Information / Stratechery（商业角度）

---

## 四、X / 知乎 / Discord 推荐关注

### X / Twitter
- @svlevine (Sergey Levine)
- @chelseabfinn
- @pabbeel
- @physical_int (Physical Intelligence)
- @1x_tech
- @figure_robot
- @drjimfan (NV GR00T lead)
- @jiajunwu_cs
- @yukez (Robocasa, NVIDIA)
- @cliangyu_

### 知乎话题
- 「具身智能」
- 「机器人学习」
- 「强化学习」
- 「机器人操作」

### Discord
- HuggingFace LeRobot
- Isaac Lab
- ManiSkill / Habitat
- LeRobot Hackathon

---

## 五、数据集速查

### 模仿学习数据
| 数据集 | 规模 | 特点 |
|--------|------|------|
| Open X-Embodiment | 1M+ episodes, 22 robots | 跨机数据集 |
| DROID | 76k 轨迹, 564 场景 | 多场景 |
| BridgeData V2 | 60k 轨迹 | 厨房 |
| RH20T | 110k 任务 | 多机 |
| ALOHA / Mobile-ALOHA | 几千轨迹 | 双臂 |
| AgiBot World | 1M+ 轨迹 | 智元开源 |

### 仿真 benchmark
| Benchmark | 任务数 | 用途 |
|-----------|------|------|
| LIBERO | 130 任务 | VLA 评测标准 |
| ManiSkill3 | 30+ | 操作 |
| Meta-World | 50 | RL 基准 |
| RoboCasa | 25 厨房任务 | 长程操作 |
| RoboMimic | 7 任务 | IL 基准 |

---

## 六、硬件 / 算力 资源

### 云算力
| 服务 | 价格 | 备注 |
|------|------|------|
| AutoDL | ~¥2-4/h（4090） | 国内首选 |
| 阿里云 PAI | 中等 | 适合企业 |
| Lambda Labs | $1.5/h（4090） | 海外 |
| RunPod | $0.5-1/h | 海外便宜 |
| Vast.ai | $0.3-1/h | 海外最便宜 |

### 真机入门（量级递增）
| 平台 | 价格 | 用途 |
|------|------|------|
| LeRobot SO-100 / SO-101 | ~$200 | 入门双臂 |
| Trossen ALOHA | ~$25k | 双臂操作 |
| Unitree Go2 / Z1 | ~$10-30k | 四足 + 机械臂 |
| Galaxea / Agibot Lite | ~$50k+ | 双臂 + 移动 |
| Unitree H1 / G1 | $90k+ | 人形入门 |

> 没真机就用 LeRobot SO-100 + Mac/PC 录数据，足够前期项目使用。

---

## 七、面试参考

### 面试库 / 题库
- [Glassdoor](https://www.glassdoor.com/) 看公司面经
- 一亩三分地（北美求职）
- 知乎「机器人面试」相关回答
- LeetCode 顶会公司题库（PI / NV / Google 等）

### 写代码题（手撕题）参考
- 写 PPO 主循环（Day 6 Week1 模板）
- 写 DDPM forward + sample
- 写 Multi-head Attention（含 mask）
- 写 GAE
- 写 DP inference（receding horizon）
- 写 LeRobot 数据加载 + 切片
- 写 SE(3) batched ops（Day 3 Week1）

---

## 八、英文求职关键词（Bound to Show in JD）

`Vision-Language-Action`, `VLA`, `imitation learning`, `policy learning`, `manipulation`, `dexterous`, `whole-body`, `loco-manipulation`, `humanoid`, `quadruped`, `Sim-to-Real`, `domain randomization`, `behavior cloning`, `diffusion policy`, `action chunking`, `flow matching`, `RL`, `MuJoCo`, `Isaac Lab`, `Isaac Gym`, `ROS2`, `LeRobot`, `Open X-Embodiment`, `LIBERO`, `RoboCasa`, `digital twin`, `teleoperation`, `data flywheel`, `MPC`, `whole-body controller`, `MoCap retargeting`.

---

## 九、组合式学习路径（按你的目标 customize）

### 路径 A — 「我要 6 个月内进 PI / GR00T / 银河通用团队」
- Week 1-3 不变
- Week 4 加强 RoboCasa mobile manipulation + 双臂
- Week 5-6 全力 VLA
- Week 7 选 🅐 操作路线或 🅒 VLA 路线
- 8 周后再用 4-8 周做一个**真机 demo**

### 路径 B — 「我想做轮式双臂人形操作方向」（✅ 你的定位）
- Week 1-2 不变
- Week 3 不变，但 DP 实验优先双臂 / 多任务
- **Week 4 重点**：动力学 + Franka RL + 双臂 + RoboCasa 全力推
- Week 5-6 VLM/VLA 部分关注「操作方向」的下游微调
- Week 7 选 🅐 操作路线或 🅑 双臂路线

### 路径 C — 「我想做学术 PhD」
- 8 周路线全部完成
- 在 Week 8 之外加 4-8 周写一篇 workshop paper
- 投 CoRL / RSS workshop 或 ICRA late breaking

---

## 十、最后的话

资源会过期，**框架不会**。
- 每年 SOTA 都在变，但「数据 / 仿真 / 表征 / 决策 / 控制」五大模块的位置永远在那。
- 这份 Resources.md 也欢迎你 8 周后回来更新。

→ [../README.md](../README.md) | [Milestones.md](./Milestones.md)
