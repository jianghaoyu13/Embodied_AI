# 8 周里程碑自检表

> 每周末花 30 分钟，对照本表打勾。
> 70% 以上 → 进入下一周；50-70% → 周末补齐弱项；< 50% → 多用 1-3 天重攻关键模块。

---

## ✅ Week 1 — PyTorch + RL Foundations

### 必达
- [ ] 手写 MNIST 训练循环（无 Lightning），收敛到 acc > 97%
- [ ] 手写 Mini-GPT，在 Tiny Shakespeare 训练 1 epoch 能采样
- [ ] Batched SE(3) 操作通过 `log(exp(x)) ≈ x` 单测
- [ ] 实现 Value Iteration / Q-Learning，在 FrozenLake 收敛
- [ ] 实现 PPO + GAE，CartPole 100% 成功，HalfCheetah reward > 2500
- [ ] 5 个实验全部上 wandb，可分享 link

### 加分
- [ ] PPO 在 LunarLander 也跑通
- [ ] FlashAttn 替换标准 attention，测得加速
- [ ] 写完一篇 ≥500 字 weekly recap

### 自检问答（口头答得出来）
- 解释 PPO 损失函数每一项
- 解释 GAE 和 TD-error 关系
- 解释为何 attention 除以 √d_k

---

## ✅ Week 2 — Isaac Lab + 模仿学习

### 必达
- [ ] Isaac Lab 安装成功，跑通 Lift-Cube-Franka
- [ ] 跑通 MuJoCo / MJX 一个 demo
- [ ] 加载 LeRobot Push-T 数据集，看懂结构
- [ ] BC baseline 在 Push-T 训练完成（成功率 ~30-45%）
- [ ] ACT 在 ALOHA Cube Transfer 复现到 ≥ 80%
- [ ] 至少 3 组 ACT 消融

### 加分
- [ ] 自录 30 条 Push-T teleop 数据
- [ ] 实现 temporal ensemble inference

### 自检
- 解释 ACT 的 chunk + CVAE 设计
- BC vs ACT 何时选哪个？

---

## ✅ Week 3 — Diffusion Policy + 表征

### 必达
- [ ] 1D toy DDPM 复现，DDIM 也跑通
- [ ] DP 论文笔记 5 页 + 架构图
- [ ] DP 在 Push-T (state) 成功率 ≥ 85%
- [ ] DP 在 Push-T (image) 成功率 ≥ 75%
- [ ] 至少 5 组消融（horizon / backbone / scheduler 等）
- [ ] DP3 在某 3D 任务跑通
- [ ] Flow Matching 1D toy 复现，对比 DDPM

### 加分
- [ ] DP + Flow Matching 接通，推理 < 8 步
- [ ] 一篇 ≥2000 字博客草稿

### 自检
- 为什么 DP 在 multimodal 上比 BC 好？
- UNet1D 的 1D 是哪个维度？
- EMA / DDIM / receding horizon 各自作用？

---

## ✅ Week 4 — 机器人动力学 + 7-DoF / 双臂 / 移动操作

### 必达
- [ ] 能用 Pinocchio 算 Franka 7-DoF 的 M, C, G，并解释每项含义
- [ ] 手算 2-DoF planar arm 动力学，与 sympy 一致
- [ ] 在 Isaac Lab 跑通 `Isaac-Lift-Cube-Franka-v0` 的 PPO 收敛
- [ ] 至少 2 种 action space 对比（joint pos / OSC delta EE）
- [ ] Contact-GraspNet + cuRobo 串通一个端到端 pick-and-place
- [ ] 双臂任务（ALOHA sim_transfer_cube / sim_insertion）跑通 ACT，成功率 ≥ 80%
- [ ] RoboCasa 至少 1 个 mobile manipulation 任务跑通
- [ ] 实现一次 cuRobo + RL Residual 训练并对比纯 RL

### 加分
- [ ] OSC + impedance 控制 demo
- [ ] 自定义一个轮式双臂的 Isaac Lab 简化平台

### 自检
- 解释 OSC 控制律每一项物理含义
- 双臂任务 chunk size 为什么需要更大？
- 为何 manipulation 倾向 OSC delta EE 而非 joint pos？
- GraspNet + cuRobo 与 端到端 RL 各自适用什么场景？

---

## ✅ Week 5 — VLM + LoRA + Action Tokenizer

### 必达
- [ ] 在 Day2 Mini-GPT 上加 RoPE + GQA + FlashAttn
- [ ] CLIP / SigLIP / DINOv2 三种 feature 在 BC 下游对比
- [ ] 自建 mini-LLaVA，能在小数据回答简单问题
- [ ] 用 LoRA / QLoRA 微调 Qwen2.5-VL-7B
- [ ] 实现 3 种 action tokenizer（Bin / VQ / FAST）
- [ ] OpenVLA 推理 pipeline 跑通

### 加分
- [ ] OpenVLA LoRA 微调到自录数据，看到改善

### 自检
- 为什么 OpenVLA 用 DINOv2 + SigLIP 双 backbone？
- LoRA 怎么和 4-bit 量化结合？
- bin / VQ / FAST 各自适用场景？

---

## ✅ Week 6 — VLA SOTA 深读

### 必达
- [ ] RT-1 → π0 演化时间轴整理完成
- [ ] π0 / π0.5 论文精读 + 推理跑通
- [ ] GR00T N1.5 推理跑通
- [ ] RDT-1B 推理跑通
- [ ] LIBERO 上微调 OpenVLA，spatial / object ≥ 80%
- [ ] 自讲 10 分钟：π0 vs GR00T vs RDT 区别

### 加分
- [ ] LIBERO-10 ≥ 50%
- [ ] 整理「VLA 全景图」并发布

### 自检（10 分钟全答出来）
- 为什么 discrete → continuous action？
- Why Flow Matching > DDPM in robotics？
- System1/System2 解决什么问题？
- 给你 10× H100，怎么训自己 VLA？

---

## ✅ Week 7 — 端到端项目

### 必达
- [ ] 选定一个项目方向（A / B / C 之一）
- [ ] PROJECT.md 写清问题、metric、baseline、stretch
- [ ] 主实验跑完，达到目标数字
- [ ] 至少 5 组消融
- [ ] 1 个 non-trivial 创新点
- [ ] 主任务视频 + GIF
- [ ] README 一键复现命令能跑
- [ ] checkpoint 上 HuggingFace
- [ ] `pytest` 至少 5 个核心测试

### 加分
- [ ] GitHub repo 拿到 ≥ 10 stars（来自朋友 / X 推广）

### 自检
- 你的方法 work 是因为 X，而不是 Y（一句话）
- 失败案例是什么？
- 再做一次会怎么改？

---

## ✅ Week 8 — 论文输出 + 求职

### 必达
- [ ] 5000 字技术博客发布到 ≥ 2 个渠道
- [ ] 简历改成 1 页（中英文各一份）
- [ ] GitHub Profile README + 3-5 pin repos
- [ ] 5 篇 SOTA 论文复盘笔记
- [ ] 投递 ≥ 10 家
- [ ] 完成 ≥ 2 场模拟面试

### 加分
- [ ] 拿到 ≥ 1 场真实面试邀请
- [ ] 在 X / LinkedIn 上发了项目串
- [ ] 收到 ≥ 1 次内推回信

### 自检
- 30 秒自我介绍流畅
- 项目讲 30 分钟不卡壳
- 反向提问准备 ≥ 5 个

---

## 全 8 周毕业认证（最后一道关）

走完 8 周后，对照下面这张「最终认证表」自评。
**8 项中至少完成 6 项**，才算完整毕业。

- [ ] **手撕能力**：从零写过 PPO / DP / Attention，10 分钟内能默写关键部分
- [ ] **仿真能力**：在 Isaac Lab 训过至少 1 个 RL 策略到 SOTA
- [ ] **IL 能力**：复现过 ≥ 2 个 SOTA IL 算法（DP / ACT / OpenVLA 中任选）
- [ ] **VLA 能力**：在公开 benchmark 微调过 ≥ 1 个 VLA 模型
- [ ] **工程能力**：项目用 wandb / hydra / docker / pytest 全套
- [ ] **沟通能力**：30 分钟讲清自己的项目，含 motivation / method / experiment / limitation
- [ ] **见识广度**：能 30 分钟讲清 π0 / GR00T / RDT 架构差异
- [ ] **市场能见度**：1 篇技术博客 ≥ 500 阅读 OR GitHub repo ≥ 20 stars OR 内推回信 ≥ 3 家

---

## 自评打分卡

| 周 | 必达完成 | 加分完成 | 状态 | 备注 |
|----|---------|---------|------|------|
| W1 | ___ / 6 | ___ / 3 | ⬜ | |
| W2 | ___ / 6 | ___ / 2 | ⬜ | |
| W3 | ___ / 7 | ___ / 2 | ⬜ | |
| W4 | ___ / 6 | ___ / 2 | ⬜ | |
| W5 | ___ / 6 | ___ / 1 | ⬜ | |
| W6 | ___ / 6 | ___ / 2 | ⬜ | |
| W7 | ___ / 9 | ___ / 1 | ⬜ | |
| W8 | ___ / 6 | ___ / 3 | ⬜ | |

> 状态符号约定：⬜ 未开始 / 🟡 进行中 / 🟢 完成 / 🔴 落后

---

## 红线：以下任一情况立即调整计划

- 单周必达 < 50% → 暂停下周新内容，先补当前周
- 连续 3 天没有 wandb 记录 → 训练 pipeline 一定有问题，停下来 debug
- 项目方向已选但 Week 7 Day 3 仍跑不通 baseline → 立刻换更小 scope 的方向
- Week 8 Day 50 没博客草稿 → 砍掉「面试题准备」时间，先发布博客

---

## 8 周总打卡

完整走完后，回到这里，给自己一个签名 + 日期：

```
我，____________，
于 2026 年 ___ 月 ___ 日，
完成了 8 周具身智能算法 bootcamp，
从零基础走到了能独立复现 SOTA、面试头部团队的水平。

下一阶段目标：____________________________________________

——————————————————————
（这一行写下你最自豪的项目链接）
```

---

→ [../README.md](../README.md) | [Resources.md](./Resources.md)
