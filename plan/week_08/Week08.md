# Week 8 — 论文输出 + 求职冲刺

> **本周定位**：把前 7 周的积累 **变现**。这周不写新代码，专心打磨产出物 + 投递。
> 目标：8 周后你能拿到至少 3-5 家头部团队（PI、银河、智元、星动、Figure、宇树、1X、宇宙等等）的面试机会。

---

## 本周目标

完成本周后，你应该能：
- [ ] 拿出一篇高质量技术博客（5000 字，附图、视频、数据）
- [ ] 拿出一份打磨过的简历 + GitHub 主页
- [ ] 完成至少 5 家公司投递
- [ ] 顺利通过 1-2 轮模拟面试（手撕 + 系统设计 + 论文讨论）

---

## 本周可交付物

```
week8-shipping/
├── day50_blog/         # 技术博客 v1
├── day51_blog_polish/  # 博客 v2 + 校对
├── day52_resume_gh/    # 简历 + GitHub 主页
├── day53_5papers/      # 复盘 5 篇 SOTA
├── day54_apply/        # 投递记录
├── day55_interview/    # 面试题准备
└── day56_mock/         # 模拟面试
```

---

## Day 50 — 技术博客 v1

### 早晨
- 读 3 篇你欣赏的机器人技术博客，分析结构（推荐：The Gradient、Lilian Weng 博客、Distill）

### Daily Tasks
1. 写出 5000 字博客 v1 草稿
2. 不追求完美，**先完成再优化**

### 博客结构模板（强烈推荐）

```markdown
# <Catchy Title> — <One-liner Subtitle>

> TL;DR: <一段不超过 80 字的总结>

## 1. Why This Matters (300 字)
- 行业背景
- 现有方法的瓶颈
- 你想解决什么

## 2. Background (800 字)
- 用通俗语言讲清楚相关 SOTA
- 避开纯数学，用图说话

## 3. Method (1500 字)
- 你的整体 pipeline 图
- 关键设计决策（每个都有 "why not X?"）
- 实现细节（让人能复现）

## 4. Experiments (1500 字)
- 主结果表
- 至少 2 个消融
- 失败案例分析（重要！显得你真懂）
- 视频 / GIF

## 5. Limitations & Future (400 字)
- 诚实列出 3-5 个 limitation
- 你认为下一步该做什么

## 6. Reproduce (200 字)
- GitHub 链接
- HuggingFace checkpoint 链接
- 一键复现命令

## 7. Acknowledgments / References (300 字)
```

### 写作技巧
- **第一段决定 80% 的读者去留**：直接放结果数字 / GIF
- **每个章节加 1-2 张图**：纯文字看一段就累
- **用第一人称「我们」**（即使你是个人项目，行话）
- **数字精确**：`reward 收敛到 ~3000` 不如 `reward = 3120 ± 84 across 5 seeds`
- **限制术语堆砌**：每用一个新术语，括号给一句话解释

### 渠道选择
| 平台 | 优势 | 劣势 |
|------|------|------|
| 知乎 | 中文圈、流量大 | SEO 一般、同质化 |
| 个人博客（Hugo/Hexo） | 自由、长期资产 | 没流量 |
| Medium / Substack | 国际曝光 | 中文用户少 |
| HuggingFace Blog | 圈内人看 | 需 PR 通过 |
| Twitter/X 串联 | 国际曝光顶配 | 需要英文 |

**建议组合**：知乎 + 个人博客 + Twitter/X 摘要 + GitHub README 嵌入。

---

## Day 51 — 博客润色 + 同行评审

### Daily Tasks
1. 自己读一遍 v1，找出 5 处可砍 / 5 处需补
2. 找 1 个朋友（或 GPT/Claude）做同行评审
3. 完善图表：每张图加标题、坐标轴、legend
4. 发布！发到至少 2 个渠道

### 评审 Checklist
- [ ] 第一段能让外行知道你做了什么？
- [ ] 主结果表是否标了 baseline 来源？
- [ ] 视频 / GIF 是否能直接展示「成功」？
- [ ] 是否有 1-2 个「自黑」时刻（限制 / 失败）？
- [ ] 是否有 reproduce 路径？
- [ ] 拼写 / 语法（用 Grammarly）

### 配图规范
| 图类型 | 工具 | 注意 |
|--------|------|------|
| 架构图 | Excalidraw / draw.io / TikZ | 字体一致、留白 |
| 曲线 | matplotlib + seaborn | 加 95% CI / std |
| 柱状图 | matplotlib | 误差棒 |
| 视频 | imageio / OBS | overlay 任务名 |
| Hero figure | Figma / Photoshop | 一图说尽方法 |

---

## Day 52 — 简历 + GitHub 主页

### Daily Tasks
1. 简历改成 1 页（中英文各一份）
2. GitHub 主页：pin 3-5 个 repo，写 profile README
3. 准备好「30 秒自我介绍」（中英文）

### 简历模板（具身算法岗）

```
姓名 | 邮箱 | GitHub | 个人博客 | 期望地点

教育
- <学校>, <专业>, <学位>, <时间>

技术栈
- Programming: Python (advanced), C++ (proficient), CUDA basic
- Frameworks: PyTorch, JAX, Isaac Lab, MuJoCo, ROS2
- Robotics: cuRobo, Pinocchio, OCS2 (基础)
- ML: Diffusion, Transformer, RLHF, LoRA, FlashAttn

项目（按重要性排序，每个 4-6 行 bullet points）
1. <Week 7 项目> — 完成度 / 主要数字 / GitHub 链接
   - Built X using Y, achieved Z% on benchmark, faster than baseline by N%
   - Designed novel <component>, contributing +X% over baseline
   - Open-sourced with M+ stars / forks

2. <你的论文复现合集> — Diffusion Policy / OpenVLA / Dreamer 等
   - Reproduced 5 SOTA methods on standard benchmarks within 2% of paper

经验（如果有）
- <实验室 / 公司>, <职位>, <时间>
  - 一句话讲清楚你做了什么贡献

发表 / 公开内容（即使没正式 paper，技术博客 + 演讲都算）
- <博客标题>, <平台>, <日期>, X views

兴趣方向
- VLA, Sim2Real, Whole-body humanoid loco-manipulation
```

### GitHub Profile README 模板

```markdown
### Hi, I'm <Name> 👋

Embodied AI engineer focused on **<具体方向>**.

**🔥 Latest projects**
- [<Week 7 Project>](link) — <one line>
- [<复现合集>](link) — diffusion-policy / openvla / dreamer-v3

**📝 Recent writing**
- [博客 1](link)
- [博客 2](link)

**📫 Reach me**: email | twitter | linkedin
```

### Pin Repo 选择
| 优先 | 内容 | 备注 |
|------|------|------|
| 1 | Week 7 项目 | 你的招牌 |
| 2 | Bootcamp 学习仓库（精炼版） | 体现学习能力 |
| 3 | 一个论文复现 | 显示能读 paper |

> ⚠️ 不要把 8 周所有代码都暴露：把杂乱的 daily 实验整理成一个干净的 portfolio repo。

---

## Day 53 — 5 篇 SOTA 论文复盘

### Daily Tasks
为以下 5 篇论文，每篇写一份 1 页深度笔记 + 自己讲 10 分钟视频（不录也行，对着镜子讲）：

1. **Diffusion Policy** (Chi 2023)
2. **OpenVLA** (Kim 2024)
3. **π0** (Black 2024)
4. **GR00T N1** (NVIDIA 2025)
5. **TD-MPC2** 或 **Dreamer V3**（任选）

### 笔记模板

```markdown
# <Paper Title>

## One-line summary
<一句话>

## Why I picked this
<为什么这篇重要 / 为什么我选它>

## Method (画图)
<架构图 + 3-5 句话>

## Key insights
1. ...
2. ...
3. ...

## Experiment highlights
| Setting | Number |
|---------|--------|
| ... | ... |

## Limitations I notice
1. ...
2. ...

## What I would do next
<如果我接着做，会怎么做>

## Connection to my project
<和我 Week 7 项目的关系>
```

### 自讲 checklist（10 分钟版）
- 30s motivation
- 1 min related work
- 3 min method
- 3 min experiment
- 1 min limitation
- 1 min Q&A 自问自答

---

## Day 54 — 投递

### Daily Tasks
1. 列 10-15 家公司，填投递 tracker
2. 每家 craft 一份针对性 cover letter（300 字内）
3. 内推优先（LinkedIn / 知乎 / 直接邮件 PM / researcher）

### 头部公司清单（截至 2026 年）

#### 国内
| 公司 | 方向 | 备注 |
|------|------|------|
| 银河通用 | VLA + 操作 | 北京 |
| 智元机器人 | 人形 + RDT 团队 | 上海 |
| 星动纪元 | 人形 + VLA | 北京 |
| 宇树 | 四足 + 人形 | 杭州 |
| 跨维智能 | sim2real + 操作 | 深圳 |
| 自变量机器人 | 操作 + VLA | 深圳 |
| 千寻智能 | 操作 | 杭州 |
| 它石智航 | 移动操作 | 北京 |
| 穹彻智能 | 灵巧操作 | 上海 |
| 大疆 / 字节 / 美团 / 华为 / 小米 | 大厂线 | 各地 |

#### 海外
| 公司 | 方向 | 备注 |
|------|------|------|
| Physical Intelligence | π0 系列 | SF |
| Figure AI | Helix / 人形 | SF |
| 1X Technologies | NEO 人形 | Norway / SF |
| Skild AI | foundation model | SF |
| Tesla Optimus 团队 | 人形 | Bay Area |
| Boston Dynamics AI | locomotion | Boston |
| NVIDIA Robotics (GR00T) | 通用平台 | Santa Clara |
| Google DeepMind Robotics | RT 系列 | Mountain View |

### 投递 Tracker

| 公司 | 岗位 | 投递日 | 状态 | 联系人 | 备注 |
|------|------|-------|------|-------|------|
| 银河通用 | VLA Algorithm Engineer | 2026-08-XX | applied | XX | 内推 |
| ... | | | | | |

---

## Day 55 — 面试题准备

### 5 类高频题（每类准备 5-10 题）

#### 1. 手撕代码
- 写 PPO 主循环（Day 6 重温）
- 写 DDPM forward / sampling
- 写 Multi-Head Attention（含 mask）
- 写 GAE
- 写 Diffusion Policy 推理 loop（含 receding horizon）

#### 2. 概念解释
- BC / DAgger / IRL / GAIL 区别
- DDPM / DDIM / Flow Matching 区别
- TD-error / Bellman / GAE 推导
- LoRA 数学原理
- VLA 各家架构差异

#### 3. 系统设计
- 「设计一个能煮咖啡的 VLA pipeline」
- 「真机 30Hz 部署一个 7B VLA，怎么做？」
- 「设计一个数据飞轮：从真机到训练再到部署」
- 「给你 100 张 H100，怎么训自己的 VLA？」
- 「sim2real 的整套基础设施怎么搭？」

#### 4. 项目深挖
- 你 Week 7 项目的 motivation 是什么？
- 你做了哪些消融，哪些 surprised you？
- 失败案例是什么？为什么？
- 如果让你再做一次，怎么改？
- 哪个组件最难调？

#### 5. 行业 / 趋势题
- 你怎么看 VLA 现状？
- 大模型 + 机器人 vs 端到端 RL，未来 5 年？
- humanoid vs quadruped vs manipulator，哪个先商业化？
- sim2real 还是关键瓶颈吗？
- 你认为下个 breakthrough 在哪？

### 面试准备技巧
1. **STAR 法则**讲项目：Situation → Task → Action → Result
2. 准备 1-2 个「我做错了什么 + 怎么纠正」的故事（体现成长心态）
3. **反向提问**：每场面试结尾都问 2-3 个问题（显示真兴趣）

### 反向提问推荐
- 团队下一年的技术 roadmap？
- 数据 pipeline 的状态（teleop？sim？neural？）？
- 真机数量 / iteration 速度？
- 团队成员背景？
- 算法和硬件团队怎么协作？

---

## Day 56 — 模拟面试 + 收尾

### Daily Tasks
1. 找 1-2 个朋友或 GPT/Claude 当面试官，跑 2 场完整面试
2. 录视频自检：眼神、语速、口头禅
3. 把高频卡壳点列出，针对性补强
4. 写一份「8 周复盘」文档（自己看，不发布）

### 模拟面试流程（1 小时版）
```
0-5min   自我介绍
5-15min  项目深挖（Week 7）
15-30min 手撕（PPO 或 DP 或 Attention）
30-45min 系统设计
45-55min 反向提问 + 行业讨论
55-60min 反馈
```

### 8 周复盘模板

```markdown
# Embodied AI 8-Week Bootcamp 复盘

## 量化产出
- 写了 X 行代码
- 跑了 X 个实验，X 个 wandb runs
- 复现了 X 篇论文
- 项目获 X stars

## 我学到的最重要 5 件事
1. ...
2. ...
3. ...
4. ...
5. ...

## 我犯的最大 3 个错
1. ...
2. ...
3. ...

## 下一阶段（接下来 3 个月）
- ...
- ...
- ...
```

---

## 本周心态调节

8 周很累，但你已经走完最难的部分了。本周的核心心态是：

1. **不再加新内容**：管住手，不再开新坑
2. **优先「让别人能看见」**：博客 / 视频 / repo 比代码本身重要
3. **被拒绝是常态**：投 15 家，2-3 家进面试就算成功
4. **建立长期 visibility**：博客和 GitHub 是 5 年资产，不是 1 周资产

---

## 本周总投入预估

| 活动 | 小时 |
|------|------|
| 博客写作 + 润色 | 15h |
| 简历 / GitHub 整理 | 8h |
| 论文复盘 | 10h |
| 投递 + cover letter | 6h |
| 面试准备 + 模拟 | 12h |
| **合计** | **51h** |

---

## 8 周毕业宣言

> 8 周前，你只懂运动学和一点点 DL/RL 概念。
>
> 8 周后，你能：
> - 从零写 Transformer / Diffusion / PPO
> - 在 Isaac Lab 训出 locomotion 策略
> - 复现 SOTA VLA 并微调
> - 拿出一个 portfolio 项目 + 1 篇技术博客
> - 在面试中聊 30 分钟具身领域不卡壳
>
> 这只是起点。**真正的算法师，是在工业一线持续打磨 5-10 年炼出来的。**
> 但你已经站在了那个起点。

→ [Resources.md](./Resources.md) | [Milestones.md](./Milestones.md)
