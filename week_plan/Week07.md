# Week 7 — 端到端项目实战（轮式双臂 / 7-DoF 操作方向）

> **本周定位**：把前 6 周学到的东西，**捏成一个 GitHub 上能让面试官点头的项目**。
> 已为你校准：所有候选项目都是**轮式双臂 / 7-DoF 机械臂的操作 / 抓取 / 移动操作**方向。

---

## 本周目标

完成本周后，你应该能：
- [ ] 拿出一个完整的、有 README / 视频 / wandb / 复现脚本的 GitHub repo
- [ ] 在某个公开 / 自建 benchmark 上报出有竞争力的数字
- [ ] 写出 1 篇技术博客（Week 8 用）
- [ ] 在面试中能 30 分钟讲清「motivation → 方法 → 实验 → 局限」

---

## 项目方向（三选一，全部围绕轮式双臂 / 7-DoF）

> 根据你的 4090 24GB + 求职方向选 **一个**。不要贪多。

### 🅐 操作路线 — Diffusion Policy on RoboCasa Mobile Manipulation
**适合**：你是首选，最贴近轮式双臂人形真实场景

**任务**：
- 在 RoboCasa（厨房 mobile manipulation）选 5 个长程任务，例如：
  - PnP-Counter-to-Cab
  - PnP-Cab-to-Counter
  - OpenSingleDoor / CloseSingleDoor
  - TurnOnStove
  - PourFromKettle
- 实现 baseline：DP-CNN（你 Week 3 已掌握）
- 创新点（任选 1-2）：
  - DP + DINOv2 backbone 替换
  - DP + 3D point cloud（DP3）
  - DP + Flow Matching action head（替换 DDPM，加速推理）
  - DP + cuRobo residual：让 DP 输出 EE 残差，cuRobo 给 base trajectory

**成功指标**：
- 5 个任务平均成功率 ≥ 50%（强 baseline 难度）
- 至少做 1 项可量化对比 (+5% 以上)
- 推理速度 ≥ 10Hz @ 4090

**硬件**：4090 24GB ✅

---

### 🅑 双臂协作路线 — Bimanual ALOHA / 7-DoF + 7-DoF
**适合**：贴近你后续轮式双臂的双臂部分，可与轮式底盘解耦先攻克双臂

**任务**：
- 任务集（任选 4 个，包含至少 1 个双臂强协作任务）：
  - sim_transfer_cube（基础协作）
  - sim_insertion（精度任务）
  - bimanual screw / unscrew（接触富）
  - bimanual cloth folding（柔性物体）
  - bimanual pick-and-handover（交接）
- baseline：ACT（Week 2）+ DP（Week 3）双 baseline
- 创新点：
  - 双臂 cross-attention 比共享 encoder 是否更好？做对照
  - Action chunking 长度对协调性影响（chunk=50 vs 100 vs 200）
  - 加 self-collision penalty 训练阶段对真机部署的影响

**成功指标**：
- 4 个任务平均成功率 ≥ 75%
- 至少 1 个 high-precision 任务 (insertion) ≥ 60%
- 完整消融：chunk size / cross-attention / DP vs ACT

**硬件**：4090 24GB ✅

---

### 🅒 VLA 路线 — OpenVLA / RDT 微调到自建轮式双臂数据
**适合**：求职 VLA / 大模型方向，希望与 π0 / GR00T / RDT 团队接轨

**任务**：
- 在 LIBERO 4 个 suite 微调 OpenVLA，逼近论文数字（基础线）
- **创新主线（二选一）**：
  - 把 OpenVLA 的 7-DoF 单臂动作头改成 14-DoF 双臂 + 3-DoF base 输出
  - 用 RDT-1B 在 LeRobot 公开双臂数据集上微调，加 mobile base condition
- （可选）小数据真机迁移：自录 30-50 条 demo，做 LoRA 微调

**成功指标**：
- LIBERO-Spatial / Object / Goal ≥ 75%, LIBERO-10 ≥ 45%
- 双臂数据集上微调成功率 ≥ baseline + 5%
- 推理速度报告（FPS @ 4090）+ 量化 / chunk 优化

**硬件**：4090 24GB（QLoRA + grad checkpointing），勉强能跑 7B

---

## Day 43 — Day 1：项目立项 + 环境固化

### Daily Tasks
1. **写 PROJECT.md**：明确问题、metric、baseline、stretch goal
2. **冻结环境**：`pip freeze > requirements.txt` + `Dockerfile`
3. **建 GitHub repo**：起个好名字（避免 `final-project`），写好初版 README
4. **跑通 baseline**：第 0 个 reproducible run，wandb 上看到 loss 曲线

### PROJECT.md 模板

```markdown
# <Project Name>

## Problem
What I'm solving in 1 sentence.

## Metric
- Primary: <e.g., success rate on RoboCasa-PnP-Counter-to-Cab>
- Secondary: <e.g., inference FPS, training time>

## Baseline
- <Reference paper> reports X%, code at <link>

## Stretch goal
- Beat baseline by Y% with <my contribution>

## Hardware
- 1× RTX 4090, 64GB RAM, 1TB NVMe

## Timeline
- D1-2: setup & baseline
- D3-4: 主实验
- D5: 消融
- D6: 视频 + 文档
- D7: 写博客
```

### Repo 结构（推荐）

```
project-name/
├── README.md
├── PROJECT.md
├── requirements.txt
├── Dockerfile
├── configs/                # hydra 配置
├── src/
│   ├── data/
│   ├── models/
│   ├── policies/
│   ├── envs/
│   └── utils/
├── scripts/
│   ├── train.py
│   ├── eval.py
│   └── make_video.py
├── tests/
├── notebooks/              # 调试 / 可视化
├── checkpoints/.gitignore
├── logs/.gitignore
└── docs/
    └── results.md
```

---

## Day 44 — Day 2：核心实验跑起来

### Daily Tasks
1. 把第一个完整训练跑起来（应能跑 4-12h）
2. 同时启动一个 baseline 对照（比如纯 BC）
3. 准备 evaluation pipeline：脚本化、可重复
4. 中途用 wandb 检查训练健康度

### 训练健康度自检

| 指标 | 健康 | 不健康 → 行动 |
|------|------|-------------|
| Loss 曲线 | 平滑下降 | 振荡：lr 大 / batch 小 |
| GPU 利用率 | > 85% | < 50%：IO 瓶颈 |
| Eval success | 缓慢上升 | 平 / 下降：reward bug 或 OOD |
| Validation loss | 跟随 train | 早期就 diverge：过拟合 |

---

## Day 45 — Day 3：关键消融

> 没消融的实验 = 没实验。这天的目的是搞清楚**你的方法哪部分在 work**。

### Daily Tasks
1. 列 5-7 个消融维度（路线相关）
2. 每个跑短训练（够看出趋势就好）
3. 用 hydra `-m` 多次 sweep
4. 整理消融表 + bar chart

### 消融维度建议

#### 🅐 操作路线消融
- backbone：ResNet18 / DINOv2 / SigLIP
- horizon：To∈{1,2,4}, Tp∈{8,16,32}
- noise scheduler：DDPM/DDIM/FM
- input modality：2D / 2D+depth / 3D point cloud
- camera setup：wrist / base / wrist+base
- base 控制方式：直接 / cuRobo residual
- data scale：100% / 50% / 25%

#### 🅑 双臂消融
- chunk size：50 / 100 / 200
- shared encoder vs cross-attention
- action 表示：joint pos / EE delta (OSC)
- self-collision penalty：on/off (training-time only)
- DP vs ACT 同任务对照
- 数据量：50% / 100% demo

#### 🅒 VLA 路线消融
- LoRA rank：8 / 16 / 32 / 64
- finetune layers：only proj / + LLM / full
- action tokenizer：bin / VQ / FAST
- image augmentation：on/off
- chunk size：1 / 4 / 8 / 16
- base + arm action 联合 vs 解耦输出

### 输出
- 一张消融柱状图
- 一段「我的方法 work 是因为 X，而不是 Y」论述

---

## Day 46 — Day 4：Stretch Goal / 创新点

### Daily Tasks
1. 实现一个 **non-trivial 的创新**（不是「跑了一遍 SOTA」）
2. 跑 final 实验，得到主表数字
3. 录主任务视频（成功 + 失败案例都要）

### 创新点示例（已为你校准为操作 / 双臂 / 移动操作）

| 路线 | 创新点 | 难度 |
|------|--------|------|
| Manip | DP + DINOv2 替换 ResNet | ⭐ |
| Manip | DP3 vs DP2D 在 RoboCasa | ⭐⭐ |
| Manip | DP + Flow Matching 替换 DDPM，推理 < 8 步 | ⭐⭐⭐ |
| Manip | DP + cuRobo residual：base 用 cuRobo，arm 用 DP | ⭐⭐⭐ |
| Bimanual | Cross-attention 双臂 head（替代 shared encoder） | ⭐⭐ |
| Bimanual | 双臂 + Self-collision aware loss | ⭐⭐⭐ |
| Mobile | base + arm 解耦 vs 联合 action head | ⭐⭐⭐ |
| Mobile | 加入 wrist + base camera 双 view | ⭐⭐ |
| VLA | OpenVLA 的 bin → FAST tokenizer | ⭐⭐ |
| VLA | OpenVLA 加 3D point cloud 输入 | ⭐⭐⭐ |
| VLA | OpenVLA 改 14-DoF 双臂 + 3-DoF base 输出 | ⭐⭐⭐⭐ |

> ⭐⭐ 难度足以打动面试官，不用追 ⭐⭐⭐⭐。

---

## Day 47 — Day 5：视频 / 可视化

### Daily Tasks
1. 录主任务 rollout 视频（每任务 3-5 个 episode）
2. 制作 1 张 hero figure（论文风格首图）
3. 制作 1 张架构图
4. 制作消融柱状图 + 主表

### 视频制作要点
- 30 fps、1080p、长度 30-60s
- 加任务名 + 时长 overlay
- 失败 case 也放，体现诚实
- 上传 YouTube / 哔哩哔哩 + 内嵌 README

### 推荐工具
- `imageio` / `moviepy` for Python 拼接
- OBS / Kdenlive for 后期
- matplotlib + LaTeX 数学公式
- tikz / Excalidraw for 架构图

### Code Template (Episode Highlights)

```python
# scripts/make_video.py
import imageio, numpy as np
from PIL import Image, ImageDraw

frames_all = []
for ep_idx in range(5):
    frames = run_episode(env, policy)        # list of [H, W, 3]
    for i, f in enumerate(frames):
        img = Image.fromarray(f)
        d = ImageDraw.Draw(img)
        d.text((10, 10), f"Ep {ep_idx}  t={i*0.1:.1f}s", fill="white")
        frames[i] = np.array(img)
    frames_all.extend(frames)

imageio.mimsave("highlights.mp4", frames_all, fps=30, quality=8)
```

---

## Day 48 — Day 6：文档 + Reproducibility

### Daily Tasks
1. 完整 README（含一键复现命令）
2. 写一个 5-min Quickstart
3. 上传 checkpoint 到 HuggingFace Hub
4. 找一个朋友（或 GPT 当朋友）按 README 跑一遍，看能不能复现

### README 模板

```markdown
# <Project Name>

<one-line tagline, e.g. "Diffusion Policy + cuRobo Residual for RoboCasa Mobile Manipulation">

[![](paper.svg)](paper.pdf) [![](hf.svg)](hf-link) [![](video.svg)](youtube)

## TL;DR
- 一句话方法
- 一句话结果
- 一句话用处

## Demo
<gif>

## Results
| Method | RoboCasa-PnP | OpenDoor | TurnOnStove |
|--------|--------------|----------|-------------|
| Vanilla DP | xx | xx | xx |
| **Ours**   | **xx** | **xx** | **xx** |

## Quick Start (5 min)
```bash
git clone ... && cd ...
pip install -e .
python scripts/eval.py --task pnp_counter_to_cab --ckpt hf://your/ckpt
```

## Train from Scratch
```bash
python scripts/train.py task=pnp_counter_to_cab seed=42
```

## Reproduce All Results
```bash
bash scripts/run_all.sh
```

## Citation
...
```

---

## Day 49 — Day 7：润色 + 内部 review

### Daily Tasks
1. 自己 code review：删冗余、加注释、清 commented-out code
2. 添加 `pytest` 至少 5 个核心模块测试
3. 跑 `ruff check .` 让代码 lint 干净
4. 全 repo 加 license（推荐 MIT 或 Apache 2.0）

### 自检清单（按这个打勾）
- [ ] README 一键复现命令能跑
- [ ] requirements.txt 完整且 pin 版本
- [ ] checkpoint 上传 HuggingFace
- [ ] 视频上传 YouTube + GIF 嵌 README
- [ ] 主结果表 + 消融表 + 架构图齐全
- [ ] 至少 1 个 Jupyter notebook 用于可视化
- [ ] License 文件
- [ ] CITATION.cff
- [ ] 没有 commented-out 代码
- [ ] `git log` 清晰，commit message 不是 "fix"

---

## 工程化规约（贯穿整周）

### 1. Hydra 配置
- 所有超参在 `configs/` 中，不在代码硬编码
- 用 `@hydra.main` 入口，可 `python train.py override=...`

### 2. wandb 全程
- 每个 run 记 git_sha, hostname, gpu_name
- log model checkpoint
- log video / image

### 3. tmux + 长训练
- 标准做法：`tmux new -s train`，进 conda env，跑训练
- 用 `nohup` 备份

### 4. 单元测试
```python
# tests/test_dataset.py
def test_dataset_shape():
    ds = MyDataset(...)
    sample = ds[0]
    assert sample["image"].shape == (3, 224, 224)
    assert sample["action"].shape == (16, 7)
```

### 5. Pre-commit hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks: [{id: ruff}, {id: ruff-format}]
```

---

## 时间陷阱（提前知道）

| 陷阱 | 表现 | 对策 |
|------|------|------|
| 调参黑洞 | 一直在调 lr | 设 hard time limit，2 天后无论如何向前 |
| 数据 bug | 训练曲线奇怪 | 第 1 天就写 sanity check（10 条数据过拟合） |
| Repo 不能复现 | 跑不通 | 用一个新环境从 README 试一次 |
| 视频太晚 | Day 7 才录 | Day 2 就录第一版 |
| 想做太多 | 进度落后 | 锁死 stretch goal，不要再加 |

---

## 本周总投入预估

| 活动 | 小时 |
|------|------|
| 训练 / 实验（含等待） | 35h |
| 写代码 / 调 bug | 15h |
| 文档 / 视频 | 8h |
| 创新模块 | 10h |
| **合计** | **68h**（这周强度最高） |

→ [Week08.md](./Week08.md)
