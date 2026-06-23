# Day 07 — Week 1 Recap + Week 2 启动准备

> **日期**：2026-06-27（Week 1, Day 7 — 周日）
> **本日定位**：Week 1 收官。今天**不写新算法**、不开新坑，只做四件事：① 整理 GitHub repo + wandb；② 写一篇 800 字 weekly recap（求职背书素材）；③ PPO 模板再打磨一轮；④ Week 2 启动前的环境健康检查（Isaac Sim 5.1 GUI + 6.0 Docker + LeRobot 安装）。
> **总投入**：约 6 小时（周日宽松版）

---

## 0. 出发前自检（5 分钟）

```bash
[embai]$ conda activate embai
[embai]$ cd ~/embodied-ai-bootcamp-8w
[embai]$ git status                # 期望 clean
[embai]$ git log --oneline | head  # 看到 Day 1-6 共 6 个 commit

# wandb 项目盘点
[embai]$ python -c "
import wandb
api = wandb.Api()
for proj in ['embai-day01','embai-day02','embai-day04','embai-day05','embai-day06']:
    try:
        runs = list(api.runs(proj))
        print(f'{proj}: {len(runs)} runs')
    except Exception as e:
        print(f'{proj}: skip ({e})')
"
```

**今天主战场**：`week1-fundamentals/day7_recap/` + 全 repo 整理

```bash
[embai]$ mkdir -p ~/embodied-ai-bootcamp-8w/week1-fundamentals/day7_recap/{plots,recap}
[embai]$ cd ~/embodied-ai-bootcamp-8w/week1-fundamentals/day7_recap
```

---

## 1. 今日时间表（hour by hour）

| 时段 | 时长 | 任务 | 产出 |
|------|------|------|------|
| 09:00 - 10:30 | 1.5h | 整理 GitHub repo + 每个 day 写 README | repo 干净 |
| 10:30 - 12:00 | 1.5h | wandb 大盘扫一遍 + 截图最佳曲线 | recap 用图 5 张 |
| 14:00 - 15:30 | 1.5h | 写 800 字 weekly recap | `recap/week1.md` 完稿 |
| 15:30 - 16:30 | 1h   | PPO 数据流图 + 模板再打磨 | 一张 architecture 图 + 模板 v1.1 |
| 16:30 - 17:30 | 1h   | Week 2 启动前的环境健康检查 | Isaac 5.1/6.0 + LeRobot 全绿 |
| 19:30 - 20:00 | 0.5h | 把 recap 发到知乎 / 博客（可选） | 公开链接 |
| 20:00 - 20:30 | 0.5h | Week 2 Day 8 morning paper 预读 | 笔记 1 页 |

---

## 2. 上午 09:00 - 10:30 — GitHub Repo 整理

### 2.1 顶层 README 重写

```bash
cd ~/embodied-ai-bootcamp-8w
cat > README.md <<'EOF'
# Embodied AI Bootcamp — 8 Weeks

跟随 [Embodied-AI 学习计划](../Documents/Embodied-AI) 的实战代码仓库。

## Progress
- [x] **Week 1** — PyTorch + RL Foundations (2026-06-19 ~ 2026-06-25)
- [ ] Week 2 — Isaac Sim + LeRobot + BC/ACT
- [ ] Week 3 — Diffusion Policy + 表征学习
- [ ] Week 4 — 动力学 + 7-DoF RL + 双臂 + 移动操作
- [ ] Week 5 — VLM 基础
- [ ] Week 6 — VLA 模型深读
- [ ] Week 7 — 端到端项目（轮式双臂 / 7-DoF）
- [ ] Week 8 — 论文 + 求职冲刺

## Week 1 Highlights
| Day | 主题 | wandb run | 关键产出 |
|-----|------|-----------|----------|
| 1 | PyTorch + MNIST | embai-day01 | acc 98.5% |
| 2 | Mini-GPT (Tiny Shakespeare) | embai-day02 | 能采样莎士比亚风文本 |
| 3 | Batched SE(3) | (单测) | gradcheck 双向通过 + Pinocchio < 1e-6 |
| 4 | MDP / VI / Q-Learning | embai-day04 | FrozenLake success > 0.7 |
| 5 | REINFORCE / A2C | embai-day05 | CartPole stable @ 500 |
| 6 | PPO + GAE | embai-day06 | HalfCheetah-v5 reward > 3000 |
| 7 | Recap | — | weekly recap + 模板化 |

## Templates
- [`templates/ppo_clean/`](./templates/ppo_clean) — 200 行 PPO，CartPole/LunarLander/HalfCheetah 验证

## Folder Layout
```
week1-fundamentals/
  day1_pytorch/   day2_transformer/   day3_se3/
  day4_mdp/       day5_pg/            day6_ppo/   day7_recap/
templates/
  ppo_clean/
```
EOF
```

### 2.2 每个 day 子文件夹的 README

```bash
# 用一个脚本批量生成（逐个填）
for d in day1_pytorch day2_transformer day3_se3 day4_mdp day5_pg day6_ppo day7_recap; do
  if [ ! -f "week1-fundamentals/$d/README.md" ]; then
    cat > "week1-fundamentals/$d/README.md" <<EOF
# $d

详见 \`day_plan/Day0X.md\`(对照填) 和本目录下 \`NOTES.md\`(执行复盘)。

## Files
$(ls week1-fundamentals/$d 2>/dev/null | sed 's/^/- /')
EOF
  fi
done
```

> 然后**人工补**每个 README 的关键 takeaway（2-3 行），别让它们变成无意义脚手架。

### 2.3 .gitignore 兜底

```bash
cat > .gitignore <<'EOF'
__pycache__/
*.pyc
.ipynb_checkpoints/
wandb/
data/
checkpoints/
*.pt
*.pth
runs/
logs/
.venv/
.vscode/
.idea/
.DS_Store
EOF

git add .gitignore README.md week1-fundamentals/*/README.md
git commit -m "Day 7: organize repo + per-day READMEs"
```

### 自检
- [ ] 顶层 README 一眼能看出 Week 1 学了什么
- [ ] 7 个 day 子文件夹各有自己的 README
- [ ] `git status` clean，`git log --oneline` 7 条 commit 对得上 7 天

---

## 3. 上午 10:30 - 12:00 — wandb 大盘 + 截图

### 3.1 把 5 个 day 的 wandb 项目盘点

```python
# day7_recap/scripts/wandb_audit.py
import wandb
api = wandb.Api()
projects = ["embai-day01", "embai-day02", "embai-day04",
            "embai-day05", "embai-day06"]
for p in projects:
    try:
        runs = list(api.runs(p))
        print(f"\n=== {p} ({len(runs)} runs) ===")
        for r in runs[:10]:
            print(f"  - {r.name}  state={r.state}  best={r.summary.get('best', '-')}")
    except Exception as e:
        print(f"{p}: {e}")
```

### 3.2 必出图（5 张，存到 `day7_recap/plots/`）

| 图 | 来源 | 用途 |
|----|------|------|
| `fig1_mnist.png` | Day 1 wandb（mnist-raw / mnist-lightning） | recap 第一节 |
| `fig2_gpt_loss.png` | Day 2 wandb（mini-GPT loss）| recap 第二节 |
| `fig3_q_learning.png` | Day 4 wandb（4x4 + 8x8 + γ 扫描）| recap 第三节 |
| `fig4_pg_variance.png` | Day 5 已生成的 `variance_compare.png` 复制过来 | recap 第四节 |
| `fig5_ppo_cheetah.png` | Day 6 wandb（HalfCheetah）| recap 第五节 |

> 截图直接 wandb 网页 → Add to report → 截屏，5 分钟一张，**不要折腾自动化**。

### 自检
- [ ] 5 张图齐了
- [ ] 每张图旁边写一句话 caption（用什么超参 / 学到了什么）

---

## 4. 下午 14:00 - 15:30 — 写 800 字 Weekly Recap

### 4.1 用模板（贴到 `day7_recap/recap/week1.md`）

```markdown
# Week 1 Recap — PyTorch + RL Foundations

**作者**：(你)
**周期**：2026-06-19 → 2026-06-25
**总投入**：~55 小时（论文 10h / 代码 25h / 调参 15h / 整理 5h）
**仓库**：https://github.com/jianghaoyu13/Embodied_AI

## TL;DR
本周从 0 把 PyTorch 工程肌肉记忆 + RL 基本功打通：手写 MNIST / Mini-GPT / batched SE(3) /
VI-PI-QL / REINFORCE-A2C-PPO，最后一天 PPO 在 HalfCheetah-v5 上 2M step 拿到 reward _____。
所有实验全程 wandb 跟踪，模板化的 PPO 代码下周直接进 Isaac Lab。

## ✅ Done
- Day 1 MNIST：手写 + Lightning 双版本，test acc 98.5%
- Day 2 Mini-GPT：Tiny Shakespeare 训练 1 epoch 能采样
- Day 3 Batched SE(3)：gradcheck 双向通过，Pinocchio 一致性 < 1e-6
- Day 4 MDP：FrozenLake 4x4/8x8 VI/PI/Q-Learning，Cliff Walking 上 SARSA vs Q-Learning 对比
- Day 5 PG：REINFORCE → A2C，CartPole 稳定 500
- Day 6 PPO：CartPole / LunarLander / HalfCheetah-v5 三连，模板化

## 🔥 Insights（必带）
> 这一节是本 recap 的主菜。不要写"我学了 PPO"——写**学到的、能复用的、原本以为不是这样**的。

1. **PPO 难的不是算法本身，是工程细节**：orth init / advantage norm / vec env / clip_vloss
   这些在论文 1 行带过，实际不写就训不出来。Huang 的 *37 Implementation Details* 是必读。
2. **on-policy 不等于"每次重新采样"**：PPO 用 importance ratio + clip 等于"近似 on-policy"，
   K-epoch 重用 rollout 是它样本效率比 vanilla PG 高的关键。
3. **value-based 和 policy-based 的真分水岭不是离散/连续，是 multimodal**：
   diffusion policy（Week 3）能学多模态 demo，PG/PPO 不能 — 这是为什么后面要 IL/diffusion 分开学。
4. **SE(3) 数值稳定的难点全在 θ→0 的 Taylor 展开**：四元数翻号、ξ 顺序约定、左右乘扰动 ——
   这些先写在 CONVENTIONS.md，后两天调试才不会怀疑人生。

## ⚠️ Pitfalls Hit（学费清单）
- Day 4：`done=True` 时没截断 `gamma * V[s2]`，幻觉收益污染 V，success 卡 0.4
- Day 5：第一次 PG loss 没加负号，return 越来越低；没 detach baseline，critic 抖
- Day 6：HalfCheetah obs/reward norm 不加直接学不出来；`approx_kl` 飙到 0.1 才发现 lr 太大

## 📊 关键数据（图见 plots/）
- MNIST acc: 98.5%
- Mini-GPT val loss: ___ → ___ (1 epoch)
- FrozenLake 4x4 slippery: VI / PI / Q-Learning success = ___ / ___ / ___
- CartPole REINFORCE / A2C 收敛 episode: ___ / ___
- HalfCheetah-v5 PPO reward @ 2M step: ___

## ➡️ Next Week (Week 2 — Isaac Sim + LeRobot + BC/ACT)
- 装 Isaac Lab（在 6.0 容器内）
- 跑通 ALOHA Cube Transfer（仿真版）的数据采集
- ACT 在自录 demo 上训练，成功率 ≥ 80%
- LeRobot pi0 / ACT 跑通推理流程

## 🧰 Reusable Templates
- [`templates/ppo_clean/`](https://github.com/jianghaoyu13/Embodied_AI/tree/main/templates/ppo_clean) — 200 行 PPO，本周末验证

## 📚 References
- *Attention Is All You Need* (Vaswani 2017)
- *A micro Lie theory for state estimation* (Solà 2020)
- *Sutton & Barto* §3-4, §13
- *PPO* (Schulman 2017), *GAE* (Schulman 2016)
- *37 Implementation Details of PPO* (Huang 2022)
```

### 4.2 写作 checklist

- [ ] **TL;DR 3 句话**讲清楚学了什么 / 跑出什么数 / 下周做什么
- [ ] **Insights 至少 3 条**，每条要有"原本以为 X，实际是 Y"的反直觉成分
- [ ] **Pitfalls 至少 3 条**，每条配一句话怎么修的
- [ ] **数据用具体数字**（acc / reward / step），不要"还不错"这种
- [ ] **下周计划具体到 ≤ 5 个 bullet**，不要 "继续学"

### 自检
- [ ] 写完整 800 字（不算引用 / 数据 / 链接）
- [ ] 给一个朋友（甚至 GPT）读一遍，能 30 秒抓住主旨
- [ ] 不出现：「学到了很多」「收获满满」「期待下周」这种水词

---

## 5. 下午 15:30 - 16:30 — PPO 数据流图 + 模板打磨

### 5.1 手画一张 PPO 数据流图

```
┌──────────────┐    rollout (n_steps × n_envs)    ┌──────────────┐
│  vec env     │ ────────────────────────────────▶│ RolloutBuffer│
│ (n=8)        │   obs/action/log_prob/reward/done│  + values     │
└──────────────┘                                  └──────────────┘
                                                          │
                                                          ▼
                                              ┌──────────────────┐
                                              │ compute_gae()    │
                                              │ A_t = δ_t + γλ·  │
                                              │      A_{t+1}     │
                                              └──────────────────┘
                                                          │
                                                          ▼
   K epoch × M minibatch ┌──────────────────────────────────────┐
   (用旧 rollout)         │ PPO update:                          │
                         │  ratio = exp(new_logp - old_logp)    │
                         │  L^CLIP = min(r·A, clip(r,1±ε)·A)    │
                         │  + 0.5·v_loss − ent_coef·H           │
                         └──────────────────────────────────────┘
                                                          │
                                          ┌───────────────┴───────────────┐
                                          ▼                               ▼
                                  approx_kl > target_kl?         next rollout
                                  → early stop                   (drop old data)
```

把这张图画出来贴到 `day7_recap/plots/ppo_dataflow.png`（用 ASCII 上面的图也可，用 Mermaid 更标准）。

### 5.2 PPO 模板打磨 v1.1

把 Day 6 写好的 `templates/ppo_clean/` 再过一遍：

- [ ] 抽出 `configs/ppo_default.yaml`、`ppo_continuous.yaml` 两个配置
- [ ] 加 `train_ppo.py` 命令行入口（`python train_ppo.py --config ppo_continuous.yaml`）
- [ ] 写 `templates/ppo_clean/README.md` —— 如何在 Isaac Lab vec env 上接（占位 + 注释「Week 4 填实」）

```bash
git add templates/ppo_clean/ week1-fundamentals/day7_recap/
git commit -m "Day 7: ppo template v1.1 (yaml configs + cli entry)"
```

### 自检
- [ ] PPO 数据流图能向新人 5 分钟讲清
- [ ] 模板的 `python train_ppo.py --config ppo_default.yaml` 能直接跑 CartPole

---

## 6. 下午 16:30 - 17:30 — Week 2 启动前的环境健康检查

> Week 2 Day 8 起 Isaac Lab 训练全部走 6.0 容器。今天**先 dry-run 一遍**，明天早上不要被环境问题耽误。

### 6.1 Isaac Sim 5.1 本地（GUI 备用）

```bash
# 找到 5.1 安装位置（Day 1 已确认过）
[host]$ ls ~/.local/share/ov/pkg/ | grep "isaac-sim-5.1" || \
        find ~ -maxdepth 4 -name "isaac-sim.sh" 2>/dev/null

# 启动一次 GUI（确认还能开 + shader cache 没坏）
[5.1]$ cd ~/isaacsim    # 你实际路径
[5.1]$ ./isaac-sim.sh
# 起来后随手开一个 USD 场景，关掉。
```

### 6.2 Isaac Sim 6.0 容器 + Isaac Lab

```bash
# 容器
[host]$ ~/isaac_workspace/run_isaac6.sh

# 容器内 sanity
[isaac6]$ nvidia-smi             # 4090 在
[isaac6]$ python -c "
from isaacsim import SimulationApp
sim = SimulationApp({'headless': True})
print('Isaac Sim 6.0 OK')
sim.close()
"

# Isaac Lab 是否已装（Day 1 §7.2 说本日不装,Week 2 Day 8 才装）
[isaac6]$ ls /workspace/IsaacLab/isaaclab.sh && \
          /workspace/IsaacLab/isaaclab.sh --help | head -20 || \
          echo "Isaac Lab 未装,Week 2 Day 8 安装"
```

> 如果 Isaac Lab 没装，**今天不装** —— Day 8 是它的官方 slot。今天只确认容器能起来即可。

### 6.3 LeRobot 装到 `embai`

```bash
[embai]$ conda activate embai
[embai]$ pip install lerobot
[embai]$ python -c "
import lerobot
print('lerobot:', lerobot.__version__)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
print('dataset class ok')
"

# 不要在今天下数据集（GB 级，明天上午 Day 8 再下）
```

### 6.4 磁盘空间盘点

```bash
[host]$ df -h ~ ~/isaac_workspace ~/embodied-ai-bootcamp-8w
# 期望 home 空闲 > 100GB（Week 2 数据集会膨胀）
[host]$ du -sh ~/isaac_workspace/nv_cache/* 2>/dev/null
# 期望 computecache 已经长到 几个 GB（说明 shader 缓存生效）
```

### 自检
- [ ] 5.1 GUI 能起 + 一个 USD 场景能开
- [ ] 6.0 容器能进 + `nvidia-smi` 正常 + `SimulationApp({'headless': True})` 通过
- [ ] LeRobot import OK
- [ ] home 空闲 > 100GB；nv_cache/computecache 已开始持久化

---

## 7. 晚上 19:30 - 20:00 — 公开发布（可选）

如果 recap 写得有干货，**强烈建议公开**：

- 知乎专栏（"具身智能学习日志 W1"）
- 个人博客（GitHub Pages / Notion 公开页）
- Twitter / 朋友圈摘要 + 链接（求职背书）

8 周下来 8 篇连起来 = 一份天然的求职作品集。

> **不要**发到内部群组（同事 / 公司 Slack）—— 这是为求职背书写的，应面向公开读者。

### 自检
- [ ] 至少发到 1 个公开平台
- [ ] 公开链接补回 `day7_recap/recap/week1.md` 顶部

---

## 8. 晚上 20:00 - 20:30 — Week 2 Day 8 morning paper 预读

> 不是看完 —— 是 skim 一遍知道明早要读什么、要重点抓什么。

### 8.1 Day 8 morning paper

- *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization* (Peng et al., 2018) —— DR 起源
- 浏览 [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/main/index.html) 的 quickstart + manipulation 任务 ID 列表
- 浏览 [LeRobot docs](https://github.com/huggingface/lerobot) 的 README quickstart

### 8.2 笔记产出 — `day_plan/notes/Day08_skim.md`

3-5 行 bullet：
- DR 关键想法是什么（一句话）
- Isaac Lab 提供哪些 manipulation 任务（列 5 个）
- LeRobot 的 dataset / policy / runner 三层抽象（一句话）

### 自检
- [ ] 3 个资源各 skim 过一次
- [ ] 笔记 ≤ 1 页

---

## 9. 今日完成确认（必达 / 加分）

### 必达（5/6 才算合格）
- [ ] GitHub repo 顶层 README + 7 个 day README 都写好，git push
- [ ] wandb 5 个项目盘点 + 5 张关键图截好
- [ ] 800 字 weekly recap 完稿
- [ ] PPO 模板 v1.1（yaml configs + cli entry）
- [ ] 5.1 GUI / 6.0 容器 / LeRobot 三件 sanity check 全绿
- [ ] Day 8 morning paper skim 过

### 加分
- [ ] recap 公开发布（知乎 / 博客）
- [ ] PPO 数据流图用 Mermaid / draw.io 画成正式图
- [ ] 把 Week 1 6 天的 NOTES.md 汇总成一份 `week1-fundamentals/INSIGHTS.md`
- [ ] 把 Day 6 的 `target_kl` early-stop 实测一组（0.015 vs None）

---

## 10. 下周 (Week 2) 预告 — Isaac Sim + LeRobot + BC/ACT

- **Day 8（周一）**：Isaac Lab 安装到 6.0 容器内 + 跑通官方 tutorial + 一个 manipulation 任务
- **Day 9**：MuJoCo / MJX 基础 + 数据采集 pipeline
- **Day 10**：LeRobot quickstart，跑通 pi0 推理
- **Day 11-13**：ALOHA Cube Transfer 数据采集 → ACT 训练 → 推理
- **Day 14**：Week 2 周报

详见 [Week02.md](../week_plan/Week02.md)。

---

## 11. 反向提醒（容易忘）

- ❗ **recap 别写成 brain dump**：必须有 TL;DR + Insights + Pitfalls + Next，结构严格
- ❗ **公开发布要去掉敏感信息**：实习公司名、内部数据、雇主邮件 —— 全删
- ❗ **Isaac Lab 今天不装**，Day 8 才是它的 slot；今天只确认容器能起来
- ❗ **LeRobot 今天不下数据集**（HuggingFace 下大数据集要走代理 / 大流量），Day 10 再下
- ❗ **wandb 截图前**，把 run 改成有意义的名字（`bs-128`、`gamma-099`），不然图上一片 `run-1 run-2`
- ❗ **PPO 数据流图**画完了再贴 GitHub —— 图床用 GitHub 自身（仓库内 .png）最稳
- ❗ **周日不要再开新坑**：今天的目标是"收尾 + 充电"，不是"再多学一个 SAC"

---

## 12. 如果今天彻底崩了 — 救火方案

如果到 17:00 还没写完 recap：

1. **保 recap 不保整理**：先把 800 字写完发出去（求职背书优先），repo 整理推到下周一晚上
2. **一图代千字**：实在没时间写文字，把 5 张关键 wandb 截图 + 30 字 caption 各贴一张，发到知乎当作图文版 recap
3. **Day 8 不延期**：环境健康检查（§6）必须做完，否则明早 Isaac Lab 装到一半发现 Docker 起不来 = 半天没了

---

## Week 1 完赛宣言

如果你做完了 Day 1-7 必达项 ≥ 90%，**给自己一个仪式感**：
- 在 `~/embodied-ai-bootcamp-8w/` 根目录 `touch WEEK1_COMPLETE`
- `git commit -am "🎉 Week 1 complete"`
- 关掉所有终端，**周日晚上不要碰键盘**，明早 7 点见 Day 8。

---

→ [Week01.md](../week_plan/Week01.md) | [Week02.md](../week_plan/Week02.md) | [Day08.md](./Day08.md)（Week 2 起，待生成）
