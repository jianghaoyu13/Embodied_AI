# Week 6 — VLA 主流模型深读 + 复现微调

> **本周定位**：本周决定你的「上限」。π0 / GR00T / RDT / Helix 是 2024-2026 工业级 VLA 的代表，理解它们的设计取舍是你在面试中拉开身位的核心武器。
> **针对你的方向（轮式双臂 + 7-DoF）特别提示**：本周复现微调时优先关注 **双臂 / 移动操作能力强的模型**：RDT-1B（双臂原生）、π0/π0.5（双臂 + 移动）、GR00T N1.5（人形双臂）。LIBERO 是公开 benchmark 中最贴近你目标场景的，重点用它做评测。

---

## 本周目标

完成本周后，你应该能：
- [ ] 对 RT-1 → RT-2 → RT-X → OpenVLA → π0 → GR00T 的演化讲一个清晰的故事
- [ ] 解释 π0 / π0.5 的 Flow Matching action expert 设计
- [ ] 解释 GR00T N1 / N1.5 的 System1/System2 双脑架构
- [ ] 解释 RDT-1B 的 unified action space + DiT 设计
- [ ] 在 LIBERO benchmark 微调一个 VLA 并报数

---

## 本周可交付物

```
week6-vla/
├── day36_rt_family/          # RT-1/2/X 家族梳理
├── day37_pi0/                # π0 / π0.5 复现
├── day38_groot/              # GR00T N1/N1.5 inference
├── day39_rdt_cogact/         # RDT-1B / CogACT
├── day40_helix_hierarchical/ # Helix 双系统架构
├── day41_libero/             # LIBERO 微调 + 数字
└── day42_recap/              # 周报 + 架构对比图
```

---

## Day 36 — RT 家族系统梳理

### 早晨论文 (3h，必须深读)
- *RT-1: Robotics Transformer for Real-World Control at Scale* (Brohan 2022)
- *RT-2: Vision-Language-Action Models* (Brohan 2023)
- *Open X-Embodiment: Robotic Learning Datasets and RT-X Models* (Padalkar 2023)

### Daily Tasks
1. 画一张 RT-1 → RT-2 → RT-X → OpenVLA 演化时间轴
2. 对每个模型填写：backbone / data scale / action repr / FPS / 公开权重
3. 写一段 500 字「为什么 RT-2 要把 action 嵌入到 LLM token 空间」
4. 跑 OpenVLA 在 BridgeData V2 的几条测试轨迹

### RT 家族对比表

| 模型 | 时间 | Backbone | Action 表示 | 数据 | 公开 |
|------|------|----------|------------|------|------|
| RT-1 | 2022 | EfficientNet + Transformer | 11-dim discrete | 130K (Google) | ❌ |
| RT-2 | 2023 | PaLI-X 55B | LLM token (256 bins) | + Web data | ❌ |
| RT-X | 2023 | RT-1/2 + 跨机数据 | 同上 | OXE 22 robots | ❌ |
| OpenVLA | 2024 | Llama2-7B + DINOv2/SigLIP | 256 bins token | OXE | ✅ |
| RT-2-X | 2023 | PaLI-X | token | OXE | ❌ |

### 核心思想线（背下来）
1. **RT-1**：证明 transformer 能吃多任务机器人数据
2. **RT-2**：把 action 当成 token 嵌入 LLM，**继承网络常识**
3. **RT-X / OXE**：跨机器人数据集成，证明 cross-embodiment 可行
4. **OpenVLA**：把 RT-2 思路开源化 + 蒸馏成 7B
5. **π0 / GR00T**：抛弃 discrete action，转向 continuous (Flow Matching / Diffusion)

---

## Day 37 — π0 / π0.5 (Physical Intelligence)

### 早晨论文 (3h)
- *π0: A Vision-Language-Action Flow Model for General Robot Control* (Black 2024)
- *π0.5: A Vision-Language-Action Model with Open-World Generalization* (2025)

### Daily Tasks
1. clone `Physical-Intelligence/openpi`
2. 跑 π0 在 LIBERO 上的 inference
3. 阅读 action expert 实现，理解 Flow Matching 怎么和 VLM 接上
4. 试着在自录数据上 LoRA 微调 π0

### π0 关键设计

```
   ┌─────────────────────────────────┐
   │  PaliGemma 3B (VLM, frozen)    │
   │  → 提取多模态 latent           │
   └────────────────┬────────────────┘
                    │ (text + image tokens)
                    │
   ┌────────────────▼────────────────┐
   │  Action Expert (3.4B)           │
   │  - Flow Matching                │
   │  - 50Hz × 50 chunk              │
   │  - 在 latent 上做 conditioning  │
   └────────────────┬────────────────┘
                    │
              [Continuous action chunk]
```

### 为什么 π0 比 OpenVLA 强？

| 维度 | OpenVLA | π0 |
|------|---------|-----|
| Action 表示 | discrete bins | continuous (Flow Matching) |
| 控制频率 | ~6Hz (autoregressive) | 50Hz (parallel chunk) |
| 动作精度 | 中（discretization 误差） | 高 |
| 训练数据 | OXE | OXE + 自采 1.5w 小时 |
| 训练成本 | 中 | 极高 |

### π0.5 增量

- 引入 **dual-system**：VLM 推 high-level reasoning（语言 plan），action expert 推 low-level
- 加入 web data co-training，开放词表
- 长程任务（"clean the kitchen"）能力突破

### Code Template (π0 Inference)

```python
# day37_pi0/run_pi0.py
from openpi.policies import policy_config
from openpi.training import config

cfg = config.get_config("pi0_libero")
policy = policy_config.create_trained_policy(cfg, checkpoint_dir="...")

obs = {
    "image": image,                  # [H, W, 3]
    "wrist_image": wrist,            # [H, W, 3]
    "state": joint_state,            # [8]
    "prompt": "pick up the cup",
}
action_chunk = policy.infer(obs)["actions"]  # [50, 8]
```

### Tuning Checklist
- [ ] 微调 π0 至少 8 张 H100，普通人玩不动 full FT，只能 LoRA
- [ ] LoRA 微调 4090 24GB 勉强可行，参考 openpi 的 LoRA 配置
- [ ] 控制频率匹配：训练 50Hz 推理也要 50Hz，否则动作 scale 不对

---

## Day 38 — GR00T N1 / N1.5 (NVIDIA)

### 早晨论文 (2.5h)
- *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots* (NVIDIA 2025)
- *GR00T N1.5* tech report (2025)
- 关键：**System1/System2 双系统**，并行人形机器人 specialist

### Daily Tasks
1. clone `NVIDIA/Isaac-GR00T`
2. 下载 GR00T N1.5 checkpoint
3. 跑 inference + 在 RoboCasa 任务做 few-shot eval
4. 阅读双脑架构源码（System2 = VLM, System1 = Diffusion Transformer）

### GR00T N1 架构

```
┌─────────────────────────────────────┐
│   System 2 — VLM (Eagle 2.5B)       │
│   - 慢推理 (~10Hz)                   │
│   - 语言 + 图像 → 高层 plan         │
└──────────────┬──────────────────────┘
               │ (latent embedding)
               │
┌──────────────▼──────────────────────┐
│   System 1 — DiT Action Expert      │
│   - Flow Matching                    │
│   - 快推理 (120Hz on H100)          │
│   - 输出 action chunk               │
└─────────────────────────────────────┘
```

### N1 vs N1.5 差异

| 维度 | N1 | N1.5 |
|------|-----|------|
| 数据 | 主要 sim + teleop | + neural trajectories（VLM 生成） |
| Embodiment | 双臂人形 | + 移动 + 单臂 |
| Few-shot | 100 条数据 | 30 条数据 |
| 推理速度 | 60Hz | 120Hz |

### 为什么是双系统？
- **System 2 慢**：语言推理不需要 100Hz
- **System 1 快**：动作必须高频，否则机器人抖
- **解耦**：可以独立优化各自模块（VLM 用 web data，Action expert 用机器人数据）

### Code Template (GR00T Inference)

```python
# day38_groot/run_groot.py
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP

policy = Gr00tPolicy(
    model_path="nvidia/GR00T-N1.5-3B",
    embodiment_tag="gr1",
    modality_config=DATA_CONFIG_MAP["gr1_arms_only"],
    device="cuda",
)
obs = {
    "video.ego_view": ego_image_seq,    # [T, H, W, 3]
    "state.left_arm": left_arm_state,
    "state.right_arm": right_arm_state,
    "annotation.human.task_description": "fold the towel",
}
action = policy.get_action(obs)
```

### Tuning Checklist
- [ ] embodiment_tag 必须对（gr1 / oxe / new_embodiment 等）
- [ ] modality_config 决定 obs 字段，错了直接报错
- [ ] 微调用 LeRobot 格式数据，参考官方 finetune 脚本

---

## Day 39 — RDT-1B / CogACT / TinyVLA

### 早晨论文 (2.5h)
- *RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation* (Liu 2024)
- *CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action* (Li 2024)

### Daily Tasks
1. clone `thu-ml/RoboticsDiffusionTransformer`
2. 跑 RDT-1B 在 ALOHA 双臂任务的 inference
3. 对比 RDT 和 π0 的差异
4. 浏览 CogACT 和 TinyVLA，理解小模型路线

### RDT-1B 关键设计

```
┌────────────────────┐
│  Image (3 views)   │ → SigLIP → patch tokens
│  Language          │ → T5 → text tokens
│  Joint state       │ → Linear → state token
└────────┬───────────┘
         │
   [concat, condition]
         │
┌────────▼───────────┐
│   DiT (1.2B)       │
│   - Diffusion      │
│   - Unified action │
│     space (128-d)  │
└────────┬───────────┘
         │
    [Action chunk]
```

**核心创新**：
- **Unified action space**：把所有 embodiment 的动作 padding 到 128 维统一
- **Multi-image support**：原生支持 3-view + 历史
- **DiT 架构**：而非 UNet，更适合 transformer 训练范式

### 三大 VLA 对比（背下来）

| 维度 | π0 | GR00T N1.5 | RDT-1B |
|------|-----|-----------|--------|
| 出品 | Physical Intelligence | NVIDIA | THU + 智元 |
| Backbone | PaliGemma 3B | Eagle 2.5B | T5 + SigLIP |
| Action Head | Flow Matching | DiT + Flow Matching | DiT + DDPM |
| 双臂 | ✅ | ✅ | ✅（原生强） |
| 公开权重 | ✅ (lite) | ✅ | ✅ |
| 主要数据 | 自采 1.5w 小时 | sim + teleop + neural | 自采 + OXE |

### Code Template (RDT inference snippet)

```python
# day39_rdt_cogact/rdt_infer.py
from scripts.agilex_inference import RoboticDiffusionTransformerModel

model = RoboticDiffusionTransformerModel(
    pretrained="robotics-diffusion-transformer/rdt-1b",
    ...
)
action_chunk = model.step(image_seq, instruction, joint_state)  # [64, 14]
```

---

## Day 40 — Helix / Hierarchical & Long-Horizon

### 早晨阅读 (2h)
- *Helix* (Figure 2025, blog post + tech report)
- *Hi-Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models* (PI 2025)

### Daily Tasks
1. 阅读 Helix 设计：dual-system humanoid VLA
2. 整理 hierarchical VLA 的三种范式：
   - 双脑（Helix / GR00T N1.5）
   - 调用 specialist（Hi-Robot / SayCan 风）
   - 融合 chunk（π0.5）
3. 写一段「为什么具身 VLA 必须分层」的论述
4. 思考你自己的 long-horizon 任务设计

### Helix 关键架构

```
┌────────────────────────┐
│  System 2 (VLM, 7B)    │  7-9Hz, 通用 reasoning
│  → latent vector       │
└──────────┬─────────────┘
           │
┌──────────▼─────────────┐
│  System 1 (80M params) │  200Hz on robot
│  → 35-DoF whole-body   │
│    continuous action   │
└────────────────────────┘
```

### Hierarchical VLA 三种范式

| 范式 | 代表 | 优点 | 缺点 |
|------|------|------|------|
| 双脑 (S2 → S1) | Helix, GR00T N1.5 | 端到端 latent | 训练复杂 |
| Skill Library | SayCan, Hi-Robot | 可解释、可复用 | 受限于 skill set |
| Chunk hierarchy | π0.5 | 简单 | 长时序仍弱 |

---

## Day 41 — LIBERO Benchmark 微调

### Daily Tasks
1. 下载 LIBERO 数据（goal / object / spatial / 10）
2. 把 OpenVLA 在 LIBERO-spatial 微调（已有官方脚本）
3. 评估 4 个 suite（spatial / object / goal / 10）
4. 报数：你的成功率 vs 论文报数

### LIBERO 微调命令

```bash
# 数据准备
python finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir <datasets_path> \
    --dataset_name libero_spatial_no_noops \
    --run_root_dir <log_dir> \
    --use_lora True \
    --lora_rank 32 \
    --batch_size 16 \
    --grad_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --image_aug True \
    --wandb_project openvla \
    --save_steps 5000
```

### 期望结果（参考公开数字）

| Suite | OpenVLA | π0 | RDT | 你的目标 |
|-------|---------|-----|-----|---------|
| spatial | 84.7% | 96.4% | - | ≥ 80% |
| object | 88.4% | 96.8% | - | ≥ 85% |
| goal | 79.2% | 95.2% | - | ≥ 75% |
| 10 | 53.7% | 85.8% | - | ≥ 50% |

### Tuning Checklist
- [ ] LoRA rank 32，batch 16，4090 OK
- [ ] 微调 30k step 到 50k step
- [ ] 失败案例：长时序任务（goal / 10）会差
- [ ] image augmentation 开 +5%

---

## Day 42 — 周报 + VLA 全景图

### Daily Tasks
1. 整理一张大图：「2022-2026 VLA 演化谱系」
2. 写一份「VLA 工程师面试速答清单」
3. 录一段 5-10 分钟的视频，自己讲 π0 / GR00T / RDT 区别
4. 准备 Week 7 项目：选择 manip / locomotion / VLA 一条线

### 自检：你应该能回答（10 分钟内全部答出来）
1. Why discrete → continuous action？
2. Why Flow Matching > DDPM in robotics？
3. System1/System2 解决了什么问题？
4. Cross-embodiment 主要难点是什么？OXE 怎么解决？
5. 给你 10 张 H100，你怎么训自己的 VLA？
6. 给你 1 张 4090，你怎么微调 SOTA VLA？
7. VLA 在真机上的瓶颈是什么？（推理速度 / 数据 / sim2real）
8. 列 3 个目前 VLA 没解决的问题

---

## 「VLA 全景图」（务必内化）

```
2022 ─── RT-1                 (action token化的开端)
  │
2023 ─┬── RT-2 / RT-X         (LLM 知识 + cross-embodiment)
      │
      └── OpenVLA              (开源民主化)
  │
2024 ─┬── π0                   (Flow Matching, 高频)
      ├── RDT-1B               (DiT, 双臂, unified action)
      ├── CogACT               (cognition + action 解耦)
      ├── TinyVLA              (小模型路线)
      └── 3D Diffuser Actor    (3D 表征)
  │
2025 ─┬── π0.5                 (open-world)
      ├── GR00T N1 / N1.5      (NV，人形 specialist)
      ├── Helix (Figure)       (双脑 humanoid)
      ├── Hi-Robot (PI)        (hierarchical)
      └── SpatialVLA           (3D 空间感)
  │
2026 ─── （你 watch 的方向）   (Test-time compute / Long horizon /
                                 Whole-body humanoid loco-manip)
```

---

## 本周总投入预估

| 活动 | 小时 |
|------|------|
| 论文阅读 | 18h |
| 代码（4 个 SOTA 跑通） | 22h |
| LIBERO 微调 | 10h |
| 周报 / 视频 | 5h |
| **合计** | **55h** |

→ [Week07.md](./Week07.md)
