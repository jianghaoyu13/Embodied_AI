# Week 4 — 机器人动力学 + 7-DoF 机械臂 / 双臂 RL + 抓取 + 移动操作

> **本周定位**（已为你校准）：你的目标是**轮式双臂人形 + 7-DoF 机械臂**的操作 / 抓取 / 移动操作。
> 所以 locomotion 不在本周内容内，本周聚焦：**机器人动力学补强 + 单臂/双臂 RL + 抓取 + 轮式底盘移动操作**。

---

## 本周目标

完成本周后，你应该能：
- [ ] 用 Pinocchio / cuRobo 推一个 7-DoF 机械臂的正/逆动力学，理解 M, C, G 矩阵在控制中的作用
- [ ] 在 Isaac Lab 训出一个 Franka 7-DoF arm 的 PPO 策略（reach / pick）
- [ ] 跑通一个双臂协作（ALOHA / Aloha-Sim）任务
- [ ] 跑通一个**轮式双臂人形**或近似平台（Galaxea R1 / 自定义 mobile manipulator）的移动抓取任务
- [ ] 解释抓取算法两条主流路线：analytic (GraspNet / contact-graspnet) 与 learning-based

---

## 本周可交付物

```
week4-arm-mobile/
├── day22_dynamics/        # 7-DoF 动力学手推 + Pinocchio 验证
├── day23_franka_rl/       # Franka Reach / Lift PPO
├── day24_grasping/        # GraspNet / Contact-GraspNet 推理
├── day25_dual_arm/        # ALOHA bimanual 任务
├── day26_mobile_manip/    # 轮式底盘 + 双臂 移动抓取
├── day27_dr_residual/     # DR + Residual RL（基于 cuRobo / OSC）
└── day28_recap/
```

---

## Day 22 — 机器人动力学补强（你点名要补的部分）

### 早晨阅读 (2.5h)
- *Modern Robotics* (Lynch & Park) 第 8 章（Dynamics of Open Chains）
- *A Mathematical Introduction to Robotic Manipulation* (Murray) 第 4 章（节选）
- Pinocchio 文档：`rnea`, `crba`, `aba` 三个核心 API

### 你必须搞清楚的 4 个公式

```
1. 拉格朗日方程：
   M(q) q̈ + C(q, q̇) q̇ + G(q) = τ + J(q)ᵀ F_ext

2. RNEA（递归牛顿欧拉，O(n)）：
   τ = RNEA(q, q̇, q̈)            ← 逆动力学

3. CRBA（合成刚体算法，O(n²)）：
   M(q) = CRBA(q)                 ← 惯性矩阵

4. ABA（铰接体算法，O(n)）：
   q̈ = ABA(q, q̇, τ)              ← 正动力学
```

### Daily Tasks
1. 安装 `pin`(`pip install pin`),加载 Franka Panda URDF
2. 用 Pinocchio 计算并打印 7-DoF 的 M, C, G
3. 手算一个 2-DoF planar arm 的动力学公式,用 sympy 验证
4. 对比「pure kinematics 控制(你 cuRobo 熟)」与「dynamics-aware 控制(CT / OSC)」差异

> **环境提示**:Day 22 全部在 **本地 `[embai]$`** 跑 —— Pinocchio 是 CPU 计算,纯 Python,与 Isaac Sim 容器无关。
> ```bash
> [embai]$ pip install pin sympy
> [embai]$ python day22_dynamics/franka_dyn.py
> ```

### Code Template — Pinocchio + Franka 7-DoF

```python
# day22_dynamics/franka_dyn.py
import pinocchio as pin
import numpy as np

URDF = "franka_panda.urdf"   # 你可以从 isaaclab assets 中找到
model = pin.buildModelFromUrdf(URDF)
data = model.createData()

q  = pin.neutral(model)            # [nq]
v  = np.zeros(model.nv)            # [nv]
a  = np.zeros(model.nv)
tau_ext = np.zeros(model.nv)

# 1) 逆动力学：给定 q, q̇, q̈ 求 τ
tau = pin.rnea(model, data, q, v, a)
print("RNEA τ:", tau)

# 2) 惯性矩阵
M = pin.crba(model, data, q)
print("M shape:", M.shape, "should be 7x7")

# 3) 重力项
G = pin.computeGeneralizedGravity(model, data, q)
print("G:", G)

# 4) 正动力学：给定 q, q̇, τ 求 q̈
qdd = pin.aba(model, data, q, v, tau)
print("ABA q̈:", qdd)

# 5) Jacobian（你已熟悉）+ Operational Space 投影
pin.computeJointJacobians(model, data, q)
J_ee = pin.getJointJacobian(model, data, model.njoints-1, pin.LOCAL_WORLD_ALIGNED)
# OSC: F = Λ ẍ_des + μ ẋ + p,  Λ = (J M⁻¹ Jᵀ)⁻¹
Minv = np.linalg.inv(M)
Lambda = np.linalg.inv(J_ee @ Minv @ J_ee.T + 1e-6 * np.eye(6))
print("Operational space inertia Λ:", Lambda)
```

### 自检（口头答出来）
- M、C、G 各代表什么物理量？为什么 C 是矩阵不是向量？
- 控制律 `τ = M(q)(K_p e + K_d ė) + C q̇ + G`，三项分别在补偿什么？
- 为何 OSC（Operational Space Control）比 joint space PD 更适合抓取任务？
- impedance control 和 admittance control 区别？什么时候用哪个？

### Tuning Checklist
- [ ] Pinocchio 的 `q` 维度可能 ≠ `nv`（floating base 时差 1）
- [ ] URDF 必须有 inertia 字段，否则 M 算出来全 0
- [ ] cuRobo 的 jacobian 约定（spatial / body / world-aligned）记清，和 Pinocchio 对比

### 加分(你装了 ROS2 Jazzy + Isaac Sim 6.0 容器以 `--network=host` 启动,host ↔ container ROS2 直连)

把 Pinocchio 的动力学计算用 ROS2 节点暴露出来,作为「真机就能直接接的接口」预演。**这个节点跑在宿主机 `[embai]$`**,Isaac Sim 在容器里通过 ros2_bridge publish `/joint_states`,host 节点订阅算重力补偿再 publish 回去 —— 跟真机部署拓扑完全一致:

```python
# day22_dynamics/ros2_dyn_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import pinocchio as pin
import numpy as np

class FrankaDynNode(Node):
    def __init__(self):
        super().__init__("franka_dyn")
        self.model = pin.buildModelFromUrdf("franka_panda.urdf")
        self.data  = self.model.createData()
        self.sub_q  = self.create_subscription(JointState, "/joint_states", self.cb, 10)
        self.pub_g  = self.create_publisher(Float64MultiArray, "/gravity_compensation", 10)
    def cb(self, msg):
        q = np.array(msg.position[:self.model.nq])
        G = pin.computeGeneralizedGravity(self.model, self.data, q)
        out = Float64MultiArray(); out.data = G.tolist()
        self.pub_g.publish(out)

rclpy.init(); rclpy.spin(FrankaDynNode())
```

```bash
# 终端 A:容器里启 Isaac Sim,开 ros2_bridge,publish /joint_states
[host]$ ~/isaac_workspace/run_isaac6.sh
[isaac6]$ ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --enable_ros2

# 终端 B:本地 embai 跑 Pinocchio ROS2 节点
[embai]$ source /opt/ros/jazzy/setup.bash
[embai]$ python day22_dynamics/ros2_dyn_node.py &
[embai]$ ros2 topic echo /gravity_compensation
```

这套节点 Week 7 真机部署可直接复用(把 `[isaac6]` 的 Isaac Sim 换成真机驱动节点就行)。

---

## Day 23 — Franka 7-DoF 单臂 RL（reach / lift）

### 早晨论文 (1.5h)
- *Operational Space Control: A Theoretical and Empirical Comparison* (Nakanishi 2008)
- 浏览 Isaac Lab `manipulation` 任务源码：`Isaac-Lift-Cube-Franka-v0`、`Isaac-Reach-Franka-v0`

### Daily Tasks
1. 在 Isaac Lab 跑通 `Isaac-Reach-Franka-v0`（PPO，约 20 分钟收敛）
2. 跑通 `Isaac-Lift-Cube-Franka-v0`
3. 把 action space 切换：joint position / joint velocity / OSC 三种，对比训练曲线
4. 切换 controller：默认 / cuRobo MPC（用 cuRobo 作为 differentiable controller，可选）

### Action Space 选择对比表（你要会画）

| Action space | 学习难度 | 真机部署 | 推荐场景 |
|--------------|---------|---------|---------|
| Joint position (delta) | 低 | 易 | RL 默认 |
| Joint velocity | 中 | 中 | dynamic 任务 |
| Joint torque | 高 | 难 | research |
| OSC (EE pose delta) | 低 | 易 | manipulation 首选 |
| OSC + impedance | 中 | 中 | 接触任务 |
| Diff IK / cuRobo MPC | 中 | 易 | 几何约束多 |

### 训练命令

> 全部在 **6.0 容器内** 跑(`[host]$ ~/isaac_workspace/run_isaac6.sh` 进容器):

```bash
[isaac6]$ cd /workspace/IsaacLab
[isaac6]$ ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Cube-Franka-v0 --headless --num_envs 4096

[isaac6]$ ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Lift-Cube-Franka-v0 --num_envs 16
```

### Reward 设计要点（manipulation 通用）

```
+ ee_to_object_dist            # 末端接近物体
+ object_to_target_dist        # 物体接近目标
- action_rate / joint_vel      # 动作平滑
- action_torque                # 节能、保护硬件
+ object_lifted_bonus          # 抬起 +
+ task_complete_bonus          # 完成 +
- collision (with table/self)  # 撞到 -
```

### Tuning Checklist
- [ ] OSC action space 在 Lift 任务上比 joint pos 快约 30% 收敛
- [ ] 加 `joint_vel_l2` 惩罚是抖动救星
- [ ] reward 的 ee-to-object 用 `1 - tanh(dist/0.1)` 比 `-dist` 稳定

### Common Pitfalls
- 抓取时 reward 容易 hack：策略把物体推到目标，而不是抓
  → 加 `gripper_grasping` reward + 必须接触检测
- gripper 没 reward 时一直张开：加 `gripper_close_when_near_object` shaping

---

## Day 24 — 抓取算法（GraspNet / Contact-GraspNet / AnyGrasp）

> 你做轮式双臂操作，**抓取算法**是必修。这天打通从点云到 6-DoF 抓取姿态的 pipeline。

### 早晨论文 (2h)
- *GraspNet-1Billion* (Fang 2020)
- *Contact-GraspNet* (Sundermeyer 2021)
- *AnyGrasp* (Fang 2023)

### 抓取算法两条路线

| 路线 | 代表 | 输入 | 输出 |
|------|------|------|------|
| Analytic (planning) | DexNet / FC-GQ-CNN | depth | grasp quality map |
| Learning + 6-DoF | GraspNet / Contact-GraspNet / AnyGrasp | RGB-D 或 PCD | 6-DoF grasp poses + score |

### Daily Tasks
1. 跑通 Contact-GraspNet:输入桌面点云,输出候选 grasp poses
2. 在 Isaac Lab 中:先用 Contact-GraspNet 选 pose,再用 cuRobo 做 motion planning,串成「感知 → 抓取规划 → 执行」demo
3. 对比「直接 RL 学抓取」 vs 「GraspNet + cuRobo 模块化」的成功率

> **环境提示**:
> - Contact-GraspNet 推理 + cuRobo 规划 → 本地 `[embai]$`(纯 PyTorch + GPU)
> - Isaac Sim 仿真环境 → 容器 `[isaac6]$`(headless 跑)
> - 两边通过 ROS2 topic 串通(同 Day 22 拓扑):容器 publish `/depth_image` + `/camera_info`,本地节点订阅 → 推理 → publish `/grasp_pose`,容器订阅后用 Isaac Lab 控制器执行

### Code Template — 端到端抓取 pipeline 骨架

```python
# day24_grasping/pick_pipeline.py
import numpy as np
from contact_graspnet_pytorch import inference as cgn

# 1) 拿点云（Isaac Sim 的 camera + depth → numpy）
pcd_xyz = depth_to_pointcloud(depth_image, K, T_cam2world)  # [N, 3]

# 2) Contact-GraspNet 给候选
grasps, scores = cgn.predict(pcd_xyz)            # grasps: [K, 4, 4], scores: [K]
best = grasps[np.argmax(scores)]                  # 6-DoF SE(3)

# 3) cuRobo 做规划（你熟）
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
mg = MotionGen(MotionGenConfig.load_from_robot_config("franka.yml"))
plan = mg.plan_single(start_state, goal_pose=best)

# 4) 执行 + 闭爪
for q in plan.position:
    env.set_joint_target(q)
env.close_gripper()
```

### 模块化 vs 端到端 — 何时选哪个

| 维度 | GraspNet + cuRobo（模块化） | RL/IL 端到端 |
|------|----------------------------|--------------|
| 数据需求 | 极少（用预训练） | 大 |
| 调试 | 每个模块独立 | 黑盒 |
| 真机迁移 | 强（基于几何） | 看 DR |
| 长程任务 | 难（需要 task planner） | 强（VLA 风） |
| 推荐入门 | ✅ | ⚠️ |

### Tuning Checklist
- [ ] 点云质量直接决定上限：去地面、去边缘 noise
- [ ] grasp pose 的 approach direction 要和 EE 朝向匹配
- [ ] cuRobo 规划时 IK seed 多给几个，否则单解失败率高

---

## Day 25 — 双臂协作（ALOHA / Bimanual）

### 早晨论文 (2h)
- *Mobile ALOHA* (Fu 2024)
- *AnyRotate / TwoArm RoboGymnastics* 等双臂 benchmark

### Daily Tasks
1. clone `tonyzhaozh/aloha`（仿真版）
2. 跑通 `aloha_sim_transfer_cube` 和 `aloha_sim_insertion`（双臂协作）
3. 用 ACT（Week 2 已掌握）训练，成功率 ≥ 80%
4. 加 DP（Week 3）替换 ACT，对比

### 双臂任务的两个核心难点

| 难点 | 解决思路 |
|------|---------|
| **协调性（coordination）** | 共享 latent / 共享 cross-attention |
| **干涉（collision）** | reward 加 self-collision penalty / cuRobo collision check |

### 训练命令

> ACT 训练是纯 PyTorch + ALOHA 自带 MuJoCo 仿真,不依赖 Isaac Sim,**在本地 `[embai]$` 跑**:

```bash
# ACT(你 Week 2 已用)
[embai]$ python imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir ./ckpts/transfer_cube_act \
    --policy_class ACT --kl_weight 10 \
    --chunk_size 100 --hidden_dim 512 \
    --batch_size 8 --lr 1e-5 --seed 0 \
    --num_epochs 2000
```

### Tuning Checklist
- [ ] 双臂数据建议 50+ 条 demo（比单臂多）
- [ ] action 维度变成 14（左 7 + 右 7），归一化要分别做
- [ ] 双臂任务 chunk_size 倾向 **更大**（150-200），因为协调动作时长更长

---

## Day 26 — 轮式双臂人形 / 移动操作

### 早晨论文 (2h)
- *Mobile ALOHA* (Fu 2024) — ALOHA + 移动底盘
- *RoboCasa* paper — 厨房 mobile manipulation 仿真
- *Galaxea R1 / Open R1* tech report（如可获得）
- *PerAct² / Hi-Robot* 中关于 mobile manipulation 的章节

### Daily Tasks
1. 在 Isaac Lab 用 `Isaac-Open-Drawer-Franka-v0` / `Isaac-Cabinet-Franka-v0` 之类**接近移动操作**的任务热身
2. clone `robocasa/robocasa`,跑通 1-2 个 kitchen mobile manipulation 任务
3. 阅读 Galaxea / 自变量 / 银河通用公开的轮式双臂资料,理解硬件抽象
4. 思考你的 Week 7 项目:在仿真中搭建一个轮式双臂平台

> **环境提示**:
> - Isaac Lab Open-Drawer / 自建轮式双臂训练 → 容器 `[isaac6]$`(`./isaaclab.sh -p ...`)
> - RoboCasa 基于 RoboSuite + MuJoCo,**不进 Isaac 容器**,直接本地 `[embai]$`:
>   ```bash
>   [embai]$ pip install robocasa robosuite
>   [embai]$ python -c "import robocasa; from robosuite import make; env=make('PnPCounterToCab', robots='PandaMobile'); env.reset()"
>   ```

### 轮式双臂人形的 action 抽象

```
action = [base_v_x, base_v_y, base_w_z,    # 底盘 3-DoF (差速 / 全向)
          left_arm_q (7),                   # 左臂
          right_arm_q (7),                  # 右臂
          left_gripper, right_gripper,      # 夹爪
          torso_q (1-3)]                    # 升降 / 俯仰
```

### 移动操作的核心难点

| 难点 | 表现 | 解决思路 |
|------|------|---------|
| **底盘 + 臂耦合** | 走着走着臂晃 | 加 EE 在 world frame 的 stability reward |
| **视觉随底盘移动** | obs 不稳 | 多 camera：base + wrist |
| **任务长程** | reward 稀疏 | curriculum / VLA 风 high-level planner |
| **平台多样性** | 每个机器人都不一样 | 用 unified action space (RDT 思路) |

### Code Template — 在 Isaac Lab 自定义一个轮式双臂

```python
# day26_mobile_manip/wheeled_bimanual_cfg.py
# 思路：拼一个简化版 = 差速底盘 + 两个 7-DoF arm
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg

@configclass
class MobileBimanualSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path="...your_robot.usd"),
        actuators={
            "base_wheels": ImplicitActuatorCfg(joint_names_expr=["wheel_.*"], stiffness=0.0, damping=200.0),
            "left_arm":    ImplicitActuatorCfg(joint_names_expr=["left_panda_joint.*"], stiffness=400, damping=80),
            "right_arm":   ImplicitActuatorCfg(joint_names_expr=["right_panda_joint.*"], stiffness=400, damping=80),
            "grippers":    ImplicitActuatorCfg(joint_names_expr=[".*finger.*"], stiffness=2e3, damping=1e2),
        },
    )
    table = ...  # 桌子 + 物体
```

### Tuning Checklist
- [ ] 底盘 reward 一定要和臂的 reward 数量级对齐，否则只学走 / 只学抓
- [ ] obs 中务必包含 base 的 odom + EE world-frame pose
- [ ] action 的 base 速度上限要小（0.5 m/s），否则训练抖
- [ ] 训练时先固定 base、训臂；再放开 base，curriculum learning

---

## Day 27 — DR + Residual RL（基于 cuRobo / OSC 残差）

> 这天利用你 **cuRobo 已经熟练** 的优势：让 RL 不从零学，而是学一个 「cuRobo 输出 + 残差修正」。

### 早晨论文 (1.5h)
- *Residual Reinforcement Learning for Robot Control* (Johannink 2018)
- *Guided Policy Search* (Levine 2013，老但思想重要)

### Daily Tasks
1. 在 Lift Cube 任务上:先让 cuRobo 给 nominal trajectory
2. RL 只学 `Δq` 残差,加在 cuRobo 解上
3. 对比纯 RL vs Residual RL 的样本效率
4. 加 DR(摩擦、质量、视觉),训鲁棒策略

> **环境提示**:Residual wrapper 跨两套环境 ——
> - cuRobo MPC nominal 解 → 本地 `[embai]$`(你已熟练的 cuRobo Python API,跑在 GPU)
> - Isaac Lab 仿真 + RL 训练 → 容器 `[isaac6]$`
> - 两边走 ROS2 桥(`/cuRobo_nominal_q` ← host publish, `/joint_state` → host subscribe),沿用 Day 22 拓扑
> - **如果想避免 ROS2 复杂度**:把 cuRobo 也装到容器内,`ResidualWrapper` 直接 in-process 调用,简单得多。Week 4 推荐先走 in-container 方案。

### Residual Action 公式

```
a_total = a_cuRobo(s) + α · a_residual_RL(s),  α∈[0, 0.3]
```

### Code Template — Residual Wrapper

```python
# day27_dr_residual/residual_wrapper.py
class ResidualEnv(gym.Wrapper):
    def __init__(self, env, base_controller, alpha=0.2):
        super().__init__(env)
        self.base = base_controller     # 比如 cuRobo MPC
        self.alpha = alpha
    def step(self, residual_a):
        nominal = self.base.compute(self.unwrapped.get_state())
        a_total = nominal + self.alpha * residual_a
        return self.env.step(a_total)
```

### DR 关键维度（manipulation 版）

| 维度 | 范围 | 备注 |
|------|------|------|
| 物体质量 | ±50% | 抓取力学敏感 |
| 物体摩擦 | [0.5, 1.5] | 滑/粘 |
| 物体尺寸 | ±20% | 适配多物体 |
| 相机外参 | ±5cm / ±5° | 真机标定误差 |
| 物体初始 pose | 桌面 ±0.2m | 泛化关键 |
| 控制延迟 | 0-50ms | 真机部署 |

### Tuning Checklist
- [ ] α 太大会让 RL 学坏 cuRobo 的好动作；太小学不到残差
- [ ] DR 范围一定要在 cuRobo 仍能给合理解的范围内
- [ ] residual 的 action norm 加 L2 penalty，鼓励它「少做事」

---

## Day 28 — 周报 + 移动操作 cheatsheet

### Daily Tasks
1. 写 weekly recap（800 字）：动力学补强收获 / RL 调参直觉 / 移动操作难点
2. 整理「轮式双臂 / 7-DoF 机械臂操作 12 条军规」（贴显示器）
3. 画一张「操作算法 stack」图（感知 → 抓取 → 规划 → 控制）
4. 准备 Week 5 的 VLM 学习

### 自检：你应该能回答
1. M, C, G 各是什么？OSC 控制律每项含义？
2. 为什么 manipulation RL 倾向 SAC，但 Isaac Lab 大并行下 PPO 更香？
3. GraspNet + cuRobo vs 端到端 RL，分别什么场景选？
4. 双臂任务 chunk size 为什么要更大？
5. 移动操作的底盘和臂耦合怎么处理？
6. Residual RL 比纯 RL 好在哪？为什么不一直用？

---

## 「轮式双臂 / 7-DoF 操作 12 条军规」（建议背诵）

1. 默认用 **OSC delta EE** 当 action space，比 joint pos 快 + 易部署
2. 抓取 reward 要分阶段：approach → grasp → lift → place
3. gripper 要单独 reward，否则学不会闭合
4. 双臂任务 chunk_size > 单臂（150-200 vs 50-100）
5. 移动操作 base 速度上限要低（0.3-0.5 m/s）
6. base + arm 一起训前先固定 base 训臂（curriculum）
7. wrist camera 比 base camera 对操作更关键
8. cuRobo 已有的能力别让 RL 重新学，用 residual
9. DR 的物体质量是抓取最关键维度
10. self-collision 必须显式 penalty（双臂尤其）
11. 真机延迟 30-50ms 是标准，训练就要加
12. 操作任务的 success 条件写严格（位置 + 朝向 + 速度都达标）

---

## 本周总投入预估

| 活动 | 小时 |
|------|------|
| 论文阅读 | 12h |
| 代码 + 动力学手推 | 24h |
| 训练等待 | 12h |
| 周报 | 5h |
| **合计** | **53h** |

→ [Week05.md](./Week05.md)
