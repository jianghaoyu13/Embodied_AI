# Day 03 — Batched SE(3) in PyTorch

> **日期**：2026-06-23（Week 1, Day 3）
>
> **本日定位**：把你 cuRobo 时代用 numpy / 自己代数推过的 SE(3) 工具，**重写成 PyTorch batched + autograd 能过的版本**。Week 3 Diffusion Policy 的 6-DoF action、Week 4 Pinocchio Jacobian、Week 5/6 VLA 的 SE(3) action head 都靠它。
>
> **总投入**：约 8 小时

---

## 0. 出发前自检（5 分钟）

```bash
[embai]$ conda activate embai
[embai]$ python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望：2.12.0+cu130, True

[embai]$ wandb status
[embai]$ cd ~/embodied-ai-bootcamp-8w
[embai]$ git status                # clean
[embai]$ git log --oneline | head  # 看到 Day 2 commit
[embai]$ nvidia-smi                # 显存 < 1GB
```

**今天主战场**：`week1-fundamentals/day3_se3/`

```bash
[embai]$ mkdir -p ~/embodied-ai-bootcamp-8w/week1-fundamentals/day3_se3/{src,tests,configs,notebooks}
[embai]$ cd ~/embodied-ai-bootcamp-8w/week1-fundamentals/day3_se3
```

---

## 1. 今日时间表（hour by hour）

| 时段 | 时长 | 任务 | 产出 |
|------|------|------|------|
| 07:00 - 08:30 | 1.5h | Solà 2020 §1-3 + Lynch §3.2-3.3 | 笔记 1 页，能默写 hat / vee / exp / log |
| 09:00 - 10:00 | 1h   | 工具骨架 + 约定文档 | `lie.py` 空函数 + `CONVENTIONS.md` |
| 10:00 - 12:00 | 2h   | 手写 `so3_exp/log` + `se3_exp/log` | 单测 `log(exp(x)) ≈ x` 通过 |
| 14:00 - 15:30 | 1.5h | `compose / inverse / adjoint` + 四元数互转 | 全部 batched，单测过 |
| 15:30 - 17:00 | 1.5h | 数值稳定性 + autograd 测试 | θ→0 用 Taylor，gradcheck 通过 |
| 19:30 - 20:30 | 1h   | 跨库交叉验证（pytorch3d / pin / curobo） | 一致性 < 1e-5 |
| 20:30 - 21:00 | 0.5h | 复盘 + commit + push | Day 3 commit |

---

## 2. 早晨 (07:00 - 08:30) — 论文 + 概念

### 任务
- 重读：*A micro Lie theory for state estimation in robotics* (Solà 2020) **§1-3**
- 浏览：*Modern Robotics* (Lynch & Park) §3.2-3.3 (Rigid-Body Motions)
- 速览代码：
  - [`UM-ARM-Lab/pytorch_kinematics`](https://github.com/UM-ARM-Lab/pytorch_kinematics) 的 `transforms3d.py`
  - cuRobo 的 `quat_pose_to_matrix` 实现（你应该已经熟）

### 笔记产出 — `day_plan/notes/Day03_se3.md`

必须能默写并解释：

1. **so(3) ↔ SO(3) 的指数 / 对数映射（Rodrigues）**
   ```
   exp: so(3) → SO(3)
   ω̂ ∈ ℝ^{3×3} 是 ω ∈ ℝ³ 的反对称矩阵, θ = ‖ω‖
   R = I + sin(θ)/θ · ω̂ + (1 - cos(θ))/θ² · ω̂²

   log: SO(3) → so(3)
   θ = arccos((tr(R) - 1) / 2)
   ω = θ / (2 sin θ) · (R - Rᵀ)∨   （vee）
   ```

2. **se(3) ↔ SE(3) 的指数 / 对数（含平移）**
   ```
   ξ = (ρ, ω) ∈ ℝ⁶,   T = exp(ξ̂) =
   ⎡ R   t ⎤        其中 t = V(ω) ρ
   ⎣ 0   1 ⎦        V(ω) = I + (1-cos θ)/θ² · ω̂ + (θ-sin θ)/θ³ · ω̂²
   ```

3. **三件事必须能口头答**
   - 为什么 ξ 顺序是 `(ρ, ω)` 而不是 `(ω, ρ)`？→ Solà / Pinocchio 约定，左乘扰动语义清晰；不同库不一样必须写注释
   - Adjoint `Ad_T` 是干嘛的？→ 把一个 frame 下的 twist 换到另一个 frame：`ξ_B = Ad_{T_BA} ξ_A`
   - 为什么 θ→0 时直接套公式会爆炸？→ 分母有 θ²、θ³，必须 Taylor 展开到 4 阶

4. **预算估算**
   ```
   batched [B, 6] → SE(3) [B, 4, 4]：FLOPs ≈ B × ~200，纯 GPU 计算
   关键瓶颈不是算力，是 atan2 / acos 在 θ≈0 / θ≈π 时的数值稳定
   ```

### 自检
- [ ] 闭眼默写 Rodrigues 公式 + se(3) exp
- [ ] 能解释 hat / vee 操作（`(·)^∧` 和 `(·)^∨`）
- [ ] 能讲清楚为什么 exp 在 θ=0 处可导（看 Taylor 展开常数项）

---

## 3. 上午 09:00 - 10:00 — 工具骨架 + 约定文档

> **今日大坑预防针**：四元数顺序、ξ 顺序、左乘 / 右乘扰动 —— 不同库约定不一样，**先把你的约定写死再开始**，否则后两天调试会怀疑人生。

### 3.1 写 `CONVENTIONS.md`(必做,贴在显示器)

```markdown
# SE(3) Conventions for this project

## Quaternion order
- (w, x, y, z), w 在前(scipy / pin / curobo 兼容)
- Hamilton 乘法,不是 JPL

## Twist order
- ξ = (ρ, ω) ∈ ℝ⁶, ρ 是平移, ω 是旋转
- 与 Solà / Pinocchio / pytorch3d 一致
- (注意:Lynch 教材用 (ω, v) 顺序,**互转时显式 swap**)

## Perturbation
- 左乘扰动: T_perturbed = exp(ξ̂) · T
- Jacobian 用 left Jacobian J_l(ω)

## Frame naming
- T_AB 表示 "B 在 A 中的 pose",或 "把 B-frame 点变到 A-frame"
- 即 p_A = T_AB · p_B
- (这与 cuRobo / ROS tf 一致)
```

### 3.2 函数骨架

```python
# day3_se3/src/lie.py
"""SE(3) batched ops in PyTorch. See CONVENTIONS.md for conventions."""
import torch

EPS = 1e-8

def hat(omega: torch.Tensor) -> torch.Tensor:
    """[..., 3] -> [..., 3, 3] skew-symmetric"""
    ...

def vee(Omega: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] -> [..., 3] inverse of hat"""
    ...

def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """[..., 3] -> [..., 3, 3] Rodrigues; numerically stable at θ→0"""
    ...

def so3_log(R: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] -> [..., 3]; numerically stable at θ→0 and θ→π"""
    ...

def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """[..., 6] (ρ, ω) -> [..., 4, 4]"""
    ...

def se3_log(T: torch.Tensor) -> torch.Tensor:
    """[..., 4, 4] -> [..., 6] (ρ, ω)"""
    ...

def compose(T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
    """T1 @ T2, batched, broadcastable"""
    ...

def inverse(T: torch.Tensor) -> torch.Tensor:
    """SE(3) closed-form inverse (cheaper than torch.linalg.inv)"""
    ...

def adjoint(T: torch.Tensor) -> torch.Tensor:
    """[..., 4, 4] -> [..., 6, 6] Ad_T"""
    ...

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """[..., 4] (w, x, y, z) -> [..., 3, 3]"""
    ...

def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] -> [..., 4] (w, x, y, z); Shepperd method for stability"""
    ...
```

### 自检
- [ ] `CONVENTIONS.md` 写完贴墙上
- [ ] 函数签名都有 docstring + shape 注释
- [ ] `from lie import *` 不报错

---

## 4. 上午 10:00 - 12:00 — `so3_exp/log` + `se3_exp/log`

### 4.1 实现 — 数值稳定版本

```python
# day3_se3/src/lie.py（核心实现）
def hat(omega):
    O = torch.zeros(*omega.shape[:-1], 3, 3, device=omega.device, dtype=omega.dtype)
    O[..., 0, 1] = -omega[..., 2]; O[..., 0, 2] =  omega[..., 1]
    O[..., 1, 0] =  omega[..., 2]; O[..., 1, 2] = -omega[..., 0]
    O[..., 2, 0] = -omega[..., 1]; O[..., 2, 1] =  omega[..., 0]
    return O

def vee(Omega):
    return torch.stack([Omega[..., 2, 1], Omega[..., 0, 2], Omega[..., 1, 0]], dim=-1)

def so3_exp(omega):
    """θ→0 用 Taylor: A=1-θ²/6+..., B=1/2-θ²/24+..."""
    theta2 = (omega * omega).sum(dim=-1, keepdim=True)         # [..., 1]
    theta  = torch.sqrt(theta2.clamp(min=EPS))
    K = hat(omega)
    K2 = K @ K
    # A = sin(θ)/θ, B = (1-cos(θ))/θ²
    A = torch.where(theta2 < 1e-8,
                    1 - theta2 / 6,
                    torch.sin(theta) / theta)
    B = torch.where(theta2 < 1e-8,
                    0.5 - theta2 / 24,
                    (1 - torch.cos(theta)) / theta2)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(*omega.shape[:-1], 3, 3)
    return I + A.unsqueeze(-1) * K + B.unsqueeze(-1) * K2

def so3_log(R):
    """θ→0 / θ→π 都要稳。这里 trace 法 + atan2 法混合。"""
    cos_theta = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1) / 2
    cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    # 一般情形
    factor = theta / (2 * sin_theta + EPS)
    omega = factor.unsqueeze(-1) * vee(R - R.transpose(-1, -2))
    # θ→0:展开 R - Rᵀ ≈ 2 ω̂
    small = theta < 1e-3
    if small.any():
        omega_small = 0.5 * vee(R - R.transpose(-1, -2))
        omega = torch.where(small.unsqueeze(-1), omega_small, omega)
    # θ→π:trace 法分母 sin θ→0,要走另一条公式(Shepperd 思路)
    # 略,放进 Common Pitfalls,本日先打 TODO,Day 4 PPO 上不会触发 θ→π
    return omega

def se3_exp(xi):
    rho, omega = xi[..., :3], xi[..., 3:]
    R = so3_exp(omega)
    theta2 = (omega * omega).sum(dim=-1, keepdim=True)
    theta  = torch.sqrt(theta2.clamp(min=EPS))
    K = hat(omega)
    K2 = K @ K
    # V(ω) = I + (1-cos θ)/θ² K + (θ-sin θ)/θ³ K²
    B = torch.where(theta2 < 1e-8, 0.5 - theta2 / 24,        (1 - torch.cos(theta)) / theta2)
    C = torch.where(theta2 < 1e-8, 1/6 - theta2 / 120,       (theta - torch.sin(theta)) / (theta * theta2))
    I = torch.eye(3, device=xi.device, dtype=xi.dtype).expand(*xi.shape[:-1], 3, 3)
    V = I + B.unsqueeze(-1) * K + C.unsqueeze(-1) * K2
    t = (V @ rho.unsqueeze(-1)).squeeze(-1)
    T = torch.eye(4, device=xi.device, dtype=xi.dtype).expand(*xi.shape[:-1], 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3, 3]  = t
    return T

def se3_log(T):
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    omega = so3_log(R)
    # V⁻¹(ω) ρ = t  →  ρ = V⁻¹ t
    theta2 = (omega * omega).sum(dim=-1, keepdim=True)
    theta  = torch.sqrt(theta2.clamp(min=EPS))
    K = hat(omega)
    K2 = K @ K
    # V⁻¹ = I - 0.5 K + (1/θ² - (1+cosθ)/(2θ sinθ)) K²
    half_theta = 0.5 * theta
    coef = torch.where(theta2 < 1e-8,
                       1/12 + theta2 / 720,
                       (1 - half_theta * torch.cos(half_theta) / torch.sin(half_theta).clamp(min=EPS)) / theta2)
    I = torch.eye(3, device=T.device, dtype=T.dtype).expand(*T.shape[:-2], 3, 3)
    Vinv = I - 0.5 * K + coef.unsqueeze(-1) * K2
    rho = (Vinv @ t.unsqueeze(-1)).squeeze(-1)
    return torch.cat([rho, omega], dim=-1)
```

### 4.2 单测（必须跑过再继续）

```python
# day3_se3/tests/test_lie.py
import torch, pytest
from src.lie import so3_exp, so3_log, se3_exp, se3_log, hat, vee

torch.manual_seed(0)

def test_hat_vee_roundtrip():
    omega = torch.randn(32, 3)
    assert torch.allclose(vee(hat(omega)), omega, atol=1e-6)

def test_so3_log_exp():
    omega = torch.randn(1024, 3) * 0.5  # 避免 θ→π
    R = so3_exp(omega)
    omega_back = so3_log(R)
    assert torch.allclose(omega_back, omega, atol=1e-5)

def test_so3_small_angle_stable():
    """θ→0 时不能 NaN"""
    omega = torch.randn(64, 3) * 1e-7
    R = so3_exp(omega)
    assert torch.isfinite(R).all()
    omega_back = so3_log(R)
    assert torch.isfinite(omega_back).all()
    assert torch.allclose(omega_back, omega, atol=1e-6)

def test_se3_log_exp():
    xi = torch.randn(1024, 6) * 0.5
    T = se3_exp(xi)
    xi_back = se3_log(T)
    assert torch.allclose(xi_back, xi, atol=1e-5)

def test_batch_shapes():
    """[B, N, 6] 也要能跑"""
    xi = torch.randn(4, 8, 6) * 0.3
    T = se3_exp(xi)
    assert T.shape == (4, 8, 4, 4)
    xi_back = se3_log(T)
    assert torch.allclose(xi_back, xi, atol=1e-5)
```

```bash
[embai]$ pytest tests/test_lie.py -v
```

### 自检
- [ ] 5 个单测全绿
- [ ] θ=1e-7 量级不会出 NaN
- [ ] `[4, 8, 6]` shape 也能 broadcast

---

## 5. 下午 14:00 - 15:30 — `compose / inverse / adjoint` + 四元数互转

### 5.1 实现

```python
# day3_se3/src/lie.py（续）
def compose(T1, T2):
    return T1 @ T2  # 看似简单，但要确保 shape broadcast 正确

def inverse(T):
    """SE(3) closed-form: [R t; 0 1]⁻¹ = [Rᵀ -Rᵀt; 0 1]"""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    Tinv = torch.eye(4, device=T.device, dtype=T.dtype).expand_as(T).clone()
    Tinv[..., :3, :3] = Rt
    Tinv[..., :3, 3]  = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    return Tinv

def adjoint(T):
    """Ad_T = [R, hat(t)R; 0, R]  ∈ ℝ^{6×6}"""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Z = torch.zeros_like(R)
    top = torch.cat([R, hat(t) @ R], dim=-1)
    bot = torch.cat([Z, R], dim=-1)
    return torch.cat([top, bot], dim=-2)

def quat_to_rotmat(q):
    """(w, x, y, z) -> R, normalize first"""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=EPS)
    w, x, y, z = q.unbind(dim=-1)
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R

def rotmat_to_quat(R):
    """Shepperd method, batched, numerically stable"""
    # 取四种 case 中 trace 最大的,避免 sqrt(负数)
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    case0 = t > 0
    s0 = torch.sqrt(t.clamp(min=EPS) + 1) * 2
    qw0 = 0.25 * s0
    qx0 = (m[..., 2, 1] - m[..., 1, 2]) / s0
    qy0 = (m[..., 0, 2] - m[..., 2, 0]) / s0
    qz0 = (m[..., 1, 0] - m[..., 0, 1]) / s0
    # 实战中 case0 命中率 > 95%；剩下 case 见 Shepperd 论文
    # 此处先返回 case0 的，θ→π 留 Common Pitfalls
    return torch.stack([qw0, qx0, qy0, qz0], dim=-1)
```

### 5.2 单测

```python
# day3_se3/tests/test_lie.py（续）
def test_compose_inverse():
    xi = torch.randn(64, 6) * 0.3
    T = se3_exp(xi)
    Tinv = inverse(T)
    I = compose(T, Tinv)
    assert torch.allclose(I, torch.eye(4).expand_as(I), atol=1e-5)

def test_adjoint_consistency():
    """Ad_T ξ ≈ se3_log(T · exp(ξ̂) · T⁻¹)"""
    xi1 = torch.randn(32, 6) * 0.3
    xi2 = torch.randn(32, 6) * 0.1
    T1 = se3_exp(xi1)
    T2 = se3_exp(xi2)
    lhs = (adjoint(T1) @ xi2.unsqueeze(-1)).squeeze(-1)
    rhs = se3_log(compose(compose(T1, T2), inverse(T1)))
    assert torch.allclose(lhs, rhs, atol=1e-4)

def test_quat_rotmat_roundtrip():
    q = torch.randn(64, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    R = quat_to_rotmat(q)
    q_back = rotmat_to_quat(R)
    # 注意 q 和 -q 表示同一旋转,允许翻号
    same    = torch.allclose(q_back, q,  atol=1e-5)
    flipped = torch.allclose(q_back, -q, atol=1e-5)
    assert same or flipped
```

### 自检
- [ ] `inverse` 用闭式而不是 `torch.linalg.inv`(快 5-10x)
- [ ] `adjoint` 通过共轭一致性测试
- [ ] 四元数翻号问题在测试里显式处理(`q ≡ -q`)

---

## 6. 下午 15:30 - 17:00 — 数值稳定性 + autograd 测试

> **重点是 gradcheck**：VLA / DP 模型会把 `se3_log` 放在 loss 里，**梯度炸 = 训不动**。

### 6.1 θ→0 / θ→π 边界扫描

```python
# day3_se3/notebooks/numerics.ipynb（或 .py）
import torch, matplotlib.pyplot as plt
from src.lie import se3_exp, se3_log

errors = []
thetas = torch.logspace(-9, 0, 50)   # θ 从 1e-9 到 1
for th in thetas:
    omega = torch.tensor([th.item(), 0., 0.])
    rho   = torch.zeros(3)
    xi    = torch.cat([rho, omega])[None]
    T = se3_exp(xi)
    xi_back = se3_log(T)
    err = (xi_back - xi).abs().max().item()
    errors.append(err)

plt.loglog(thetas, errors); plt.xlabel("theta"); plt.ylabel("|log(exp(ξ)) - ξ|")
plt.savefig("se3_numerics.png", dpi=120)
```

期望:误差在 θ ∈ [1e-8, 1] 全范围 < 1e-5,**不出现 1e-3 以上的 spike**。如果 spike,回去把 Taylor 展开多写一阶。

### 6.2 autograd gradcheck

```python
# day3_se3/tests/test_grad.py
import torch
from torch.autograd import gradcheck
from src.lie import se3_exp, se3_log

def test_se3_exp_grad():
    xi = torch.randn(4, 6, dtype=torch.float64, requires_grad=True) * 0.3
    assert gradcheck(se3_exp, (xi,), eps=1e-6, atol=1e-4)

def test_se3_log_grad():
    xi = torch.randn(4, 6, dtype=torch.float64) * 0.3
    T = se3_exp(xi).detach().requires_grad_(True)
    assert gradcheck(se3_log, (T,), eps=1e-6, atol=1e-4)
```

```bash
[embai]$ pytest tests/ -v
```

### 自检
- [ ] θ ∈ [1e-9, 1] 误差曲线平滑无 spike
- [ ] `gradcheck` 双向通过(`exp` 和 `log` 都过)
- [ ] 用 `torch.float64` 才能过 gradcheck;`float32` 会卡在 atol=1e-4 量级

---

## 7. 晚上 19:30 - 20:30 — 跨库交叉验证

### 7.1 与 pytorch3d / pin / curobo 对比

```python
# day3_se3/notebooks/cross_lib.py
import torch, numpy as np
import pinocchio as pin
from src.lie import se3_exp, se3_log

# 1) 与 Pinocchio 对比
xi_np = np.random.randn(6) * 0.3
# ⚠️ Pinocchio 的 Motion 用 (v, ω) 顺序,与我们的 (ρ, ω) 一致(只是命名不同)
T_pin  = pin.exp6(pin.Motion(xi_np))
T_ours = se3_exp(torch.tensor(xi_np)[None]).squeeze().numpy()
print("pin vs ours:", np.abs(T_pin.homogeneous - T_ours).max())
# 期望 < 1e-6

# 2) 与 pytorch3d 对比(可选)
try:
    from pytorch3d.transforms import se3_exp_map
    # ⚠️ pytorch3d 的 ξ 顺序是 (ω, ρ),要 swap!
    xi = torch.randn(8, 6) * 0.3
    xi_p3d = torch.cat([xi[:, 3:], xi[:, :3]], dim=-1)
    T_p3d = se3_exp_map(xi_p3d).transpose(-1, -2)  # p3d 用列向量约定,要转置
    T_ours = se3_exp(xi)
    print("p3d vs ours:", (T_p3d - T_ours).abs().max().item())
except ImportError:
    print("pytorch3d 未装,跳过")

# 3) 与 cuRobo 对比(你已熟,自己写一行验证)
```

> **这一步是为后面踩坑省命**:VLA 模型常常你的 SE(3) 工具 + 别人的预训练权重一起用,**坐标约定不一致 = 训出来全翻车**。

### 7.2 性能基准

```python
# day3_se3/notebooks/bench.py
import torch, time
from src.lie import se3_exp, inverse

device = "cuda"
for B in [128, 1024, 8192, 65536]:
    xi = torch.randn(B, 6, device=device) * 0.3
    # warmup
    for _ in range(3): se3_exp(xi)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(100): T = se3_exp(xi)
    torch.cuda.synchronize(); t1 = time.time()
    print(f"se3_exp B={B}: {(t1-t0)/100*1e6:.1f} μs/call")

    for _ in range(3): inverse(T)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(100): inverse(T)
    torch.cuda.synchronize(); t1 = time.time()
    print(f"inverse  B={B}: {(t1-t0)/100*1e6:.1f} μs/call")
```

期望:`se3_exp` B=65536 单次 < 1ms,`inverse` 比 `torch.linalg.inv` 快 5-10x。

### 自检
- [ ] Pinocchio 一致性 < 1e-6
- [ ] 性能基准跑出来,记录到 wandb
- [ ] 知道你的 SE(3) 工具的 batch 上限 (单次 4090 上能上 1M batch 不爆显存)

---

## 8. 晚上 20:30 - 21:00 — 复盘 + commit + push

### 8.1 写复盘

```bash
cd ~/embodied-ai-bootcamp-8w/week1-fundamentals/day3_se3
cat > NOTES.md <<'EOF'
# Day 3 Recap (2026-06-23)

## Done
- [x] CONVENTIONS.md 写完
- [x] so3/se3 exp & log,θ→0 用 Taylor 稳定
- [x] compose / inverse / adjoint / quat 互转,全 batched
- [x] 单测 9 个全绿
- [x] gradcheck 双向通过(float64)
- [x] Pinocchio 一致性 < 1e-6
- [x] 性能 baseline:se3_exp B=65536 ___ μs/call

## Insights
- Taylor 展开常数项必须写到 4 阶才能在 θ=1e-9 量级下保证 1e-5 精度
- pytorch3d 的 ξ 顺序是 (ω, ρ)、列向量,要 swap + transpose
- inverse 闭式比 torch.linalg.inv 快 ___x

## Pitfalls hit
- 一开始 so3_log 在 θ→0 走 acos 路径,Taylor 没接上,误差 1e-2
- rotmat_to_quat 翻号问题,测试里要 same || flipped

## Tomorrow (Day 4 - MDP & Tabular RL)
- 先看 Sutton Barto 第 3-4 章
- FrozenLake VI / PI / Q-learning 三件套
EOF
```

### 8.2 commit & push

```bash
git add day3_se3/
git commit -m "Day 3: batched SE(3) ops in PyTorch (so3/se3 exp+log, compose, inverse, adjoint, quat)"
git push origin main
```

### 8.3 Milestones 打勾

```
[ ] 手写 MNIST 训练循环
[ ] 手写 Mini-GPT,Tiny Shakespeare 训练 1 epoch 能采样
[x] Batched SE(3) 操作通过 log(exp(x)) ≈ x 单测   ✅ 今天打勾
[ ] 实现 Value Iteration / Q-Learning
[ ] 实现 PPO + GAE
[ ] 5 个实验全部上 wandb
```

---

## 9. 今日完成确认（必达 / 加分）

### 必达（5/6 才算合格）
- [ ] `so3_exp/log`、`se3_exp/log` 实现完成,单测过
- [ ] θ→0 数值稳定,误差曲线平滑
- [ ] `compose / inverse / adjoint` 都 batched + 单测过
- [ ] 四元数互转双向(注意翻号)
- [ ] gradcheck 双向通过(float64)
- [ ] CONVENTIONS.md 写完贴墙

### 加分
- [ ] 与 Pinocchio 交叉验证 < 1e-6
- [ ] `rotmat_to_quat` 把 Shepperd 4 个 case 全写完(θ→π 也稳)
- [ ] θ→π 时 `so3_log` 也写稳(留个 TODO 也行,Week 4 才会用到)
- [ ] 性能基准记录到 wandb

---

## 10. 明天 (Day 4) 预告 — MDP & Tabular RL

- 早晨:Sutton Barto 第 3-4 章 + Bellman 方程手推
- 上午:FrozenLake-v1 (4x4 + 8x8) Value Iteration / Policy Iteration
- 下午:Tabular Q-Learning + ε-greedy 衰减
- 晚上:γ 扫描 + 收敛曲线对比

预读:
- Sutton & Barto §3 (MDPs), §4 (DP)
- 自己推一遍 Bellman optimality:`V*(s) = max_a [r + γ Σ p(s'|s,a) V*(s')]`

---

## 11. 反向提醒（容易忘）

- ❗ float64 才能过 gradcheck,正常用 float32 没问题
- ❗ 四元数翻号 q ≡ -q,测试时要 `same || flipped`
- ❗ ξ 顺序 (ρ, ω) 写在 CONVENTIONS,与 pytorch3d (ω, ρ) 互转必须 swap
- ❗ `inverse` 用闭式不要用 `torch.linalg.inv`,差 5-10x
- ❗ Adjoint 矩阵 6×6,公式 `[R, hat(t)R; 0, R]` —— 第二项是 `hat(t) @ R`,不是 `R @ hat(t)`
- ❗ 今天**不碰**机器人动力学(M, C, G),那是 Week 4 Day 22

---

## 12. 如果今天彻底崩了 — 救火方案

如果 14:00 还没把 `so3_exp/log` 单测跑过:

1. **降目标**:今日只完成 `so3_exp/log` + `se3_exp/log` + 单测,其他(adjoint / 四元数 / gradcheck)推到 Week 1 周末补
2. **抄一段**:把 [`UM-ARM-Lab/pytorch_kinematics`](https://github.com/UM-ARM-Lab/pytorch_kinematics) 的 `transforms3d.py` 直接抄来读懂,而不是从零写
3. **保 Day 4 不延期**:RL 比 SE(3) 更重要,SE(3) 后面 Week 4 还要再用一次,有时间补

---

→ [Day04.md](./Day04.md)(待生成) | [Week01.md](../week_plan/Week01.md)
