# Day 03 Notes - Batched SE(3) in PyTorch

> Goal: 把 SO(3) / SE(3) 的核心公式、工程约定、数值稳定坑位压成一页复习卡。写代码时优先看 `CONVENTIONS.md`，这里负责“为什么”。

---

## 1. Hat / Vee

`hat` 把 3D 向量变成反对称矩阵，表示叉乘线性算子：

```text
omega = [wx, wy, wz]^T

omega^ =
[  0  -wz   wy
   wz   0  -wx
  -wy  wx    0 ]

omega^ x = omega cross x
vee(omega^) = omega
```

记忆点：`hat(omega)` 不是旋转矩阵，它是 Lie algebra `so(3)` 里的元素；`exp(hat(omega))` 才落到 `SO(3)`。

---

## 2. SO(3) Exp / Log

令 `theta = ||omega||`，`K = hat(omega)`。

```text
Exp_SO3(omega) = R
R = I + A K + B K^2

A = sin(theta) / theta
B = (1 - cos(theta)) / theta^2
```

`theta -> 0` 时直接除会炸，要用 Taylor：

```text
A = 1 - theta^2 / 6 + theta^4 / 120 + O(theta^6)
B = 1/2 - theta^2 / 24 + theta^4 / 720 + O(theta^6)
```

Log 的普通公式：

```text
theta = arccos((trace(R) - 1) / 2)
omega = theta / (2 sin(theta)) * vee(R - R^T)
```

Log 的坑：

- `theta -> 0`: `R - R^T ~= 2 hat(omega)`，所以可用 `omega ~= 0.5 * vee(R - R^T)`。
- `theta -> pi`: `sin(theta) -> 0`，trace 公式病态；需要用对角线最大分量 / Shepperd-like 分支恢复旋转轴。
- `acos` 的输入必须 clamp 到 `[-1, 1]` 附近，否则浮点误差会给 NaN。

---

## 3. SE(3) Exp / Log

本项目 twist 顺序固定为：

```text
xi = (rho, omega) in R^6
rho: translation part
omega: rotation part
```

指数映射：

```text
Exp_SE3(xi) = T =
[ R  t
  0  1 ]

R = Exp_SO3(omega)
t = V(omega) rho

V = I + B K + C K^2
B = (1 - cos(theta)) / theta^2
C = (theta - sin(theta)) / theta^3
```

`theta -> 0` 时：

```text
B = 1/2 - theta^2 / 24 + theta^4 / 720 + O(theta^6)
C = 1/6 - theta^2 / 120 + theta^4 / 5040 + O(theta^6)
```

对数映射：

```text
omega = Log_SO3(R)
rho = V^{-1}(omega) t

V^{-1} = I - 1/2 K + D K^2
D = 1/theta^2 - (1 + cos(theta)) / (2 theta sin(theta))
```

更稳定的等价写法常用：

```text
D = (1 - (theta / 2) cot(theta / 2)) / theta^2
```

小角度：

```text
D = 1/12 + theta^2 / 720 + O(theta^4)
```

---

## 4. Compose / Inverse / Adjoint

SE(3) 变换采用列向量语义：

```text
T_AB: B frame pose in A frame
p_A = T_AB p_B
```

组合和逆：

```text
compose(T1, T2) = T1 T2

T = [R t; 0 1]
T^{-1} = [R^T  -R^T t; 0 1]
```

Adjoint 把 twist 从一个 frame 换到另一个 frame。对 `xi = (rho, omega)`：

```text
Ad_T =
[ R  hat(t) R
  0      R    ]

xi_A = Ad_T_AB xi_B
```

最重要的单测不是背公式，而是共轭一致性：

```text
Ad_T xi ~= Log(T Exp(xi) T^{-1})
```

常见错误：右上角是 `hat(t) @ R`，不是 `R @ hat(t)`。

---

## 5. Conventions 必背

```text
Quaternion: (w, x, y, z), Hamilton, q and -q 表示同一旋转
Twist:      xi = (rho, omega)
Perturb:    left perturbation, T' = Exp(xi) T
Frame:      T_AB maps p_B to p_A
```

跨库时必须先问四件事：

| 库 / 文献 | twist 顺序 | quaternion | pose 语义 |
|---|---:|---:|---|
| 本项目 | `(rho, omega)` | `(w, x, y, z)` | `p_A = T_AB p_B` |
| Solà / Pinocchio | `(v, omega)` | 通常 `(x, y, z, w)` 或对象封装，需查接口 | Lie group 左扰动常见 |
| Lynch Modern Robotics | 常写 `(omega, v)` | 不主打 quaternion | body / space twist 要区分 |
| pytorch3d | 常见 se3 向量为 `(omega, translation)` | 接口需查 | 可能有转置 / 行列向量差异 |
| cuRobo / ROS tf | 工程上常见 `(w, x, y, z)` | `(w, x, y, z)` | `T_AB` 风格常见 |

结论：不要靠记忆跨库，写转换函数时显式 `swap`、`transpose`、注释约定。

---

## 6. Autograd / Numerics Checklist

必须测：

- `vee(hat(omega)) == omega`
- `Log_SO3(Exp_SO3(omega)) ~= omega`，先限制 `||omega|| < pi`
- `Log_SE3(Exp_SE3(xi)) ~= xi`
- 小角度 `theta = 1e-9 ... 1e-3` 不 NaN、误差无 spike
- `gradcheck(se3_exp)` 和 `gradcheck(se3_log)` 用 `float64`
- batched shape: `[B, 6]`、`[B, N, 6]` 都能工作
- quaternion roundtrip 要允许 `q == -q`

数值红线：

```text
theta -> 0: 走 Taylor
theta -> pi: so3_log / rotmat_to_quat 走特殊分支
acos input: clamp
division: clamp denominator, but 小角度最好不要只靠 EPS 硬除
```

---

## 7. Performance Notes

`[B, 6] -> [B, 4, 4]` 是纯张量代数：

```text
FLOPs ~= B * O(200)
瓶颈通常不是 matmul，而是 sin / cos / acos / atan2 和分支稳定性
```

工程原则：

- `inverse(T)` 用闭式，不用 `torch.linalg.inv`。
- 尽量保持 broadcastable，不手写 for-loop。
- 常规训练用 `float32`，`gradcheck` 用 `float64`。
- `torch.where` 两边都会被计算，所以分母仍要 clamp，不能指望未选分支不出 NaN。

---

## 8. 口头自测

- Rodrigues 公式是什么？`A`、`B` 分别是什么？
- 为什么 `theta -> 0` 时 Exp 仍可导？因为 `sin(theta)/theta` 等系数有良性 Taylor 极限。
- `V(omega)` 在 SE(3) Exp 里负责什么？把 Lie algebra 里的平移 `rho` 映到 group 里的平移 `t`。
- 为什么 twist 顺序要写死？不同库 `(rho, omega)` / `(omega, rho)` 混用会让模型 loss 看起来正常、物理意义全错。
- Adjoint 是干嘛的？把 twist / perturbation 从一个 frame 表达到另一个 frame。
- 四元数 roundtrip 为什么不能直接 `allclose(q, q_back)`？因为 `q` 和 `-q` 是同一个旋转。

---

## 9. 今日一句话

SE(3) 代码真正难的不是公式本身，而是把“约定”和“极限情况”写得足够明确，让后面的 Diffusion Policy、Pinocchio Jacobian、VLA action head 都不会在坐标系里悄悄歪掉。
