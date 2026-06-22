# Day 01 — 环境验证 + PyTorch 工程肌肉记忆 + MNIST baseline

> **日期**：2026-06-19（Week 1, Day 1）
> **本日定位**：把"硬件 + 系统 + Python + PyTorch + wandb"全套打通，跑出本计划第一个 wandb run。
> **总投入**：约 8 小时（含早晨 / 上午 / 下午 / 晚上）

---

## 0. 出发前自检（5 分钟）

打开终端跑下面这一串，全部正常再继续：

```bash
# OS 信息
uname -a
lsb_release -a            # 期望 Ubuntu 24.04（你已装 ROS2 Jazzy → 推断 24.04）

# GPU & 驱动
nvidia-smi                # 期望看到 RTX 4090, Driver 590.48.01, CUDA 13.1

# Python
which python3
python3 --version         # 期望 3.10+ 或 3.12

# Conda（如果还没装）
which conda || echo "需要装 miniconda"

# ROS2（你已装）
source /opt/ros/jazzy/setup.bash
ros2 --version
ros2 doctor               # 期望 0 errors

# 已下载的 Isaac Sim
ls ~/.local/share/ov/pkg/ 2>/dev/null      # 看是否有 isaac-sim-5.1 / 6.0
# 或
which isaacsim 2>/dev/null
```

如果上面任何一项不对，**先停下来修这一项**，再开始 Day 1 内容。

---

## 1. 今日时间表（hour by hour）

| 时段 | 时长 | 任务 | 产出 |
|------|------|------|------|
| 07:00 - 08:30 | 1.5h | 早晨论文 + 概念预热 | 笔记 1 页 |
| 09:00 - 10:30 | 1.5h | conda 主环境搭建 + PyTorch 验证 | `embai` env 可用 |
| 10:30 - 12:00 | 1.5h | wandb 注册 + GitHub repo 初始化 | wandb 第一个 run |
| 14:00 - 16:00 | 2h   | MNIST baseline（手写 + Lightning 重构） | 训练 acc > 97% |
| 16:00 - 17:00 | 1h   | batch_size 扫描实验 + `torch.compile` 测速 | 一张速度对比图 |
| 19:30 - 20:30 | 1h   | Isaac Sim 6.0 sanity check | GUI 能起来 |
| 20:30 - 21:00 | 0.5h | 写 Day 1 复盘 + commit + push | GitHub 第一次 commit |

---

## 2. 早晨 (07:00 - 08:30) — 论文 + 概念

### 任务
- 读：[The Illustrated Transformer (Jay Alammar)](http://jalammar.github.io/illustrated-transformer/)（不用全读，重点看 Self-Attention 部分）
- 读：*Attention Is All You Need*（Vaswani 2017）**第 3 节**
- 公式手推：`Attention(Q,K,V) = softmax(QK^T / √d_k) V`，能解释为什么除以 `√d_k`

### 产出
在 `Day_Plan/notes/Day01_attention.md` 写一页：
- 公式
- 为什么除以 `√d_k`：让 dot product 方差稳定，防 softmax 进入梯度消失区
- multi-head 的意义：相当于多个子空间并行学
- Q / K / V 的物理解释（用 retrieval 类比）

### 自检
- [ ] 闭眼能默写 attention 公式
- [ ] 能解释为什么 attention 的 mask 在 causal LM 里是上三角

---

## 3. 上午 09:00 - 10:30 — Conda 主环境 + PyTorch

### 3.1 装 miniconda（如已装跳过）

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 一路回车 + yes，结束后重启 shell
exec $SHELL
conda --version           # 期望 conda 24+
conda config --set auto_activate_base false
```

### 3.2 创建 embai 主环境

```bash
conda create -n embai python=3.10 -y
conda activate embai

# PyTorch（Driver 590 → 用 cu124 wheels，最稳）
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 基础工具
pip install numpy scipy matplotlib einops tqdm rich pyyaml
pip install gymnasium[classic_control,box2d,mujoco] stable-baselines3
pip install wandb hydra-core lightning
pip install jupyter ipython notebook
```

### 3.3 验证 PyTorch + CUDA

```bash
python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
print('device name:', torch.cuda.get_device_name(0))
print('cuda version (compiled):', torch.version.cuda)

# 跑一个真实 GPU 矩阵乘
a = torch.randn(4096, 4096, device='cuda')
b = torch.randn(4096, 4096, device='cuda')
c = a @ b
torch.cuda.synchronize()
print('matmul ok, c.sum():', c.sum().item())
"
```

期望输出：
```
torch: 2.5.x
cuda available: True
device count: 1
device name: NVIDIA GeForce RTX 4090
cuda version (compiled): 12.4
matmul ok, c.sum(): <some number>
```

### 3.4 常见踩坑

| 报错 | 原因 | 解决 |
|------|------|------|
| `cuda available: False` | PyTorch 装成了 cpu 版 | `pip uninstall torch -y && pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| `Driver too old` | 不会发生，你 590 够新 | — |
| `libcudart.so not found` | 系统 CUDA toolkit 路径混 | PyTorch 自带 runtime，`unset LD_LIBRARY_PATH` 试试 |

### 自检
- [ ] `embai` 环境激活后 `which python` 指向 conda env
- [ ] PyTorch CUDA 可用，能在 4090 上跑矩阵乘
- [ ] `pip list | grep -E "torch|gym|wandb|lightning"` 全绿

---

## 4. 上午 10:30 - 12:00 — wandb + GitHub

### 4.1 wandb

```bash
# 注册 https://wandb.ai/site，然后
wandb login
# 粘贴 API key
```

写一个 hello world 验证：

```python
# Day_Plan/scripts/wandb_hello.py
import wandb, time, random
wandb.init(project="embai-day01", name="hello", config={"lr": 1e-3})
for step in range(50):
    wandb.log({"loss": 1.0 / (step + 1) + random.random() * 0.05}, step=step)
    time.sleep(0.05)
wandb.finish()
print("打开 wandb URL 查看曲线")
```

```bash
mkdir -p ~/Documents/Embodied-AI/Day_Plan/scripts
cd ~/Documents/Embodied-AI/Day_Plan/scripts
python wandb_hello.py
```

### 4.2 GitHub Repo 初始化

```bash
mkdir -p ~/embodied-ai-bootcamp-8w
cd ~/embodied-ai-bootcamp-8w
git init -b main

# 目录结构
mkdir -p week1-fundamentals/{day1_pytorch,day2_transformer,day3_se3,day4_mdp,day5_pg,day6_ppo,day7_recap}
mkdir -p week1-fundamentals/day1_pytorch/{src,configs,logs,checkpoints}

# .gitignore
cat > .gitignore <<'EOF'
__pycache__/
*.pyc
.ipynb_checkpoints/
wandb/
logs/
checkpoints/
*.pt
*.pth
data/
.venv/
.vscode/
.idea/
EOF

# README
cat > README.md <<'EOF'
# Embodied AI Bootcamp — 8 Weeks

跟随 [Embodied-AI 学习计划](../Documents/Embodied-AI) 的实战代码仓库。

## Week 1 — PyTorch + RL Foundations
- Day 1: MNIST baseline + wandb
- Day 2: Mini-GPT
- Day 3: Batched SE(3)
- Day 4: MDP & Tabular RL
- Day 5: REINFORCE / A2C
- Day 6: PPO + GAE
- Day 7: Recap
EOF

git add .
git commit -m "init: bootcamp scaffold"

# GitHub 上建私有 repo（先用 gh 或网页都行）
# 如果有 gh：
# gh auth login
# gh repo create embodied-ai-bootcamp-8w --private --source=. --remote=origin --push

# 没 gh 就网页建 repo，然后：
# git remote add origin git@github.com:<your_user>/embodied-ai-bootcamp-8w.git
# git push -u origin main
```

### 自检
- [ ] wandb 网页上看到 `embai-day01/hello` run，曲线正常
- [ ] GitHub 上 repo 已建好（公开或私有都行），有第一次 commit

---

## 5. 下午 14:00 - 16:00 — MNIST 手写 + Lightning 重构

### 5.1 手写 MNIST 训练（不用 Lightning）

```bash
cd ~/embodied-ai-bootcamp-8w/week1-fundamentals/day1_pytorch
```

```python
# src/train_mnist_raw.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

class MLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 10),
        )
    def forward(self, x):
        return self.net(x)

def main():
    cfg = dict(lr=1e-3, batch=128, epochs=5, hidden=256, seed=42)
    wandb.init(project="embai-day01", name="mnist-raw", config=cfg)
    torch.manual_seed(cfg["seed"])
    device = "cuda"

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("./data", train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, num_workers=2, pin_memory=True)

    model = MLP(cfg["hidden"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    step = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 50 == 0:
                acc = (logits.argmax(-1) == y).float().mean().item()
                wandb.log({"train/loss": loss.item(), "train/acc": acc}, step=step)
            step += 1

        # eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(-1)
                correct += (pred == y).sum().item()
                total   += y.size(0)
        test_acc = correct / total
        wandb.log({"test/acc": test_acc, "epoch": epoch}, step=step)
        print(f"Epoch {epoch}: test acc = {test_acc:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()
```

```bash
python src/train_mnist_raw.py
```

**期望**：5 epoch 后 `test/acc > 0.97`，wandb 曲线平滑下降。

### 5.2 Lightning 重构（同样的逻辑用 Lightning 实现一遍）

```python
# src/train_mnist_lightning.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import lightning as L
from lightning.pytorch.loggers import WandbLogger

class LitMNIST(L.LightningModule):
    def __init__(self, hidden=256, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc  = (logits.argmax(-1) == y).float().mean()
        self.log_dict({"train/loss": loss, "train/acc": acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc  = (logits.argmax(-1) == y).float().mean()
        self.log_dict({"val/loss": loss, "val/acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

def main():
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("./data", train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, num_workers=2, pin_memory=True)

    logger = WandbLogger(project="embai-day01", name="mnist-lightning")
    trainer = L.Trainer(max_epochs=5, accelerator="gpu", devices=1,
                        logger=logger, log_every_n_steps=50)
    trainer.fit(LitMNIST(), train_loader, test_loader)

if __name__ == "__main__":
    main()
```

```bash
python src/train_mnist_lightning.py
```

### 5.3 自检
- [ ] 两份代码都能跑出 acc > 97%
- [ ] wandb 上有 `mnist-raw` 和 `mnist-lightning` 两条 run
- [ ] 你能口头说清 Lightning 帮你省了哪几行（device.to()、zero_grad、optimizer.step、eval loop、log）

---

## 6. 下午 16:00 - 17:00 — Batch Size 扫描 + torch.compile

### 6.1 batch_size 扫描

```bash
# 用 wandb sweep
cat > configs/sweep_batch.yaml <<'EOF'
program: src/train_mnist_raw.py
method: grid
parameters:
  batch:
    values: [16, 32, 64, 128, 256, 512]
EOF
```

简化做法：直接手动跑 5 个 size，wandb name 区分：

```python
# src/sweep_batch.py
import subprocess
for b in [16, 32, 64, 128, 256, 512]:
    subprocess.run([
        "python", "-c",
        f"""
import torch, time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn, torch.nn.functional as F
import wandb
wandb.init(project='embai-day01', name='bs-{b}', config={{'batch': {b}}}, reinit=True)
tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
ds = datasets.MNIST('./data', train=True, download=True, transform=tfm)
loader = DataLoader(ds, batch_size={b}, shuffle=True, num_workers=4, pin_memory=True)
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
torch.cuda.synchronize(); t0 = time.time()
for x, y in loader:
    x, y = x.cuda(), y.cuda()
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
torch.cuda.synchronize(); dt = time.time() - t0
wandb.log({{'epoch_time_s': dt, 'samples_per_s': len(ds) / dt}})
wandb.finish()
"""
    ])
```

### 6.2 torch.compile 测速

```python
# src/test_compile.py
import torch, time
import torch.nn as nn

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 1024), nn.ReLU(),
                      nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 10)).cuda()
x = torch.randn(256, 1, 28, 28, device="cuda")

# warmup
for _ in range(10): model(x)
torch.cuda.synchronize(); t0 = time.time()
for _ in range(1000): out = model(x)
torch.cuda.synchronize()
print(f"eager: {time.time() - t0:.3f}s")

cmodel = torch.compile(model)
for _ in range(10): cmodel(x)   # warmup（首次 compile 会慢）
torch.cuda.synchronize(); t0 = time.time()
for _ in range(1000): out = cmodel(x)
torch.cuda.synchronize()
print(f"compiled: {time.time() - t0:.3f}s")
```

期望：compile 后比 eager 快 1.2-2x（小模型加速比有限）。

### 自检
- [ ] wandb 上看到 6 条 batch size 不同的曲线
- [ ] 知道 4090 + MNIST 这种小任务上，batch=128~256 是甜点
- [ ] `torch.compile` 体感跑过

---

## 7. 晚上 19:30 - 20:30 — Isaac Sim 6.0 健康检查

> **本日不开始装 Isaac Lab**，只验证你已下载的 Isaac Sim 6.0 能起来。

### 7.1 找到你的 Isaac Sim 6.0 安装位置

```bash
# 方式 A：Omniverse Launcher 装的
ls ~/.local/share/ov/pkg/ | grep isaac

# 方式 B：手动解压
ls ~/isaac-sim* 2>/dev/null
ls /opt/isaac-sim* 2>/dev/null

# 方式 C：你直接下载 zip 解压到了某处，自己找一下
find ~ -maxdepth 4 -name "isaac-sim.sh" 2>/dev/null
```

设个环境变量方便后续调用：

```bash
echo 'export ISAACSIM_PATH=$HOME/.local/share/ov/pkg/isaac-sim-6.0.0' >> ~/.bashrc
# 把路径换成你实际找到的，然后
source ~/.bashrc
```

### 7.2 启动 Isaac Sim 6.0 GUI 一次

```bash
cd $ISAACSIM_PATH
./isaac-sim.sh
```

第一次启动会下载 shader cache，10-30 分钟。**不要急，挂着等**。

成功标志：
- 看到 NVIDIA Isaac Sim 主界面
- 没有 Vulkan / Driver / RTX 报错
- 左侧 Stage panel 显示 `World`

### 7.3 跑一个最简单的 Python 例子（确认 headless 能用）

在 Isaac Sim 6.0 里，打开 `Window > Examples Browser`，找到 `Python Scripting Examples > Simulation > Add Cubes` 之类，点 Play 看能动。

或命令行：

```bash
$ISAACSIM_PATH/python.sh -c "
from isaacsim import SimulationApp
sim = SimulationApp({'headless': True})
print('Isaac Sim started!')
sim.close()
"
```

期望：打印 `Isaac Sim started!` 然后正常退出。

### 7.4 常见踩坑

| 报错 | 解决 |
|------|------|
| `Vulkan validation error` | 装 `sudo apt install libvulkan1 vulkan-tools` |
| `Driver too old` | 不会，你 590 足够新 |
| `Failed to find shader cache` | 让它跑完第一次 download |
| `GLIBC_2.X not found` | Ubuntu 24.04 没问题；如果用更老 OS 需升级 |
| 完全黑屏 | Wayland → X11 切换试试 |

### 自检
- [ ] Isaac Sim 6.0 GUI 能启动（不闪退）
- [ ] `python.sh` 命令行能用 SimulationApp 跑一次

---

## 8. 晚上 20:30 - 21:00 — 复盘 + commit + push

### 8.1 写复盘

```bash
cd ~/embodied-ai-bootcamp-8w/week1-fundamentals/day1_pytorch
cat > NOTES.md <<'EOF'
# Day 1 Recap (2026-06-19)

## Done
- [x] embai conda env，PyTorch CUDA 可用
- [x] wandb 注册 + hello world
- [x] GitHub repo 初始化
- [x] MNIST 手写训练 acc = ___（填你的）
- [x] MNIST Lightning 重构 acc = ___
- [x] batch_size 扫描，最快 ___ samples/s @ batch=___
- [x] torch.compile 加速 ___x
- [x] Isaac Sim 6.0 GUI 能起来

## Insights
- Lightning 帮我省的 6 行：device 转移 / zero_grad / loss.backward / opt.step / eval mode 切换 / log 接口
- batch_size 越大不一定越快，受 IO 和 GPU 利用率限制
- ...

## Pitfalls hit
- ...

## Tomorrow (Day 2 - Mini-GPT)
- 早晨：Vaswani 2017 第 3 节再读一遍
- 上午：写 CausalSelfAttention 模块
- 下午：拼 mini-GPT 在 Tiny Shakespeare 训
EOF
```

### 8.2 commit & push

```bash
cd ~/embodied-ai-bootcamp-8w
git add .
git commit -m "day1: env setup + mnist baseline + isaac sim sanity check"
git push origin main
```

### 8.3 打卡 Milestones

打开 `~/Documents/Embodied-AI/Milestones.md`，对照 Week 1 的清单，**只勾"手写 MNIST 训练循环"那一项**——其他还要等后续天数。

---

## 9. 今日完成确认（必达 / 加分）

### 必达（完成 5/6 才算合格）
- [ ] embai conda env + PyTorch CUDA 可用
- [ ] wandb 第一个 run（hello + MNIST）
- [ ] GitHub repo 初始化并 push
- [ ] MNIST 手写训练 acc > 97%
- [ ] MNIST Lightning 版本同样 acc > 97%
- [ ] Isaac Sim 6.0 GUI 至少启动一次

### 加分
- [ ] batch_size 扫描完成 + 速度对比图
- [ ] `torch.compile` 跑过
- [ ] 复盘文档写完并 push

---

## 10. 明天 (Day 2) 预告 — Mini-GPT

明早 7 点准时打开 `Week01.md` 的 Day 2 部分。要点：
- 实现 `CausalSelfAttention`、`Block`、`MiniGPT`
- 在 Tiny Shakespeare 上训练，能采样出像样文字
- 4090 上整训练 < 30 分钟

**不要今晚提前看代码**——明早带着脑子读 Vaswani 论文，自己先想怎么写，再对答案。

---

## 11. 反向提醒（容易忘）

1. **关 Isaac Sim 再睡觉**：那东西占 8GB+ 显存，关了让 GPU 喘息
2. **wandb run 别都叫 run-1 run-2**：用 `name=` 起有意义的名字
3. **commit 信息别写 "fix"**：写"day1: <做了什么>"
4. **明天起床先 `nvidia-smi`**：确认 GPU 没被任何僵尸进程占着

晚安。明天打开 `Day_Plan/Day02.md`（如果我后续给你生成的话）或直接看 `Week01.md` Day 2。

---

> Good luck, Day 1 完成 = 整套计划 ignition 完成。
