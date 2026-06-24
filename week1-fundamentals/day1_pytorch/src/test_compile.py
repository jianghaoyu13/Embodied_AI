import torch
import time
import torch.nn as nn

# 构建一个放在 GPU 上的 MLP，用来比较 eager 模式和 torch.compile 后的前向推理速度。
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10),
).cuda()

# 构造一批假的 MNIST 形状输入：(batch, channel, height, width)。
x = torch.randn(256, 1, 28, 28, device="cuda")

# 先做几次预热，避免第一次运行时的初始化开销影响后面的计时。
for _ in range(10):
    model(x)
torch.cuda.synchronize()

# 测 eager mode（普通 PyTorch 执行方式）下 1000 次前向传播的总耗时。
t0 = time.time()
for _ in range(1000):
    out = model(x)
torch.cuda.synchronize()
print(f"eager: {time.time() - t0:.3f}s")

# torch.compile 会尝试对模型进行图捕获和优化，换取后续执行更高的效率。
cmodel = torch.compile(model)

# 编译后的前几次执行通常会更慢，因为这里会发生真正的编译和优化。
for _ in range(10):
    cmodel(x)  # warmup（首次 compile 会慢）
torch.cuda.synchronize()

# 测 compile 后 1000 次前向传播的总耗时。
t0 = time.time()
for _ in range(1000):
    out = cmodel(x)
torch.cuda.synchronize()
print(f"compiled: {time.time() - t0:.3f}s")
