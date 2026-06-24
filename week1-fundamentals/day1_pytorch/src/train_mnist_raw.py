import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb


class MLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        # 用最基础的多层感知机做 MNIST 分类：展平 -> 两层隐藏层 -> 10 类输出。
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)


def main():
    # 把主要超参数集中放在 cfg 中，方便记录到 wandb，也方便后续调参。
    cfg = dict(lr=1e-3, batch=128, epochs=5, hidden=256, seed=42)
    wandb.init(project="embai-day01", name="mnist-raw", config=cfg)
    torch.manual_seed(cfg["seed"])
    device = "cuda"

    # 先转成张量，再按 MNIST 的均值和标准差做标准化。
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # 下载并构建训练集 / 测试集。
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tfm)

    # DataLoader 负责按 batch 取数据；训练集需要打乱，pin_memory 有助于加速拷贝到 GPU。
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch"], shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(test_ds, batch_size=512, num_workers=2, pin_memory=True)

    # 初始化模型和优化器。
    model = MLP(cfg["hidden"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    step = 0
    for epoch in range(cfg["epochs"]):
        # 训练模式会启用诸如 Dropout / BatchNorm 的训练行为。
        model.train()
        for x, y in train_loader:
            # non_blocking=True 配合 pin_memory 使用时，GPU 拷贝通常更高效。
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # 前向传播得到分类 logits，并计算交叉熵损失。
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            # 标准的反向传播三步：清梯度 -> 反向传播 -> 参数更新。
            opt.zero_grad()
            loss.backward()
            opt.step()

            # 每隔 50 个 step 记录一次训练指标，避免日志过密。
            if step % 50 == 0:
                acc = (logits.argmax(-1) == y).float().mean().item()
                wandb.log({"train/loss": loss.item(), "train/acc": acc}, step=step)
            step += 1

        # 切换到评估模式，并在不计算梯度的情况下统计测试集准确率。
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(-1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        test_acc = correct / total
        wandb.log({"test/acc": test_acc, "epoch": epoch}, step=step)
        print(f"Epoch {epoch}: test acc = {test_acc:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
