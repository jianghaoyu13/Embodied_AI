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
        # 保存构造参数，后面可以通过 self.hparams 访问，也便于日志记录和断点恢复。
        self.save_hyperparameters()

        # 模型结构与 raw 版本一致：展平后接两层隐藏层，最后输出 10 类 logits。
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

    def training_step(self, batch, _):
        # Lightning 会自动从 DataLoader 中取出一个 batch 传进来。
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(-1) == y).float().mean()

        # self.log_dict 会把指标交给 Lightning 的日志系统，由 logger 统一写出。
        self.log_dict({"train/loss": loss, "train/acc": acc}, prog_bar=True)

        # 返回 loss 后，Lightning 会自动完成 backward / optimizer.step() / zero_grad()。
        return loss

    def validation_step(self, batch, _):
        # 验证阶段的逻辑单独写在 validation_step，Trainer 会自动在验证流程中调用。
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(-1) == y).float().mean()
        self.log_dict({"val/loss": loss, "val/acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        # 把优化器的配置交给 Lightning，Trainer 会负责在训练中使用它。
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


def main():
    # 把主要超参数集中放在 cfg 中，尽量与 raw 版本保持一致，方便公平对比。
    cfg = dict(lr=1e-3, batch=128, epochs=5, hidden=256, seed=42)
    L.seed_everything(cfg["seed"])

    # 数据预处理与 raw 版本相同：转张量后做标准化。
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # 下载并构建训练集 / 测试集。
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tfm)

    # DataLoader 依然需要我们自己定义，但训练和验证循环不再手写。
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch"], shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(test_ds, batch_size=512, num_workers=2, pin_memory=True)

    # WandbLogger 接入 Lightning 的日志体系，不需要手动 wandb.log / wandb.finish。
    logger = WandbLogger(project="embai-day01", name="mnist-lightning", config=cfg)

    # Trainer 统一管理 epoch 数、设备、日志频率，以及 fit/validate 的完整流程。
    trainer = L.Trainer(
        max_epochs=cfg["epochs"],
        accelerator="gpu",
        devices=1,
        logger=logger,
        log_every_n_steps=50,
    )
    trainer.fit(LitMNIST(hidden=cfg["hidden"], lr=cfg["lr"]), train_loader, test_loader)


if __name__ == "__main__":
    main()
