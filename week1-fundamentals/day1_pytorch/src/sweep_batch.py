import subprocess
import textwrap


# 逐个测试不同 batch size 对单个 epoch 吞吐的影响。
BATCH_SIZES = [16, 32, 64, 128, 256, 512]


for batch_size in BATCH_SIZES:
    # 用子进程单独跑每个实验，避免不同实验之间互相污染状态。
    script = textwrap.dedent(
        f"""
        import time

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import wandb
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        wandb.init(
            project="embai-day01",
            name="bs-{batch_size}",
            config={{"batch": {batch_size}}},
            reinit=True,
        )

        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
        loader = DataLoader(
            ds,
            batch_size={batch_size},
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        ).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        torch.cuda.synchronize()
        t0 = time.time()
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        torch.cuda.synchronize()
        dt = time.time() - t0

        wandb.log({{"epoch_time_s": dt, "samples_per_s": len(ds) / dt}})
        wandb.finish()
        """
    )

    subprocess.run(["python", "-c", script], check=True)
