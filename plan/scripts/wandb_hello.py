import wandb
import time
import random

wandb.init(project="embai-day01", name="hello", config={"lr": 1e-3})
for step in range(50):
    wandb.log({"loss": 1.0 / (step + 1) + random.random() * 0.05}, step=step)
    time.sleep(0.05)
wandb.finish()
print("打开 wandb URL 查看曲线")
