import os
import torch
from exp.exp_stock import Exp_Stock
from config.config import Args



# 训练主流程
if __name__ == "__main__":
    os.makedirs(Args.checkpoints, exist_ok=True)
    print(f"使用设备: {Args.device}")
    print(f"读取数据: {Args.data_path}")

    # 初始化并训练模型
    exp = Exp_Stock(Args)
    exp.train()
