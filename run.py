import os
import torch
from exp.exp_stock import Exp_Stock

class Args:
    # 数据路径指向 scripts 文件夹下
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "600519_data.csv")
    
    seq_len = 60
    batch_size = 64
    lr = 1e-4
    epochs = 10
    
    # 使用 GPU，如果可用，否则回退 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoints = "checkpoints"

if __name__ == "__main__":
    # 创建保存模型的目录
    os.makedirs(Args.checkpoints, exist_ok=True)
    
    print(f"使用设备: {Args.device}")
    print(f"读取数据: {Args.data_path}")
    
    # 初始化实验对象
    exp = Exp_Stock(Args)
    
    # 开始训练
    exp.train()
