import torch
from dataclasses import dataclass
import os

@dataclass
class StockConfig:
    n_layer: int = 2          # Transformer Block 层数
    n_head: int = 8          # 注意力头数
    hidden_dim: int = 512    # embedding维度（即d_model）
    head_size: int = 64       # 每个注意力头的维度hidden_dim // n_head
    dropout: float = 0.3      # Dropout 概率
    block_size: int = 60      # 序列长度（最大输入长度） 要与seq_len一样，保持维度一致

    # 路径配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "600519_data.csv"))
    checkpoints = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))

    # 模型训练参数
    seq_len = 60
    batch_size = 64
    lr = 1e-4
    epochs = 100
    save_interval = 2
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
