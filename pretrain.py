import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_provider.stock_loader import StockDataset
from models.stock_gpt import StockGPT
from config.config import StockConfig      # ← 使用你的 StockConfig
from utils.early_stop import EarlyStopping


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():
    cfg = StockConfig()   # ← 实例化配置

    # ----------- 打印模型参数配置 -----------
    print("模型配置参数：")
    for k, v in cfg.__dict__.items():
        print(f"{k}: {v}")
    print("\n")

    # ----------- 1. 加载数据集（并划分训练/验证） -----------
    full_dataset = StockDataset(cfg.data_path, seq_len=cfg.seq_len)
    train_size = int(len(full_dataset) * 0.8)
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # ----------- 2. 初始化模型 -----------
    model = StockGPT(
        seq_len=cfg.seq_len,
        d_model=cfg.hidden_dim,
        nhead=cfg.n_head,
        num_layers=cfg.n_layer,
        dropout=cfg.dropout,
    ).to(cfg.device)

    # ----------- 打印模型参数总量 -----------
    total_params = count_parameters(model)
    print(f"模型总参数量: {total_params} ({total_params/1e6:.2f}M)\n")

    # ----------- 3. 损失函数 & 优化器 -----------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # ----------- 4. EarlyStopping 初始化 -----------
    os.makedirs(cfg.checkpoints, exist_ok=True)
    best_model_path = os.path.join(cfg.checkpoints, "best_model.pth")
    early_stopping = EarlyStopping(patience=10, verbose=True)

    # ----------- 记录 Loss 曲线 -----------
    train_losses = []
    val_losses = []

    print("开始训练模型...\n")
    for epoch in range(cfg.epochs):
        # =================== 训练 ===================
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(cfg.device)
            batch_y = batch_y.to(cfg.device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # =================== 验证 ===================
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(cfg.device)
                y_val = y_val.to(cfg.device)
                pred = model(x_val)
                loss = criterion(pred, y_val)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{cfg.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # =================== EarlyStopping 检查 ===================
        early_stopping(avg_val_loss, model, best_model_path)
        if early_stopping.early_stop:
            print("\n验证集长期无提升，触发 EarlyStopping，提前停止训练！")
            break

    print("\n训练结束！最佳模型已保存：", best_model_path)

    # ----------- 画 Loss 曲线 -----------
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.checkpoints, "loss_curve.png"), dpi=300)
    plt.close()
    print(f"Loss 曲线已保存到: {os.path.join(cfg.checkpoints, 'loss_curve.png')}")


if __name__ == "__main__":
    cfg = StockConfig()
    print(f"使用设备: {cfg.device}")
    print(f"数据路径: {cfg.data_path}\n")
    train()
