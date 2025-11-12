import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from data_provider.stock_loader import StockDataset
from models.stock_gpt import StockGPT
from utils.early_stop import EarlyStopping
from utils.metrics import mae, rmse
from config.config import StockConfig
import os
import matplotlib.pyplot as plt


class Exp_Stock:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        # 数据加载
        full_dataset = StockDataset(args.data_path, seq_len=args.seq_len)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False)

        # 模型配置
        config = StockConfig(block_size=args.seq_len)
        self.model = StockGPT(
            seq_len=args.seq_len,
            d_model=config.hidden_dim,
            dropout=config.dropout,
        ).to(self.device)

        print(f"模型结构配置：{config}")
        print(f"模型总参数量: {sum(p.numel() for p in self.model.parameters())}")

        # 优化器与损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss()

        # 早停与保存路径
        self.early_stopper = EarlyStopping(patience=10, verbose=True)
        self.checkpoint_path = os.path.join(args.checkpoints, "best_model.pth")

    def train_batch(self, X_batch, y_batch):
        self.model.train()
        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

        preds = self.model(X_batch).squeeze(-1)
        y_batch = y_batch.squeeze(-1)

        loss = self.criterion(preds, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                preds = self.model(X_batch).squeeze(-1)
                y_batch = y_batch.squeeze(-1)

                loss = self.criterion(preds, y_batch)
                total_loss += loss.item()

                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        preds_np = np.concatenate(all_preds)
        targets_np = np.concatenate(all_targets)
        return avg_loss, mae(preds_np, targets_np), rmse(preds_np, targets_np)

    def train(self):
        print(f"开始训练，共 {len(self.train_loader)} 个训练批次，{len(self.val_loader)} 个验证批次")
        train_losses, val_losses = [], []

        for epoch in range(1, self.args.epochs + 1):
            epoch_train_loss = [self.train_batch(X, y) for X, y in self.train_loader]
            avg_train_loss = np.mean(epoch_train_loss)
            train_losses.append(avg_train_loss)

            val_loss, val_mae, val_rmse = self.evaluate()
            val_losses.append(val_loss)

            print(f"Epoch {epoch}/{self.args.epochs} | "
                  f"Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f}")

            # 使用早停机制
            self.early_stopper(val_loss, self.model, self.checkpoint_path)

            if self.early_stopper.early_stop:
                print("⚠️ Early stopping 触发，训练终止。")
                break

        # 训练完成后绘制损失曲线
        self.plot_loss(train_losses, val_losses)

        # 最终确认模型保存
        if os.path.exists(self.checkpoint_path):
            print(f"✅ 最优模型已保存: {self.checkpoint_path}")
        else:
            print("⚠️ 未检测到模型保存，请检查 EarlyStopping 逻辑。")

    def plot_loss(self, train_losses, val_losses):
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss", linewidth=2)
        plt.plot(val_losses, label="Val Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        save_path = os.path.join(self.args.checkpoints, "loss_curve.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📈 Loss 曲线已保存到: {save_path}")
