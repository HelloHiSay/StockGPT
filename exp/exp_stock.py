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

        #block_size要匹配seq_len
        config = StockConfig(block_size=args.seq_len)

        self.model = StockGPT(
            seq_len=args.seq_len,
            d_model=config.hidden_dim,
            dropout=config.dropout,
        ).to(self.device)

        print(f"模型结构配置：{config}")
        print(f"模型总参数量: {sum(p.numel() for p in self.model.parameters())}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss()

        self.early_stopper = EarlyStopping(patience=10, verbose=True)
        self.checkpoint_path = os.path.join(args.checkpoints, "best_model.pth")

    def train_batch(self, X_batch, y_batch):
        self.model.train()
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        predictions = self.model(X_batch).squeeze(-1)
        y_batch = y_batch.squeeze(-1)

        loss = self.criterion(predictions, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

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
        train_losses = []
        val_losses = []

        for epoch in range(self.args.epochs):
            epoch_train_loss = [self.train_batch(X_batch, y_batch) for X_batch, y_batch in self.train_loader]
            avg_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
            train_losses.append(avg_train_loss)

            val_loss, val_mae, val_rmse = self.evaluate()
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{self.args.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f}")

            self.early_stopper(val_loss, self.model, self.checkpoint_path)
            if self.early_stopper.early_stop:
                print("Early stopping 触发，训练终止。")
                break

            if val_loss == min(val_losses):
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, self.checkpoint_path)
                print(f"✅ 模型已保存到: {self.checkpoint_path}")

        self.plot_loss(train_losses, val_losses)

    def plot_loss(self, train_losses, val_losses):
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.args.checkpoints, "loss_curve.png"))
        plt.close()
