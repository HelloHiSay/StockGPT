import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_provider.stock_loader import StockDataset
from models.stock_gpt import StockGPT
import os

class Exp_Stock:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
        # 加载数据
        self.train_dataset = StockDataset(args.data_path, seq_len=args.seq_len)
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        # 初始化模型
        self.model = StockGPT(
            seq_len=args.seq_len,
            d_model=128,
            nhead=8,
            num_layers=4,
            dropout=0.1
        ).to(self.device)
        
        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss()
        
    def train_batch(self, X_batch, y_batch):
        """训练一个批次"""
        X_batch = X_batch.to(self.device)  # (batch, seq_len, 1)
        y_batch = y_batch.to(self.device)  # (batch, 1)
        
        # 前向传播
        predictions = self.model(X_batch)
        
        # 处理预测形状：确保是 (batch, 1)
        # 如果模型返回了 (batch, seq_len, 1)，只取最后一个时间步
        if predictions.dim() == 3:
            predictions = predictions[:, -1, :]  # (batch, 1)
        # 如果返回了 (batch, seq_len)，只取最后一个
        elif predictions.dim() == 2 and predictions.size(1) > 1:
            predictions = predictions[:, -1:]  # (batch, 1)
        # 如果返回了 (batch,)，扩展为 (batch, 1)
        elif predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)
        
        # 确保 y_batch 形状是 (batch, 1)
        if y_batch.dim() == 3:
            y_batch = y_batch.squeeze(-1) if y_batch.size(-1) == 1 else y_batch[:, -1, :]
        elif y_batch.dim() == 1:
            y_batch = y_batch.unsqueeze(1)
        elif y_batch.dim() == 2 and y_batch.size(1) > 1:
            y_batch = y_batch[:, 0:1]  # 只取第一个特征
        
        # 最终检查：确保形状匹配
        assert predictions.shape == y_batch.shape, f"形状不匹配: predictions {predictions.shape} vs y_batch {y_batch.shape}"
        
        loss = self.criterion(predictions, y_batch)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self):
        """训练模型"""
        print(f"开始训练，共 {len(self.train_loader)} 个批次")
        
        for epoch in range(self.args.epochs):
            self.model.train()
            train_losses = []
            
            for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
                loss = self.train_batch(X_batch, y_batch)
                train_losses.append(loss)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.args.epochs} | Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {loss:.6f}")
            
            avg_loss = sum(train_losses) / len(train_losses)
            print(f"Epoch {epoch+1}/{self.args.epochs} | Average Loss: {avg_loss:.6f}")
            
            # 保存模型检查点
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(self.args.checkpoints, f"model_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"模型已保存到: {checkpoint_path}")
