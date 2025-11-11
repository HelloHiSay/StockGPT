import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, file_path, seq_len=60):
        """
        股票数据集类：
        - 读取包含'日期'与'收盘'列的CSV文件
        - 生成定长时间序列样本用于模型训练与预测
        """
        # ✅ 保存文件路径，供预测脚本调用
        self.file_path = file_path

        # 读取 CSV，解析 '日期' 列为 datetime 类型
        df = pd.read_csv(file_path, parse_dates=['日期'])
        df = df.sort_values('日期').reset_index(drop=True)

        # 可选：重命名 '日期' 为 'Date'，保持一致性
        df.rename(columns={'日期': 'Date'}, inplace=True)

        # 取收盘价并归一化（国内股票列名通常为“收盘”）
        if '收盘' not in df.columns:
            raise KeyError("CSV 文件中未找到列 '收盘'，请确认数据格式正确。")

        data = df['收盘'].values.reshape(-1, 1)
        self.scaler = MinMaxScaler()
        data = self.scaler.fit_transform(data)

        # 保存序列长度
        self.seq_len = seq_len

        # 构建样本序列
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)

    def __len__(self):
        """返回样本数量"""
        return len(self.X)

    def __getitem__(self, idx):
        """返回一个样本 (输入序列, 目标值)"""
        return self.X[idx], self.y[idx]
