import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, file_path, seq_len=60):
        # 读取 CSV，解析 '日期' 列为 datetime
        df = pd.read_csv(file_path, parse_dates=['日期'])
        df = df.sort_values('日期').reset_index(drop=True)

        # 可选：重命名 '日期' 为 'Date'，保持一致性
        df.rename(columns={'日期': 'Date'}, inplace=True)

        # 取收盘价并归一化
        data = df['收盘'].values.reshape(-1, 1)  # 国内股票列名是中文 '收盘'
        self.scaler = MinMaxScaler()
        data = self.scaler.fit_transform(data)

        self.seq_len = seq_len
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
