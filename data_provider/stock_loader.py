import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import torch


class StockDataset(Dataset):
    def __init__(self, file_path, seq_len=60,
                 feature_cols=None,
                 target_col='收盘',
                 use_log_volume=True):
        """
        - file_path: CSV 路径
        - seq_len: 序列长度
        - feature_cols: 输入特征列
        - target_col: 预测目标列
        - use_log_volume: 是否对成交量取log（强烈推荐）
        """

        self.seq_len = seq_len
        self.file_path = file_path

        # 默认使用 4 个特征
        if feature_cols is None:
            feature_cols = ['开盘', '收盘', '成交量', '换手率']

        self.feature_cols = feature_cols
        self.target_col = target_col

        # 加载 CSV
        df = pd.read_csv(file_path, parse_dates=['日期']).sort_values('日期')
        df = df.reset_index(drop=True)

        # 特征存在性检查
        for col in feature_cols + [target_col]:
            if col not in df.columns:
                raise KeyError(f"CSV 缺少列: {col}")

        # -------------------------
        # 特征预处理（log 成交量）
        # -------------------------
        df_features = df[feature_cols].copy()

        if use_log_volume and '成交量' in df_features.columns:
            # 防止成交量为0
            df_features['成交量'] = np.log1p(df_features['成交量'])

        # numpy array
        features = df_features.values.astype(np.float32)
        target = df[target_col].values.astype(np.float32).reshape(-1, 1)

        # -------------------------
        # 独立归一化：特征 scaler + 目标 scaler
        # -------------------------
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        features = self.feature_scaler.fit_transform(features)
        target = self.target_scaler.fit_transform(target)

        # -------------------------
        # 构建时间序列
        # X: (N, seq_len, feature_dim)
        # y: (N, 1)
        # -------------------------
        X, y = [], []
        for i in range(len(features) - seq_len):
            X.append(features[i:i + seq_len])
            y.append(target[i + seq_len])

        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)

    # --------------------------------
    # 保存 scaler（预测用）
    # --------------------------------
    def save_scalers(self, feature_path='feature_scaler.pkl', target_path='target_scaler.pkl'):
        joblib.dump(self.feature_scaler, feature_path)
        joblib.dump(self.target_scaler, target_path)

    # 加载 scaler（predict.py 用）
    @staticmethod
    def load_scalers(feature_path, target_path):
        feature_scaler = joblib.load(feature_path)
        target_scaler = joblib.load(target_path)
        return feature_scaler, target_scaler

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
