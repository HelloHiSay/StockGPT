import numpy as np

# 平均绝对误差
def mae(preds, targets):
    return np.mean(np.abs(preds - targets))

# 均方根误差
def rmse(preds, targets):
    return np.sqrt(np.mean((preds - targets) ** 2))