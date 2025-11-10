import numpy as np

def mae(preds, targets):
    return np.mean(np.abs(preds - targets))

def rmse(preds, targets):
    return np.sqrt(np.mean((preds - targets) ** 2))