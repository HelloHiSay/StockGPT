import os
import sys
import argparse
from datetime import timedelta
import joblib

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
plt.rcParams['axes.unicode_minus'] = False
for font in ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']:
    if font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
        plt.rcParams['font.sans-serif'] = [font]
        break
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import StockConfig
from models.models import StockGPT
from data_provider.stock_loader import StockDataset
from utils.metrics import mae, rmse

# ---------------------------
# 历史评估
# ---------------------------
def evaluate_history(model, dataset, device, output_dir):
    model.eval()
    X_test = torch.tensor(dataset.X, dtype=torch.float32).to(device)
    y_test = dataset.y  # shape=(N,1)

    preds = []
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]
            pred = model(batch)
            if pred.dim() == 3:
                pred = pred[:, -1, :]
            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # 反归一化
    preds_denorm = dataset.target_scaler.inverse_transform(preds)
    y_denorm = dataset.target_scaler.inverse_transform(y_test)

    # 日期
    df = pd.read_csv(dataset.file_path, parse_dates=['日期']).sort_values('日期')
    dates = df['日期'].values[-len(y_denorm):]

    mae_score = mae(preds_denorm, y_denorm)
    rmse_score = rmse(preds_denorm, y_denorm)

    print(f"\n历史评估结果: MAE={mae_score:.4f}, RMSE={rmse_score:.4f}")

    plt.figure(figsize=(14,6))
    plt.plot(dates, y_denorm, label="真实值")
    plt.plot(dates, preds_denorm, label="预测值")
    plt.title(f"历史预测 (MAE: {mae_score:.4f}, RMSE: {rmse_score:.4f})")
    plt.xlabel("日期")
    plt.ylabel("股价")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_history.png"), dpi=300)
    plt.close()

# ---------------------------
# 预测未来 N 天
# ---------------------------
# def predict_future(model, dataset, device, seq_len, days):
#     model.eval()
#     df = pd.read_csv(dataset.file_path, parse_dates=['日期']).sort_values('日期')
#
#     # 4 个特征
#     features = df[dataset.feature_cols].values
#
#     # 最近 seq_len 行
#     last_seq = features[-seq_len:]
#
#     # 使用加载的 scaler 归一化
#     norm_seq = dataset.feature_scaler.transform(last_seq)
#
#     current_seq = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(device)
#
#     preds = []
#     for _ in range(days):
#         with torch.no_grad():
#             pred = model(current_seq)
#             if pred.dim() == 3:
#                 pred = pred[:, -1, :]
#         next_price = pred.cpu().numpy()[0, 0]
#         preds.append(next_price)
#
#         # 将预测补入序列：4 维中只有“收盘”被预测
#         fake_next = np.array([[next_price, last_seq[-1,1], last_seq[-1,2], last_seq[-1,3]]])
#         fake_next_norm = dataset.feature_scaler.transform(fake_next)
#
#         next_input = torch.tensor(fake_next_norm, dtype=torch.float32).unsqueeze(0).to(device)
#         current_seq = torch.cat([current_seq[:, 1:, :], next_input], dim=1)
#
#     future_norm = np.array(preds).reshape(-1,1)
#     return dataset.target_scaler.inverse_transform(future_norm).flatten()

# ---------------------------
# 预测下一天收盘价（单步）
# ---------------------------
def predict_future(model, dataset, device, seq_len, days=1):
    assert days == 1, "本函数仅支持预测下一天"
    model.eval()
    df = pd.read_csv(dataset.file_path, parse_dates=['日期']).sort_values('日期')

    # 1. 取最后 seq_len 条原始数据
    raw = df[dataset.feature_cols].values[-seq_len:].astype(np.float32)
    # 成交量 log1p 与训练保持一致
    if '成交量' in dataset.feature_cols:
        raw[:, dataset.feature_cols.index('成交量')] = np.log1p(raw[:, dataset.feature_cols.index('成交量')])

    # 2. 归一化
    norm = dataset.feature_scaler.transform(raw)
    x = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(device)

    # 3. 单步预测
    with torch.no_grad():
        pred_norm = model(x)
        if pred_norm.dim() == 3:
            pred_norm = pred_norm[:, -1, :]
    next_price = dataset.target_scaler.inverse_transform(pred_norm.cpu().numpy()).item()

    return np.array([next_price])

# ---------------------------
# 保存未来预测
# ---------------------------
def save_predictions(prices, dataset, output_dir):
    df = pd.read_csv(dataset.file_path, parse_dates=['日期']).sort_values('日期')
    last_date = df['日期'].iloc[-1]
    future_dates = []
    current = last_date

    while len(future_dates) < len(prices):
        current += timedelta(days=1)
        if current.weekday() < 5:  # 工作日
            future_dates.append(current)

    result = pd.DataFrame({'日期': future_dates, '预测收盘价': prices})
    csv_path = os.path.join(output_dir, "future_predictions.csv")
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"未来预测结果已保存: {csv_path}")

    return result

# ---------------------------
# 可视化未来预测
# ---------------------------
def visualize_predictions(prices, dataset, output_dir):
    df = pd.read_csv(dataset.file_path, parse_dates=['日期']).sort_values('日期')

    # 只取最近 30 个交易日
    last_30 = df.tail(30)

    last_date = df['日期'].iloc[-1]

    # 构造未来日期
    future_dates = []
    current = last_date
    while len(future_dates) < len(prices):
        current += timedelta(days=1)
        if current.weekday() < 5:
            future_dates.append(current)

    # ----------- 绘图 ----------- #
    plt.figure(figsize=(14,6))

    # 最近 30 天历史
    plt.plot(last_30['日期'], last_30['收盘'], label="历史股价", linewidth=2)

    # 未来预测
    plt.plot(future_dates, prices, label="未来预测", linestyle='--')

    # 垂直分割线（历史 / 未来）
    plt.axvline(x=last_date, color='gray', linestyle=':')

    # 在未来曲线上标数值
    for d, p in zip(future_dates, prices):
        plt.text(d, p, f"{p:.2f}", fontsize=10, ha='center', va='bottom')

    plt.xlabel("日期")
    plt.ylabel("股价")
    plt.title("未来股价预测（最近 30 天）")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir,"future_predictions.png"), dpi=300)
    plt.close()

def visualize_truth_with_prediction(truth_path, pred_path, output_dir):
    """
    truth_path: 600519_data_truth.csv
    pred_path: future_predictions.csv
    """

    # --- 读取真实数据 ---
    truth_df = pd.read_csv(truth_path, parse_dates=['日期'])
    truth_df = truth_df.sort_values('日期')

    # --- 读取预测结果 ---
    pred_df = pd.read_csv(pred_path, parse_dates=['日期'])
    pred_date = pred_df['日期'].iloc[0]
    pred_price = pred_df['预测收盘价'].iloc[0]

    # --- 提取预测日前 30 天（含当天） ---
    mask = (truth_df['日期'] <= pred_date)
    last_30 = truth_df.loc[mask].tail(30)

    # 最后一天真实值（用于标注）
    last_date = last_30['日期'].iloc[-1]
    last_price = last_30['收盘'].iloc[-1]

    # --- 绘图 ---
    plt.figure(figsize=(14,6))

    # 历史 30 天真实收盘价
    plt.plot(last_30['日期'], last_30['收盘'], label="真实收盘价（近30天）", linewidth=2)

    # 🔵 标注真实值最后一天
    plt.scatter(last_date, last_price, color='blue', s=70)
    plt.text(last_date, last_price,
             f"{last_price:.2f}",
             fontsize=12, ha='right', va='bottom',
             color='blue')

    # 预测点（红色标记）
    plt.scatter(pred_date, pred_price, color='red', s=80, label="预测值")
    plt.text(pred_date, pred_price, f"{pred_price:.2f}",
             fontsize=12, ha='left', va='bottom', color='red')

    plt.title("真实收盘价（近30天）与预测值对比")
    plt.xlabel("日期")
    plt.ylabel("收盘价")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, "truth_vs_prediction.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"对比图已保存：{save_path}")

# ---------------------------
# 主函数
# ---------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_path = os.path.join(base_dir, "scripts", "600519_data.csv")
    checkpoint_path = os.path.join(base_dir, "checkpoints", "best_model.pth")
    feature_scaler_path = os.path.join(base_dir, "checkpoints", "feature_scaler.pkl")
    target_scaler_path  = os.path.join(base_dir, "checkpoints", "target_scaler.pkl")
    output_dir = os.path.join(base_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------
    # 加载 scaler
    # ---------------------------
    feature_scaler, target_scaler = StockDataset.load_scalers(feature_scaler_path, target_scaler_path)
    print("✅ scalers 加载完成")

    # ---------------------------
    # 加载模型
    # ---------------------------
    cfg = StockConfig()
    model = StockGPT(cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("✅ 模型加载完成")

    # ---------------------------
    # 加载数据集并替换 scalers
    # ---------------------------
    dataset = StockDataset(data_path, seq_len=cfg.seq_len)
    dataset.feature_scaler = feature_scaler
    dataset.target_scaler  = target_scaler

    # 历史预测
    evaluate_history(model, dataset, device, output_dir)

    # 未来预测
    future_prices = predict_future(model, dataset, device, cfg.seq_len, args.days)
    save_predictions(future_prices, dataset, output_dir)
    # visualize_predictions(future_prices, dataset, output_dir)

    # 未来预测文件路径
    pred_csv = os.path.join(output_dir, "future_predictions.csv")
    # 真实数据路径
    truth_csv = os.path.join(base_dir, "scripts", "600519_data_truth.csv")
    # 绘制真实 30 天与预测点对比
    visualize_truth_with_prediction(truth_csv, pred_csv, output_dir)

    print("\n✅ 预测任务完成！")
