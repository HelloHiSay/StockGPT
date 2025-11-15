import os
import sys
import argparse
from datetime import datetime, timedelta

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
from models.stock_gpt import StockGPT
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
    preds_denorm = dataset.scaler.inverse_transform(preds)
    y_denorm = dataset.scaler.inverse_transform(y_test)

    # 日期
    df = pd.read_csv(dataset.file_path, parse_dates=['日期']).sort_values('日期')
    dates = df['日期'].values[-len(y_denorm):]  # 对齐最后 len(y_denorm) 个样本

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
    print(f"历史预测图已保存到: {output_dir}/prediction_history.png")
# ---------------------------
# 预测未来 N 天
# ---------------------------
def predict_future(model, dataset, device, seq_len, days):
    model.eval()
    df = pd.read_csv(dataset.file_path, parse_dates=['日期']).sort_values('日期')
    last_prices = df['收盘'].values[-seq_len:].reshape(-1, 1)
    norm_seq = dataset.scaler.transform(last_prices)
    current_seq = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(device)

    preds = []
    for _ in range(days):
        with torch.no_grad():
            pred = model(current_seq)
            if pred.dim() == 3:
                pred = pred[:, -1, :]
        preds.append(pred.cpu().numpy()[0,0])
        next_input = torch.tensor(dataset.scaler.transform(np.array([[preds[-1]]])), dtype=torch.float32).unsqueeze(0).to(device)
        current_seq = torch.cat([current_seq[:,1:,:], next_input], dim=1)

    future_norm = np.array(preds).reshape(-1,1)
    return dataset.scaler.inverse_transform(future_norm).flatten()

# ---------------------------
# 保存预测结果
# ---------------------------
def save_predictions(prices, dataset, output_dir):
    df = pd.read_csv(dataset.file_path, parse_dates=['日期']).sort_values('日期')
    last_date = df['日期'].iloc[-1]
    future_dates = []
    current = last_date
    while len(future_dates) < len(prices):
        current += timedelta(days=1)
        if current.weekday() < 5:  # 只保留周一到周五
            future_dates.append(current)

    result_df = pd.DataFrame({'日期': future_dates, '预测收盘价': prices})
    csv_path = os.path.join(output_dir, "future_predictions.csv")
    result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"未来预测结果已保存到: {csv_path}")
    return result_df

# ---------------------------
# 绘图
# ---------------------------
def visualize_predictions(prices, dataset, output_dir):
    df = pd.read_csv(dataset.file_path, parse_dates=['日期']).sort_values('日期')
    last_100 = df.tail(100)
    last_date = df['日期'].iloc[-1]

    future_dates = []
    current = last_date
    while len(future_dates) < len(prices):
        current += timedelta(days=1)
        if current.weekday() < 5:
            future_dates.append(current)

    plt.figure(figsize=(14,6))
    plt.plot(last_100['日期'], last_100['收盘'], label="历史股价", linewidth=2)
    plt.plot(future_dates, prices, label="未来预测", linestyle='--', marker='o')
    plt.axvline(x=last_date, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel("日期")
    plt.ylabel("股价")
    plt.title("未来股价预测")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"future_predictions.png"), dpi=300)
    plt.close()
    print(f"未来预测图已保存到: {output_dir}/future_predictions.png")

# ---------------------------
# 主函数
# ---------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=1, help="预测未来交易日数量")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "scripts", "600519_data.csv")
    checkpoint_path = os.path.join(base_dir, "checkpoints", "best_model.pth")
    output_dir = os.path.join(base_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        print("❌ 未找到模型文件，请先训练")
        exit(1)

    # ---------------------------
    # 加载模型
    # ---------------------------
    cfg = StockConfig()
    model = StockGPT(seq_len=cfg.seq_len, d_model=cfg.hidden_dim,
                     nhead=cfg.n_head, num_layers=cfg.n_layer,
                     dropout=cfg.dropout).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 判断 checkpoint 类型
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"✅ 模型加载完成")

    # ---------------------------
    # 加载数据集
    # ---------------------------
    dataset = StockDataset(data_path, seq_len=cfg.seq_len)

    # 历史评估
    evaluate_history(model, dataset, device, output_dir)

    # 未来预测
    future_prices = predict_future(model, dataset, device, cfg.seq_len, args.days)
    save_predictions(future_prices, dataset, output_dir)
    visualize_predictions(future_prices, dataset, output_dir)
    print("\n✅ 预测任务完成！")
