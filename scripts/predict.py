import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import os
import sys
import argparse
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.stock_gpt import StockGPT
from data_provider.stock_loader import StockDataset
from utils.metrics import mae, rmse

# 中文字体配置
plt.rcParams['axes.unicode_minus'] = False
try:
    font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    for font in font_list:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            print(f"使用中文字体: {font}")
            break
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("⚠️ 未找到中文字体，使用默认字体")
except Exception as e:
    print(f"字体设置失败: {e}")

# 历史预测评估
def evaluate_history(model, dataset, device, output_dir):
    model.eval()
    X_test = torch.tensor(dataset.X, dtype=torch.float32).to(device)
    y_test = torch.tensor(dataset.y, dtype=torch.float32)

    predictions = []
    with torch.no_grad():
        for i in range(0, len(X_test), 64):
            batch = X_test[i:i+64]
            pred = model(batch)
            if pred.dim() == 3:
                pred = pred[:, -1, :]
            predictions.append(pred.cpu().numpy())

    preds = np.concatenate(predictions, axis=0)
    preds_denorm = dataset.scaler.inverse_transform(preds)
    real_denorm = dataset.scaler.inverse_transform(dataset.y)

    mae_score = mae(preds_denorm, real_denorm)
    rmse_score = rmse(preds_denorm, real_denorm)

    print(f"\n历史评估结果:")
    print(f"MAE: {mae_score:.4f} | RMSE: {rmse_score:.4f}")

    plt.figure(figsize=(14, 6))
    plt.plot(real_denorm, label="真实值")
    plt.plot(preds_denorm, label="预测值")
    plt.title(f"历史预测 (MAE: {mae_score:.4f}, RMSE: {rmse_score:.4f})")
    plt.xlabel("时间步")
    plt.ylabel("股价")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(output_dir, "prediction_result.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"历史预测图保存到: {path}")

# 预测未来 N 天
def predict_future(model, dataset, device, seq_len, days):
    df = pd.read_csv(dataset.file_path, parse_dates=['日期']).sort_values('日期')
    last_prices = df['收盘'].values[-seq_len:].reshape(-1, 1)
    norm_seq = dataset.scaler.transform(last_prices)
    current_seq = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(device)

    preds = []
    with torch.no_grad():
        for _ in range(days):
            pred = model(current_seq)
            if pred.dim() == 3:
                pred = pred[:, -1, :]
            preds.append(pred.cpu().numpy()[0, 0])
            current_seq = torch.cat([current_seq[:, 1:, :], pred.unsqueeze(0)], dim=1)

    future_norm = np.array(preds).reshape(-1, 1)
    return dataset.scaler.inverse_transform(future_norm).flatten()

# 保存预测到 CSV
def save_predictions(prices, dataset, output_dir):
    df = pd.read_csv(dataset.file_path, parse_dates=['日期']).sort_values('日期')
    last_date = df['日期'].iloc[-1]
    future_dates = []
    current = last_date
    while len(future_dates) < len(prices):
        current += timedelta(days=1)
        if current.weekday() < 5:
            future_dates.append(current)

    result_df = pd.DataFrame({
        '日期': future_dates,
        '预测收盘价': prices
    })

    csv_path = os.path.join(output_dir, "future_predictions.csv")
    result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n未来预测结果已保存到: {csv_path}")
    return result_df

# 绘图
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

    plt.figure(figsize=(14, 6))
    plt.plot(last_100['日期'], last_100['收盘'], label="历史股价", linewidth=2)
    plt.plot(future_dates, prices, label="未来预测", linestyle='--', marker='o')
    plt.axvline(x=last_date, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel("日期")
    plt.ylabel("股价 (元)")
    plt.title("未来股价预测")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(output_dir, "future_predictions.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"未来预测图保存到: {path}")

# 主逻辑
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, help="指定预测日期 (YYYY-MM-DD)")
    parser.add_argument('--days', type=int, default=10, help="预测未来多少天 (仅交易日)")
    args = parser.parse_args()

    seq_len = 60
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "scripts", "600519_data.csv")
    checkpoint_path = os.path.join(base_dir, "checkpoints", "best_model.pth")
    output_dir = os.path.join(base_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        print("❌ 未找到模型文件，请先训练")
        exit(1)

    print(f"加载模型检查点: {checkpoint_path}")
    model = StockGPT(seq_len, 128, 8, 4, 0.1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ 模型加载完成，训练损失: {checkpoint['loss']:.6f}")

    dataset = StockDataset(data_path, seq_len=seq_len)

    # 执行预测
    future_prices = predict_future(model, dataset, device, seq_len, args.days)
    future_df = save_predictions(future_prices, dataset, output_dir)

    if args.date:
        try:
            target = datetime.strptime(args.date, "%Y-%m-%d").date()
            row = future_df[future_df['日期'] == pd.Timestamp(target)]
            if not row.empty:
                price = row['预测收盘价'].values[0]
                print(f"\n📅 {args.date} 的预测收盘价为：{price:.2f} 元")
            else:
                print(f"\n❌ 日期 {args.date} 不在预测范围内，范围为：")
                print(future_df['日期'].dt.strftime('%Y-%m-%d').tolist())
        except ValueError:
            print("❌ 日期格式应为 YYYY-MM-DD")
    else:
        evaluate_history(model, dataset, device, output_dir)
        visualize_predictions(future_prices, dataset, output_dir)
        print("\n✅ 所有预测任务已完成！")
