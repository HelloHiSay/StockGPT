import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager
matplotlib.use('Agg')  # 强制使用非交互式后端
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stock_gpt import StockGPT
from data_provider.stock_loader import StockDataset


# 配置中文字体（Windows系统）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 尝试设置中文字体
try:
    # Windows常见中文字体
    font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    
    # 找到第一个可用的中文字体
    chinese_font = None
    for font in font_list:
        if font in available_fonts:
            chinese_font = font
            break
    
    if chinese_font:
        plt.rcParams['font.sans-serif'] = [chinese_font]
        plt.rcParams['font.family'] = 'sans-serif'
        print(f"使用中文字体: {chinese_font}")
    else:
        # 如果没有找到，使用默认字体（可能不支持中文，但至少不会报错）
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
        print("警告: 未找到中文字体，中文可能显示为方框")
except Exception as e:
    print(f"字体配置警告: {e}")
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']

def main():
    # 配置参数（与训练时保持一致）
    seq_len = 60
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 数据路径
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "scripts", "600519_data.csv")
    
    # 检查点路径（使用最新的模型）
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_10.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"未找到检查点文件: {checkpoint_path}")
        print("请先运行训练脚本 (python run.py)")
        return
    
    # 初始化模型
    model = StockGPT(
        seq_len=seq_len,
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    # 加载检查点
    print(f"加载模型检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型已加载，训练损失: {checkpoint['loss']:.6f}")
    
    # 加载数据
    print(f"加载数据: {data_path}")
    dataset = StockDataset(data_path, seq_len=seq_len)
    
    # 准备测试数据（使用所有数据）
    X_test = torch.tensor(dataset.X, dtype=torch.float32).to(device)
    y_test = torch.tensor(dataset.y, dtype=torch.float32)
    
    # 进行预测
    print("开始预测...")
    predictions = []
    with torch.no_grad():
        # 批量预测以提高效率
        batch_size = 64
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            batch_pred = model(batch_X)
            
            # 处理预测形状
            if batch_pred.dim() == 3:
                batch_pred = batch_pred[:, -1, :]
            elif batch_pred.dim() == 2 and batch_pred.size(1) > 1:
                batch_pred = batch_pred[:, -1:]
            elif batch_pred.dim() == 1:
                batch_pred = batch_pred.unsqueeze(1)
            
            predictions.append(batch_pred.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    # 反归一化
    preds_denorm = dataset.scaler.inverse_transform(predictions)
    real_denorm = dataset.scaler.inverse_transform(dataset.y)
    
    # 计算评估指标
    from utils.metrics import mae, rmse
    mae_score = mae(preds_denorm, real_denorm)
    rmse_score = rmse(preds_denorm, real_denorm)
    
    print(f"\n评估指标:")
    print(f"MAE (平均绝对误差): {mae_score:.4f}")
    print(f"RMSE (均方根误差): {rmse_score:.4f}")
    
    # 可视化结果
    plt.figure(figsize=(14, 6))
    plt.plot(real_denorm, label='真实值', alpha=0.7, linewidth=1.5)
    plt.plot(preds_denorm, label='预测值', alpha=0.7, linewidth=1.5)
    plt.xlabel('时间步')
    plt.ylabel('股价')
    plt.title(f'StockGPT 股价预测结果 (MAE: {mae_score:.4f}, RMSE: {rmse_score:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "prediction_result.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n预测结果已保存到: {output_path}")
    
    plt.close()  # 关闭图形以释放内存

def predict_future(model, dataset, device, seq_len, days=7):
    """
    预测未来N天的股价
    :param model: 训练好的模型
    :param dataset: 数据集对象（包含scaler和原始数据）
    :param device: 设备
    :param seq_len: 序列长度
    :param days: 要预测的天数
    :return: 预测的股价列表（已反归一化）
    """
    model.eval()
    
    # 获取最后seq_len天的数据作为输入
    # 需要访问原始数据，我们需要重新加载
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  "scripts", "600519_data.csv"), 
                     parse_dates=['日期'])
    df = df.sort_values('日期').reset_index(drop=True)
    
    # 获取最后seq_len天的收盘价
    last_prices = df['收盘'].values[-seq_len:].reshape(-1, 1)
    
    # 归一化（使用训练时的scaler）
    last_prices_norm = dataset.scaler.transform(last_prices)
    
    # 转换为tensor
    current_sequence = torch.tensor(last_prices_norm, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, 1)
    
    future_predictions = []
    
    print(f"\n开始预测未来 {days} 天的股价...")
    with torch.no_grad():
        for day in range(days):
            # 预测下一步
            pred = model(current_sequence)
            
            # 处理预测形状
            if pred.dim() == 3:
                pred = pred[:, -1, :]
            elif pred.dim() == 2 and pred.size(1) > 1:
                pred = pred[:, -1:]
            elif pred.dim() == 1:
                pred = pred.unsqueeze(1)
            
            # 获取预测值
            pred_value = pred.cpu().numpy()[0, 0]
            future_predictions.append(pred_value)
            
            # 更新序列：移除第一个元素，添加预测值
            current_sequence = torch.cat([
                current_sequence[:, 1:, :],  # 移除第一个时间步
                pred.unsqueeze(0)  # 添加预测值
            ], dim=1)
            
            print(f"  第 {day+1} 天预测完成")
    
    # 反归一化
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_prices = dataset.scaler.inverse_transform(future_predictions).flatten()
    
    return future_prices

def save_future_predictions(future_prices, output_dir):
    """
    保存未来预测结果到CSV文件
    """
    # 获取最后日期
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  "scripts", "600519_data.csv"), 
                     parse_dates=['日期'])
    df = df.sort_values('日期').reset_index(drop=True)
    last_date = df['日期'].iloc[-1]
    
    # 生成未来日期（跳过周末，只计算交易日）
    future_dates = []
    current_date = last_date
    days_added = 0
    
    while len(future_dates) < len(future_prices):
        current_date += timedelta(days=1)
        # 简单假设：跳过周末（周六=5, 周日=6）
        if current_date.weekday() < 5:  # 周一到周五
            future_dates.append(current_date)
            days_added += 1
    
    # 创建DataFrame
    future_df = pd.DataFrame({
        '日期': future_dates,
        '预测收盘价': future_prices
    })
    
    # 保存到CSV
    output_path = os.path.join(output_dir, "future_predictions.csv")
    future_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n未来预测结果已保存到: {output_path}")
    print("\n未来7天股价预测:")
    print("=" * 50)
    for i, row in future_df.iterrows():
        print(f"{row['日期'].strftime('%Y-%m-%d')}: {row['预测收盘价']:.2f} 元")
    print("=" * 50)
    
    return future_df

def visualize_future_predictions(dataset, future_prices, output_dir):
    """
    可视化历史数据和未来预测
    """
    # 获取历史数据（最后100天用于展示）
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  "scripts", "600519_data.csv"), 
                     parse_dates=['日期'])
    df = df.sort_values('日期').reset_index(drop=True)
    
    # 获取最后100天的收盘价
    history_prices = df['收盘'].values[-100:]
    history_dates = df['日期'].values[-100:]
    
    # 生成未来日期
    last_date = df['日期'].iloc[-1]
    future_dates = []
    current_date = last_date
    
    while len(future_dates) < len(future_prices):
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:  # 周一到周五
            future_dates.append(current_date)
    
    # 绘图
    plt.figure(figsize=(16, 8))
    
    # 绘制历史数据
    plt.plot(history_dates, history_prices, label='历史股价', alpha=0.7, linewidth=2, color='blue')
    
    # 绘制未来预测
    plt.plot(future_dates, future_prices, label='未来预测', alpha=0.8, linewidth=2, 
             color='red', linestyle='--', marker='o', markersize=6)
    
    # 在连接点添加垂直线
    if len(history_dates) > 0 and len(future_dates) > 0:
        plt.axvline(x=last_date, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('股价 (元)', fontsize=12)
    plt.title('股票价格历史数据与未来7天预测', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, "future_predictions.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"未来预测可视化已保存到: {output_path}")
    
    plt.close()

if __name__ == '__main__':
    # 配置参数（与训练时保持一致）
    seq_len = 60
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 数据路径
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "scripts", "600519_data.csv")
    
    # 检查点路径（使用最新的模型）
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_10.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"未找到检查点文件: {checkpoint_path}")
        print("请先运行训练脚本 (python run.py)")
        exit(1)
    
    # 初始化模型
    model = StockGPT(
        seq_len=seq_len,
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    # 加载检查点
    print(f"加载模型检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型已加载，训练损失: {checkpoint['loss']:.6f}")
    
    # 加载数据
    print(f"加载数据: {data_path}")
    dataset = StockDataset(data_path, seq_len=seq_len)
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("第一部分：历史数据预测评估")
    print("="*60)
    
    # 执行历史数据预测（原有功能）
    main()
    
    print("\n" + "="*60)
    print("第二部分：未来7天股价预测")
    print("="*60)
    
    # 预测未来7天
    future_prices = predict_future(model, dataset, device, seq_len, days=7)
    
    # 保存预测结果
    future_df = save_future_predictions(future_prices, output_dir)
    
    # 可视化未来预测
    visualize_future_predictions(dataset, future_prices, output_dir)
    
    print("\n" + "="*60)
    print("所有预测任务完成！")
    print("="*60)
