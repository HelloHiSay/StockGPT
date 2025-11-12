import akshare as ak
import pandas as pd
import time
import os

def download_stock_data(symbol="600519", adjust="qfq", max_retries=3, retry_delay=5):
    """
    下载国内 A 股历史数据并保存到当前 StockGPT 文件夹
    """
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date="20000101",
                end_date="20251101",
                adjust=adjust
            )

            if df is None or df.empty:
                print(f"[尝试 {attempt}] 未获取到数据，等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                continue

            # 保存到 StockGPT 根目录
            current_folder = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_folder, f"{symbol}_data.csv")

            df.to_csv(file_path, index=False)
            print(f"已成功下载 {symbol} 数据，保存到 {file_path}")
            return df

        except Exception as e:
            print(f"[尝试 {attempt}] 获取数据失败：{e}")
            time.sleep(retry_delay)

    print(f"多次尝试后仍未获取到 {symbol} 数据，请检查网络或接口。")
    return None


if __name__ == "__main__":
    download_stock_data(symbol="600519", adjust="qfq")
