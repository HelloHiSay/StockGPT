# StockGPT

简化版 StockGPT 项目（GPT-style Transformer 用于股票价格预测）

快速开始（在项目根目录下）：
1. 创建虚拟环境并安装依赖：
   ```
   pip install -r requirements.txt
   ```
2. 下载 AAPL 数据（会保存到 C:\Users\39756\Desktop\课设Deeplearn\StockGPT\scripts\600519_data.csv）：
   ```
   python scripts/download_data.py
   ```
3. 训练模型（CPU模式）：
   ```
   python run.py
   ```
4. 预测并可视化（训练完成后）：
   ```
   python scripts/predict.py
   ```

文件说明：
- data_provider/stock_loader.py    数据集加载
- models/stock_gpt.py             GPT-style 模型实现
- exp/exp_stock.py                训练/实验逻辑
- utils/metrics.py                评估指标
- scripts/download_data.py        下载A股数据
- scripts/predict.py              预测与绘图脚本
- run.py                          训练入口脚本
