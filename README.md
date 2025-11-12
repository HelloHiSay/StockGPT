# StockGPT

StockGPT 项目（GPT-style Transformer 用于股票价格预测）

快速开始（在项目根目录下）：

1. 创建虚拟环境并安装依赖：
   ```
   pip install -r requirements.txt
   ```
2. 下载 AAPL 数据：
   ```
   python scripts/download_data.py
   ```
3. 训练模型：
   ```
   python run.py
   ```
4. 预测并可视化（训练完成后）：
   ```
   python scripts/predict.py
   ```

文件说明：

- data_provider/stock_loader.py   数据集加载
- models/models.py                模型实现
- exp/exp_stock.py                训练/实验逻辑
- utils/metrics.py                评估指标
- scripts/download_data.py        下载A股数据
- scripts/predict.py              预测与绘图脚本
- run.py                          训练入口脚本
