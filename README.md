# StockGPT

StockGPT 项目（GPT-style Transformer 用于股票价格预测）

快速开始（在项目根目录下）：

1. 创建虚拟环境并安装依赖：
   ```
   pip install -r requirements.txt
   ```
2. 下载股票数据：
   ```
   python scripts/download_data.py
   ```
3. 训练模型：
   ```
   python pretrain.py
   ```
4. 预测并可视化（训练完成后）：
   ```
   python scripts/predict.py
   ```

文件说明：

* checkpoints                     权重保存目录
* config/config.py                训练参数、路径设置，模型配置参数


* data\_provider/stock\_loader.py   数据集加载和处理


* exp/exp\_stock.py                训练、验证流程以及模型保存逻辑


* models/models.py                模型实现
* models/positional_encoding.py   PE位置编码


* scripts/download\_data.py        下载A股数据
* scripts/predict.py              预测与绘图脚本


* utils/early\_stop.py             早停机制代码
* utils/metrics.py                评估指标


* pretrain.py                     训练入口脚本
* requirement.txt                 Python依赖包列表
