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
   python -m scripts.predict
   ```

项目结构：

```

StockGPT/
├── pretrain.py                # 预训练入口
├── README.md                  # 项目说明
├── requirements.txt           # 依赖列表
├── checkpoints/               # 模型权重与训练曲线
│   ├── best\_model.pth         # 当前最优权重
│   ├── best\_model\_1.pth       # 历史最优权重（备份）
│   └── loss\_curve.png          # 训练 loss 曲线图
├── config/                    # 配置文件
│   └── config.py               # 超参、路径、模型规模
├── data\_provider/             # 数据加载与预处理
│   └── stock\_loader.py         # 返回 train/valid/test DataLoader
├── exp/                       # 实验流程封装（训练/验证/测试）
│   └── exp\_stock.py            # StockGPT 实验类，主逻辑
├── models/                    # 模型定义
│   ├── models.py               # 通用模型骨架
│   ├── stock\_gpt.py            # StockGPT 具体结构（Transformer + 股票特征）
│   └── positional\_encoding.py  # 位置编码（时间步/交易日）
├── results/                   # 推理输出
│   ├── future\_predictions.csv  # 未来 N 日预测值
│   ├── future\_predictions.png  # 预测曲线图
│   └── prediction\_history.png  # 历史回测图
├── scripts/                   # 数据与脚本工具
│   ├── download\_data.py        # 自动下载股票日线数据
│   ├── predict.py              # 单脚本快速推理（加载权重直接出图）
│   ├── 600519\_data.csv         # 示例原始数据
│   ├── 600519\_data\_contrast.csv# 预测 vs 真实对比
│   └── 600519\_data\_truth.csv   # 真实标签（用于回测）
└── utils/                     # 工具箱
├── early\_stop.py           # 早停策略
└── metrics.py              # 评价指标（MAE、RMSE）
└── structure.png          # 模型结构图
```

# 模型结构

StockGPT 模型维度流水（单个 Transformer Block）:

- Inputs:            (64, 60, 1)          # 原始输入序列，batch=64, 序列长度=60, 特征=1
- Embedding:         (64, 60, 512)        # 输入经过线性 embedding 层，将特征维度映射到 d_model=512
- Multi-Head Attention (MHA):
  - Q, K, V:         (64, 60, 512)        # 计算 Q,K,V 张量
  - Heads:           (64, 8, 60, 64)      # 拆分成 8 个头，每个 head 维度 head_dim=64
  - Output:          (64, 60, 512)        # 多头注意力 concat 后，再投影回 d_model=512
- Feed-Forward Network (FFN):
  - Expand:          (64, 60, 2048)       # 先将 hidden_dim 扩展 4 倍
  - Reduce:          (64, 60, 512)        # 再降回 hidden_dim=512
- Prediction:        (64, 60, 1)          # 输出股价预测，单特征

## 1️⃣ Inputs: `(64, 60, 1)`

* **64** → Batch size，一次输入模型的样本数量。
* **60** → 序列长度，模型一次看到 60 天的历史数据（`seq_len=60`）。
* **1** → 输入特征数，这里我们只用收盘价作为特征。

**解释**：每个 batch 包含 64 条序列，每条序列 60 天，每天 1 个特征。

---

## 2️⃣ Embedding: `(64, 60, 512)`

* 原始输入 `(64,60,1)` 通过线性层或 embedding 投影到 `hidden_dim=512`。
* 每个时间步的特征被映射到 512 维向量，以便 Transformer 后续处理。

**解释**：Transformer 需要固定维度 `d_model`，所以把输入扩展到 512 维。

---

## 3️⃣ MHA Q,K,V: `(64, 60, 512)`

* Multi-Head Attention 会生成 **查询（Q）、键（K）、值（V）**，每个都是 `(batch, seq_len, hidden_dim)`。
* 这里 `hidden_dim=512`，和 embedding 一致。

**解释**：每个时间步有一个 512 维的向量作为 Q、K、V，用于计算注意力分数。

---

## 4️⃣ Heads: `(64, 8, 60, 64)`

* 注意力头数 `n_head=8`，每个头的维度 `head_size = hidden_dim / n_head = 64`。
* 在计算多头注意力时，会把 `(64, 60, 512)` 拆分成 8 个头，每个头 64 维：
  `(B, seq_len, hidden_dim) → (B, n_head, seq_len, head_size)`

**解释**：多头注意力机制允许模型关注序列中不同的子空间特征。

---

## 5️⃣ MHA Output: `(64, 60, 512)`

* 每个头计算注意力后，输出再次拼接（concat）回 `hidden_dim=512`。
* 拼接后通常再经过一个线性投影层，保证维度仍然是 512。

**解释**：拼接所有头的输出，再映射回原始 hidden\_dim，为下一层做准备。

---

### 6️⃣ FFN: `(64, 60, 2048) → (64, 60, 512)`

* FFN (Feed-Forward Network) 每个时间步独立计算：
  * 先线性扩展到 4×hidden\_dim = 2048
  * 再线性降回 hidden\_dim = 512
* `(64, 60, 512) → (64, 60, 2048) → (64, 60, 512)`

**解释**：FFN 增加非线性能力，使每个时间步的表示更丰富，同时保持与隐藏维度一致。

---

## 7️⃣ Prediction: `(64, 60, 1)`

* 最后一个线性层将 hidden\_dim=512 映射回原始特征维度 1（股价预测）。
* 输出与输入序列长度保持一致 `(seq_len=60)`，每个时间步预测一个股价。

**解释**：模型的输出形状对应输入序列，每天一个预测收盘价。

---

## 总结

* **第一维**：batch size，训练时固定为 64
* **第二维**：序列长度，决定 Transformer 看到多少历史天数
* **第三维**：特征维度或隐藏维度，根据不同模块变化
* **多头注意力**：拆成 `(B, n_head, seq_len, head_size)`，然后再 concat 回 hidden\_dim
* **FFN**：扩维再降维，增强每个时间步的表示能力
* **最终预测**：回到原始特征维度，生成每天股价
