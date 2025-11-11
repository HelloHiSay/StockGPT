import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config.config import StockConfig
from positional_encoding import PositionalEncoding

# -------------------------------
# 1. 单头注意力机制（带 causal mask）
# -------------------------------
class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.hidden_dim, config.head_size)
        self.query = nn.Linear(config.hidden_dim, config.head_size)
        self.value = nn.Linear(config.hidden_dim, config.head_size)
        self.head_size = config.head_size
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        """
        x: (B, T, C)
        mask: (1, T, T)
        """
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)

        # ✅ 加入 causal mask，防止模型看到未来信息
        if mask is not None:
            weights = weights.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v
        return out


# -------------------------------
# 2. 多头注意力机制
# -------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([
            SingleHeadAttention(config) for _ in range(config.n_head)
        ])
        self.proj = nn.Linear(config.n_head * config.head_size, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        # 所有 head 都共享 mask
        out = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


# -------------------------------
# 3. 前馈神经网络
# -------------------------------
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------
# 4. Transformer Block
# -------------------------------
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)

    def forward(self, x, mask=None):
        x = x + self.att(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x


# -------------------------------
# 5. StockGPT 主模型（带 causal mask + 位置编码）
# -------------------------------
class StockGPT(nn.Module):
    def __init__(self, seq_len=60, d_model=128, dropout=0.1, config: StockConfig = StockConfig()):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = d_model

        # 输入映射
        self.embedding = nn.Linear(1, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer Blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # 输出层
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, 1)

        # ✅ 注册下三角 mask（在训练和推理时重复使用）
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        )

    def forward(self, x):
        """
        输入: x: (batch_size, seq_len, 1)
        输出: (batch_size, 1)
        """
        B, T, _ = x.size()

        # 嵌入 + 位置编码
        x = self.embedding(x)
        x = self.pos_encoder(x)

        # 获取适配当前序列长度的 causal mask
        mask = self.causal_mask[:, :T, :T].to(x.device)

        # 逐层传递
        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_f(x)
        x = x[:, -1, :]  # 只取最后一个时间步的输出
        out = self.fc_out(x)
        return out
