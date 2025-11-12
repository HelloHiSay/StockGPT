import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config.config import StockConfig, Args
from positional_encoding import PositionalEncoding


# 1. 多头注意力机制（合并 QKV + causal mask）
class MultiHeadAttention(nn.Module):
    def __init__(self, config: StockConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.hidden_dim // config.n_head
        self.hidden_dim = config.hidden_dim
        self.dropout = nn.Dropout(config.dropout)

        # 合并 Q, K, V
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # 预生成 causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .unsqueeze(0)
            .unsqueeze(0)  # (1, 1, T, T)
        )

    def forward(self, x):
        B, T, C = x.shape  # Batch, Seq_len, Hidden_dim

        # 1. 一次性计算 Q,K,V 并 reshape 为多头形式
        qkv = self.qkv(x)  # (B, T, 3*C)
        qkv = qkv.view(B, T, self.n_head, 3 * self.head_size)
        q, k, v = qkv.chunk(3, dim=-1)  # (B, T, n_head, head_size)

        q = q.transpose(1, 2)  # (B, n_head, T, head_size)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 2. 注意力得分
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)

        # 3. causal mask，防止看到未来信息
        mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # 4. 加权求和 + 合并 heads
        out = att @ v  # (B, n_head, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)

        # 5. 输出投影
        out = self.proj(out)
        out = self.dropout(out)
        return out


# 2. 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, config: StockConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


# 3. Transformer Block
class Block(nn.Module):
    def __init__(self, config: StockConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# 4. StockGPT 主模型
class StockGPT(nn.Module):
    def __init__(self, config: StockConfig = StockConfig()):
        super().__init__()

        self.seq_len = config.block_size
        self.hidden_dim = config.hidden_dim

        self.embedding = nn.Linear(1, config.hidden_dim)
        self.pos_encoder = PositionalEncoding(config.hidden_dim)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.fc_out = nn.Linear(config.hidden_dim, 1)

    def forward(self, x):
        """
        输入: x: (batch_size, seq_len, 1)
        输出: (batch_size, 1)
        """
        x = self.embedding(x)        # (B, seq_len, hidden_dim)
        x = self.pos_encoder(x)      # 位置编码
        x = self.blocks(x)           # Transformer 层
        x = self.ln_f(x)
        x = x[:, -1, :]              # 取最后一个时间步
        out = self.fc_out(x)         # 输出预测值
        return out


# ✅ 示例：加载配置并实例化模型
if __name__ == "__main__":
    config = StockConfig()
    model = StockGPT(config)
    dummy_input = torch.randn(8, config.block_size, 1)
    out = model(dummy_input)
    print("Output shape:", out.shape)
    print("Device:", Args.device)
