import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config.config import StockConfig
from .positional_encoding import PositionalEncoding


# 单头注意力机制
class SingleHeadAttention(nn.Module):
    def __init__(self, head_size, dropout):
        super().__init__()
        self.scale = math.sqrt(head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        """
        q, k, v: (B, T, head_size)
        mask:    (1, 1, T, T)
        """
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        att = (q @ k.transpose(-2, -1)) / self.scale
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        return att @ v   # (B, T, head_size)


# 1. 多头注意力机制（合并 QKV + causal mask）
class MultiHeadAttention(nn.Module):
    def __init__(self, config: StockConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.hidden_dim // config.n_head
        self.hidden_dim = config.hidden_dim

        self.qkv = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)

        self.heads = nn.ModuleList([
            SingleHeadAttention(self.head_size, config.dropout)
            for _ in range(self.n_head)
        ])

        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        # 修改这里：mask 改为 2 维
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, self.n_head, 3 * self.head_size)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 直接切片 2 维 mask
        mask = self.mask[:T, :T]

        head_outputs = []
        for i in range(self.n_head):
            out_i = self.heads[i](q[:, i], k[:, i], v[:, i], mask)
            head_outputs.append(out_i)

        out = torch.cat(head_outputs, dim=-1)
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
            nn.Dropout(config.dropout)
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

    # def forward(self, x):
    #     x = x + self.att(self.ln1(x))
    #     x = x + self.ffn(self.ln2(x))
    #     return x

    def forward(self, x):
        x_ln1 = self.ln1(x)
        # print("Block ln1 输出 shape:", x_ln1.shape)  # 调试
        x = x + self.att(x_ln1)
        # print("Block att 输出 shape:", x.shape)  # 调试
        x_ln2 = self.ln2(x)
        # print("Block ln2 输出 shape:", x_ln2.shape)  # 调试
        x = x + self.ffn(x_ln2)
        # print("Block ffn 输出 shape:", x.shape)  # 调试
        return x


# 4. StockGPT 主模型
class StockGPT(nn.Module):
    def __init__(self, config: StockConfig = StockConfig()):
        super().__init__()
        self.seq_len = config.block_size
        self.hidden_dim = config.hidden_dim
        self.embedding = nn.Linear(4, config.hidden_dim)
        self.pos_encoder = PositionalEncoding(config.hidden_dim)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.fc_out = nn.Linear(config.hidden_dim, 1)  # 预测1天
        # self.fc_out = nn.Linear(config.hidden_dim, 5)  # 预测5天

    def forward(self, x):
        """
        输入: x: (batch_size, seq_len, 4)
        输出: (batch_size, 1)
        """
        x = self.embedding(x)        # (B, seq_len, hidden_dim)
        x = self.pos_encoder(x)      # 位置编码
        x = self.blocks(x)           # Transformer 层
        x = self.ln_f(x)
        x = x[:, -1, :]              # 取最后一个时间步
        out = self.fc_out(x)         # 输出预测值
        return out


if __name__ == "__main__":
    config = StockConfig()
    model = StockGPT(config)
    dummy_input = torch.randn(8, config.block_size, 4)
    out = model(dummy_input)
    print("Output shape:", out.shape)
