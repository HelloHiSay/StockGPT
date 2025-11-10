import torch
import torch.nn as nn

class StockGPT(nn.Module):
    def __init__(self, seq_len=60, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=dropout, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        x = self.embedding(x)           # -> (batch, seq_len, d_model)
        x = x.permute(1, 0, 2)          # Transformer expects (S, N, E)
        out = self.transformer(x)       # (S, N, E)
        out = out.permute(1, 0, 2)      # (N, S, E)
        return self.fc_out(out[:, -1, :])  # predict next step