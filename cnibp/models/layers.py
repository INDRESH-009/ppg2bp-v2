import math, torch, torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # [1, L, D]
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: [B,L,D]
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)

class TemporalInception(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        b = out_ch // 3
        self.b1 = nn.Conv1d(in_ch, b, kernel_size=3, padding=1)
        self.b2 = nn.Conv1d(in_ch, b, kernel_size=5, padding=2)
        self.b3 = nn.Conv1d(in_ch, b, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):       # [B,C,T]
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        y = self.bn(y)
        return self.act(y)
