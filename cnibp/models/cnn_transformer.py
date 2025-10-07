import torch, torch.nn as nn
from .layers import PositionalEncoding, TemporalInception

class PPGTransformerRegressor(nn.Module):
    """
    CNN (Inception-style) frontend -> Transformer encoder -> GAP -> FC
    Meta (age, gender) is fused late.
    Outputs: [SBP, DBP]
    """
    def __init__(self, conv_channels=(64,128,256), d_model=256, nhead=4, nlayers=2, dropout=0.1):
        super().__init__()
        c1, c2, c3 = conv_channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.GELU()
        )
        self.inc1 = TemporalInception(32, c1)
        self.inc2 = TemporalInception(c1, c2)
        self.inc3 = TemporalInception(c2, c3)

        self.proj = nn.Conv1d(c3, d_model, kernel_size=1)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=4*d_model,
                                               dropout=dropout, batch_first=True,
                                               activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        # regression head (meta fused late)
        self.head = nn.Sequential(
            nn.Linear(d_model + 2, 128),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x, meta):
        # x: [B,1,T], meta: [B,2]
        y = self.stem(x)
        y = self.inc1(y)
        y = self.inc2(y)
        y = self.inc3(y)
        y = self.proj(y)                 # [B,D,T]
        y = y.transpose(1,2)             # [B,T,D]
        y = self.pos(y)
        y = self.encoder(y)              # [B,T,D]
        y = y.mean(dim=1)                # GAP over time -> [B,D]
        z = torch.cat([y, meta], dim=1)  # [B, D+2]
        out = self.head(z)               # [B,2] (SBP, DBP)
        return out
