import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

import torch.nn.functional as F

class SwiGLU(nn.Module):
    """SwiGLU activation FFN (Shazeer 2020) used in LLaMA."""
    def __init__(self, d_model, d_ff=None, dropout=0.0):
        super().__init__(); d_ff = d_ff or int(d_model * 8/3)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.w2(self.drop(F.silu(self.w1(x)) * self.w3(x)))
