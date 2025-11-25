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

class MixtureOfExperts(nn.Module):
    """Sparse MoE: route each token to top-k experts (Fedus 2022)."""
    def __init__(self, d_model, n_experts=8, top_k=2, d_ff=2048):
        super().__init__()
        self.top_k = top_k; self.n_exp = n_experts
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(d_model, d_ff) for _ in range(n_experts)])
    def forward(self, x):
        B, T, D = x.shape; x_flat = x.view(-1, D)
        logits = self.router(x_flat); gates, idx = torch.topk(torch.softmax(logits,-1), self.top_k, dim=-1)
        gates = gates / gates.sum(-1, keepdim=True)
        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.n_exp):
                mask = (idx[:, k] == e)
                if mask.any(): out[mask] += gates[mask, k:k+1] * self.experts[e](x_flat[mask])
        return out.view(B, T, D)

class ExpertChoiceRouting(nn.Module):
    """Expert-choice MoE: experts select top tokens (Zhou 2022)."""
    def __init__(self, d_model, n_experts=8, capacity=32, d_ff=2048):
        super().__init__()
        self.n_exp = n_experts; self.cap = capacity
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(d_model, d_ff) for _ in range(n_experts)])
    def forward(self, x):
        B, T, D = x.shape; x_flat = x.view(-1, D); N = x_flat.size(0)
        scores = torch.softmax(self.router(x_flat).T, dim=-1)
        _, indices = scores.topk(min(self.cap, N), dim=-1)
        out = torch.zeros_like(x_flat)
        for e, expert in enumerate(self.experts):
            idx = indices[e]; out[idx] += expert(x_flat[idx])
        return out.view(B, T, D)
