import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        B = q.shape[0]
        q = self.split_heads(self.W_q(q))
        k = self.split_heads(self.W_k(k))
        v = self.split_heads(self.W_v(v))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.W_o(out), attn

def make_causal_mask(size, device):
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return ~mask

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang 2019)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight

class GroupedQueryAttention(nn.Module):
    """GQA: fewer KV heads than Q heads (Ainslie 2023)."""
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads; self.n_kv = n_kv_heads; self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model); self.drop = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        B, T, _ = x.shape; g = self.n_heads // self.n_kv
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_kv, self.d_k).transpose(1, 2).repeat_interleave(g, dim=1)
        v = self.W_v(x).view(B, T, self.n_kv, self.d_k).transpose(1, 2).repeat_interleave(g, dim=1)
        scores = torch.matmul(q, k.transpose(-2,-1)) / self.d_k**0.5
        if mask is not None: scores = scores.masked_fill(~mask, -1e9)
        out = torch.matmul(self.drop(torch.softmax(scores,-1)), v)
        return self.W_o(out.transpose(1,2).reshape(B,T,-1))

class KVCacheAttention(nn.Module):
    """Attention with KV cache for fast autoregressive inference."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads; self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model); self.out = nn.Linear(d_model, d_model)
    def forward(self, x, past_kv=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        attn = torch.softmax(q @ k.transpose(-2,-1) / self.d_k**0.5, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out), (k, v)

class AttentionWithDropPath(nn.Module):
    """Stochastic depth / drop path for attention layers."""
    def __init__(self, d_model, n_heads, drop_path=0.1, attn_drop=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, attn_drop)
        self.drop_path_prob = drop_path
    def forward(self, x, mask=None):
        out, w = self.attn(x, x, x, mask)
        if self.training and self.drop_path_prob > 0:
            keep = torch.rand(x.shape[0], 1, 1, device=x.device) > self.drop_path_prob
            out = out * keep.float() / (1 - self.drop_path_prob)
        return x + out, w

class LoRALinear(nn.Module):
    """LoRA: Low-Rank Adaptation for efficient fine-tuning (Hu 2021)."""
    def __init__(self, linear, r=8, alpha=16):
        super().__init__(); self.linear = linear
        for p in linear.parameters(): p.requires_grad_(False)
        import math; d_in, d_out = linear.in_features, linear.out_features
        self.A = nn.Parameter(torch.randn(r, d_in) / math.sqrt(r))
        self.B = nn.Parameter(torch.zeros(d_out, r)); self.scale = alpha / r
    def forward(self, x):
        return self.linear(x) + x @ self.A.T @ self.B.T * self.scale

class SlidingWindowAttention(nn.Module):
    """Local attention with sliding window of size w (Beltagy 2020)."""
    def __init__(self, d_model, n_heads, window=256):
        super().__init__(); self.window = window
        self.attn = MultiHeadAttention(d_model, n_heads)
    def forward(self, x, mask=None):
        B, T, D = x.shape; w = self.window; outputs = []
        for start in range(0, T, w):
            end = min(start + w, T)
            ctx_start = max(0, start - w // 2)
            chunk = x[:, ctx_start:end]
            out, _ = self.attn(x[:, start:end], chunk, chunk)
            outputs.append(out)
        return torch.cat(outputs, dim=1)
