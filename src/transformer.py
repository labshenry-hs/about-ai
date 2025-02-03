import torch
import torch.nn as nn
from src.attention import MultiHeadAttention
from src.feedforward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.ff(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        a1, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.drop(a1))
        a2, _ = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.drop(a2))
        return self.norm3(x + self.ff(x))

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048, dropout=0.1, max_len=512):
        super().__init__()
        from src.positional_encoding import PositionalEncoding
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        x = self.pos(self.embed(src))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.pos(self.embed(tgt))
        for layer in self.decoder:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc = self.encode(src, src_mask)
        return self.decode(tgt, enc, src_mask, tgt_mask)
