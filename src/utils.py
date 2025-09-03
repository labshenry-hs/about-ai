import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import math
def perplexity(loss): return math.exp(loss)

def accuracy_topk(logits, targets, k=5):
    topk = logits.topk(k, dim=-1).indices
    correct = topk.eq(targets.unsqueeze(-1)).any(-1)
    return correct.float().mean().item()

def get_linear_schedule(optimizer, num_warmup, num_training):
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < num_warmup: return float(current_step) / float(max(1, num_warmup))
        progress = float(current_step - num_warmup) / float(max(1, num_training - num_warmup))
        return max(0.0, 1.0 - progress)
    return LambdaLR(optimizer, lr_lambda)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience; self.min_delta = min_delta; self.best = float('inf'); self.count = 0; self.stop = False
    def __call__(self, val_loss):
        if val_loss < self.best - self.min_delta: self.best = val_loss; self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience: self.stop = True
        return self.stop
