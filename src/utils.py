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
