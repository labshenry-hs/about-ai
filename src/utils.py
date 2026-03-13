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

def cosine_similarity_matrix(a, b):
    import torch.nn.functional as F
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
    return torch.mm(a, b.T)

def moving_average(values, window=10):
    return [sum(values[max(0,i-window):i+1])/len(values[max(0,i-window):i+1]) for i in range(len(values))]

def compute_bleu(reference, hypothesis, n=4):
    from collections import Counter
    import math
    score = 0.0
    for k in range(1, n+1):
        ref_ngrams = Counter(zip(*[reference[i:] for i in range(k)]))
        hyp_ngrams = Counter(zip(*[hypothesis[i:] for i in range(k)]))
        overlap = sum((hyp_ngrams & ref_ngrams).values())
        total = sum(hyp_ngrams.values())
        score += math.log(overlap / total + 1e-10) if total > 0 else 0
    bp = min(1.0, math.exp(1 - len(reference) / (len(hypothesis) + 1e-10)))
    return bp * math.exp(score / n)

def load_model(path, model, device='cpu'):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model']); return model, state.get('step', 0)

def quantize_linear(module, bits=8):
    """Post-training static quantization helper."""
    import copy; m = copy.deepcopy(module)
    for name, layer in m.named_modules():
        if isinstance(layer, torch.nn.Linear):
            w = layer.weight.data
            scale = w.abs().max() / (2**(bits-1)-1)
            layer.weight.data = (w / scale).round().clamp(-(2**(bits-1)), 2**(bits-1)-1) * scale
    return m

def export_onnx(model, dummy_input, path='model.onnx'):
    import torch.onnx
    torch.onnx.export(model, dummy_input, path, opset_version=17,
                      input_names=['input_ids'], output_names=['logits'],
                      dynamic_axes={'input_ids':{0:'batch',1:'seq'},'logits':{0:'batch',1:'seq'}})
    return path
