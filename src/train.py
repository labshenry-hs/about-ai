import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    for batch in loader:
        src = batch[:, :-1].to(device)
        tgt = batch[:, 1:].to(device)
        optimizer.zero_grad()
        out = model(src, src)
        loss = criterion(out.reshape(-1, out.size(-1)), tgt.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.opt = optimizer; self.d = d_model; self.warmup = warmup_steps; self.step_num = 0
    def step(self):
        self.step_num += 1
        lr = self.d**-0.5 * min(self.step_num**-0.5, self.step_num * self.warmup**-1.5)
        for g in self.opt.param_groups: g['lr'] = lr
        self.opt.step()

def evaluate(model, loader, device):
    model.eval(); total = 0; criterion = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        for batch in loader:
            src = batch[:, :-1].to(device); tgt = batch[:, 1:].to(device)
            out = model(src, src)
            total += criterion(out.reshape(-1, out.size(-1)), tgt.reshape(-1)).item()
    return total / len(loader)
