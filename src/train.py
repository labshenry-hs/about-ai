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
