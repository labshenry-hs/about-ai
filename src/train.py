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

def generate(model, prompt_ids, max_new=50, device='cpu'):
    model.eval()
    ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max_new):
            from src.attention import make_causal_mask
            mask = make_causal_mask(ids.size(1), device)
            logits = model(ids, ids)
            next_id = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == 3: break
    return ids[0].tolist()

def generate_topk(model, prompt_ids, max_new=50, top_k=50, temp=1.0, device='cpu'):
    model.eval(); ids = torch.tensor(prompt_ids).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(ids, ids)[:, -1] / temp
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1:]] = -float('inf')
            probs = torch.softmax(logits, -1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], 1)
            if next_id.item() == 3: break
    return ids[0].tolist()

import os
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "loss": loss}, path)
def load_checkpoint(model, optimizer, path):
    ck = torch.load(path, map_location='cpu')
    model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
    return ck['epoch'], ck['loss']

def train_epoch_amp(model, loader, optimizer, scaler, device):
    model.train(); total = 0; crit = nn.CrossEntropyLoss(ignore_index=0)
    for batch in loader:
        src = batch[:, :-1].to(device); tgt = batch[:, 1:].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(src, src)
            loss = crit(out.reshape(-1, out.size(-1)), tgt.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        total += loss.item()
    return total / len(loader)

def beam_search(model, prompt_ids, beam_size=4, max_new=50, device='cpu'):
    model.eval()
    ids = torch.tensor(prompt_ids).unsqueeze(0).to(device)
    beams = [(ids, 0.0)]
    with torch.no_grad():
        for _ in range(max_new):
            candidates = []
            for seq, score in beams:
                logits = model(seq, seq)[:, -1]
                log_probs = torch.log_softmax(logits, -1)
                topk_lp, topk_ids = log_probs.topk(beam_size)
                for i in range(beam_size):
                    new_seq = torch.cat([seq, topk_ids[:, i:i+1]], 1)
                    candidates.append((new_seq, score + topk_lp[0, i].item()))
            beams = sorted(candidates, key=lambda x: -x[1])[:beam_size]
    return beams[0][0][0].tolist()

def nucleus_sampling(model, prompt_ids, p=0.9, temp=1.0, max_new=100, device='cpu'):
    model.eval(); ids = torch.tensor(prompt_ids).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(ids, ids)[:, -1] / temp
            probs = torch.softmax(logits, -1)
            sorted_p, sorted_idx = probs.sort(descending=True)
            cumsum = sorted_p.cumsum(-1)
            sorted_p[cumsum - sorted_p > p] = 0
            sorted_p /= sorted_p.sum()
            next_id = sorted_idx[0, torch.multinomial(sorted_p, 1)]
            ids = torch.cat([ids, next_id.view(1, 1)], 1)
    return ids[0].tolist()
