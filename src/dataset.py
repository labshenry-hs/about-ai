import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_len=128):
        self.data = []
        for text in texts:
            ids = tokenizer.encode(text)[:max_len]
            self.data.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch):
    max_len = max(x.size(0) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :x.size(0)] = x
    return padded

def get_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
