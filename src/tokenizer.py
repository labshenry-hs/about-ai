import re
from collections import Counter
from typing import List, Dict

class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}

    def build_vocab(self, texts: List[str]):
        counter = Counter()
        for text in texts:
            counter.update(re.findall(r'\w+', text.lower()))
        for word, _ in counter.most_common(self.vocab_size - 4):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def encode(self, text: str) -> List[int]:
        tokens = re.findall(r'\w+', text.lower())
        return [self.word2idx.get(t, 1) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.idx2word.get(i, "<unk>") for i in ids)
