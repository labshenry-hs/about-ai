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

class BPETokenizer:
    """Byte-Pair Encoding tokenizer."""
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size; self.merges = {}; self.vocab = {}
    def _get_pairs(self, word):
        return set(zip(word[:-1], word[1:]))
    def train(self, texts):
        words = {}
        for t in texts:
            for w in t.split():
                w_tuple = tuple(list(w) + ['</w>'])
                words[w_tuple] = words.get(w_tuple, 0) + 1
        for _ in range(self.vocab_size):
            pairs = {}
            for word, freq in words.items():
                for p in self._get_pairs(list(word)):
                    pairs[p] = pairs.get(p, 0) + freq
            if not pairs: break
            best = max(pairs, key=pairs.get)
            self.merges[best] = len(self.merges)
