"""A reversible byte-level BPE tokenizer, trained only on training text."""
from collections import Counter
import json
from pathlib import Path


class BytePairTokenizer:
    def __init__(self, merges=None):
        self.merges = [tuple(pair) for pair in (merges or [])]
        self.tokens = {i: bytes([i]) for i in range(256)}
        for index, (left, right) in enumerate(self.merges, 256):
            self.tokens[index] = self.tokens[left] + self.tokens[right]

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    @staticmethod
    def merge(ids, pair, replacement):
        result, index = [], 0
        while index < len(ids):
            if index + 1 < len(ids) and (ids[index], ids[index+1]) == pair:
                result.append(replacement)
                index += 2
            else:
                result.append(ids[index])
                index += 1
        return result

    @classmethod
    def train(cls, text, vocab_size=384):
        if not 256 <= vocab_size <= 4096:
            raise ValueError('Vocabulary size must be between 256 and 4096')
        ids, merges = list(text.encode('utf-8')), []
        if not ids:
            raise ValueError('Cannot train a tokenizer on empty text')
        while 256 + len(merges) < vocab_size:
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break
            pair = max(counts, key=counts.get)
            if counts[pair] < 2:
                break
            ids = cls.merge(ids, pair, 256 + len(merges))
            merges.append(pair)
        return cls(merges)

    def encode(self, text):
        ids = list(text.encode('utf-8'))
        for index, pair in enumerate(self.merges, 256):
            ids = self.merge(ids, pair, index)
        return ids

    def decode(self, ids):
        return b''.join(self.tokens[index] for index in ids).decode('utf-8', errors='replace')

    def save(self, path):
        Path(path).write_text(json.dumps({'merges': self.merges}), encoding='utf-8')

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8'))['merges'])
