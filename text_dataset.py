import random
from matplotlib.pyplot import text
import torch
import data_util
from pathlib import Path


def fucked(rnd: random.Random, indexer: data_util.SymbolIndexer, symbol_indices: torch.LongTensor) -> torch.LongTensor:
    N = len(symbol_indices)
    MASK_PROB = 0.15
    MASK_IDX = 0
    SWAP_PROB = 0.075

    symbol_indices = symbol_indices.clone()
    for idx in range(N):
        if rnd.random() < MASK_PROB:
            symbol_indices[idx] = MASK_IDX
        elif (idx + 1) < N and rnd.random() < SWAP_PROB:
            symbol_indices[idx], symbol_indices[idx+1] = symbol_indices[idx+1], symbol_indices[idx] 
    return symbol_indices


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_length: int, path: Path, indexer: data_util.SymbolIndexer):
        super().__init__()
        self.sequence_length = sequence_length
        self.indexer = indexer
        self.rnd = random.Random(0)

        self.data = []
        with open(path) as text_data:
            for line in text_data:
                self.data.extend(line.split())
        self.data = [s for s in self.data  if self.indexer.is_known(s)]

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx: int) -> torch.ByteTensor:
        if idx >= len(self):
            raise IndexError
        x = torch.LongTensor([self.indexer.to_index(s) for s in self.data[idx:idx + self.sequence_length]])
        return x, fucked(self.rnd, self.indexer, x)

if __name__  == "__main__":
    words, embeddings = data_util.load_glove("glove.txt", dim=100)
    indexer = data_util.SymbolIndexer(words)
    for elem in TextDataset(64, "train.txt", indexer):
        print(elem)
