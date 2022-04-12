from symtable import Symbol
import torch
from pathlib import Path
from typing import Dict, Iterable, Optional, List, Iterable, Tuple
from torch import Tensor


def load_glove(path: Path, dim: int) -> Tuple[List[str], Tensor]:
    with open(path) as glove_data:
        words: List[str] = []
        vecs: List[List[float]] = []
        for line in glove_data:
            tokens = line.split()
            words.append(tokens[0])
            vecs.append([float(elem) for elem in tokens[1:]])

        embeddings: Tensor = torch.empty(len(words), dim)
        for idx, vec in enumerate(vecs):
            tokens = line.split()
            embeddings[idx] = torch.tensor(vec)

    return words, embeddings


class SymbolIndexer:
    _known_symbol_to_index: Dict[str, int]
    _index_to_known_symbol: Dict[int, str]
    _size: int

    def _add_symbol(self, symbol: str):
        self._known_symbol_to_index[symbol] = self._size
        self._index_to_known_symbol[self._size] = symbol
        self._size += 1

    def __init__(self, words: Iterable[str]):
        self._size = 0
        self._known_symbol_to_index = {}
        self._index_to_known_symbol = {}

        for word in words:
            self._add_symbol(word)

    def size(self) -> int:
        return self._size

    def to_index(self, symbol: str) -> int:
        return self._known_symbol_to_index[symbol]

    def to_symbol(self, index: int) -> Optional[str]:
        return self._index_to_known_symbol[index] if index in self._index_to_known_symbol else None

    def is_known(self, symbol: str) -> int:
        return symbol in self._known_symbol_to_index

if __name__  == "__main__":
    words, embeddings = load_glove("glove.txt", dim=100)
    print(SymbolIndexer(words)._index_to_known_symbol)