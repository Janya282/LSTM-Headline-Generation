import json
import os
import random
import re
from collections import Counter
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"
SPECIAL_TOKENS = [PAD, UNK, SOS, EOS]

_token_re = re.compile(r"[a-z0-9]+|[^\sa-z0-9]", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    text = text.lower().strip()
    return _token_re.findall(text)


class Vocab:
    def __init__(self, counter: Counter = None, max_size: int = 30000, min_freq: int = 2):
        self.itos = list(SPECIAL_TOKENS)
        self.stoi = {}
        if counter is not None:
            for tok, freq in counter.most_common():
                if freq < min_freq or len(self.itos) >= max_size:
                    break
                self.itos.append(tok)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        toks = []
        for i in ids:
            tok = self.itos[i] if i < len(self.itos) else UNK
            if tok == EOS:
                break
            if tok in (PAD, SOS):
                continue
            toks.append(tok)
        return toks

    def __len__(self):
        return len(self.itos)


def build_vocab(pairs: List[Tuple[List[str], List[str]]], max_size=30000, min_freq=2) -> Vocab:
    counter = Counter()
    for src, tgt in pairs:
        counter.update(src)
        counter.update(tgt)
    return Vocab(counter, max_size=max_size, min_freq=min_freq)


class HeadlineDataset(Dataset):
    """Expects a list of (article_tokens, headline_tokens) pairs."""

    def __init__(self, pairs: List[Tuple[List[str], List[str]]], vocab: Vocab,
                 max_src_len: int = 60, max_tgt_len: int = 20):
        self.pairs = pairs
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_toks, tgt_toks = self.pairs[idx]
        src_ids = self.vocab.encode(src_toks[: self.max_src_len])
        tgt_ids = ([self.vocab.stoi[SOS]]
                   + self.vocab.encode(tgt_toks[: self.max_tgt_len])
                   + [self.vocab.stoi[EOS]])
        if len(src_ids) == 0:
            src_ids = [self.vocab.stoi[UNK]]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch, pad_idx=0):
    srcs, tgts = zip(*batch)
    src_lens = torch.tensor([len(s) for s in srcs], dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in tgts], dtype=torch.long)

    src_pad = torch.full((len(srcs), int(src_lens.max())), pad_idx, dtype=torch.long)
    tgt_pad = torch.full((len(tgts), int(tgt_lens.max())), pad_idx, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, : len(s)] = s
        tgt_pad[i, : len(t)] = t

    src_lens, sort_idx = src_lens.sort(descending=True)
    src_pad = src_pad[sort_idx]
    tgt_pad = tgt_pad[sort_idx]
    tgt_lens = tgt_lens[sort_idx]
    return src_pad, src_lens, tgt_pad, tgt_lens


def load_split(path):
    """Load a JSONL split saved by preprocess.py -> list of (source, target)."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            pairs.append((ex["source"], ex["target"]))
    return pairs


def load_frozen_splits(data_dir="data"):
    """Load frozen val/test splits + vocab saved by preprocess.py."""
    val_pairs = load_split(os.path.join(data_dir, "val.jsonl"))
    test_pairs = load_split(os.path.join(data_dir, "test.jsonl"))
    return val_pairs, test_pairs


def load_gigaword_splits(max_train=200000, max_val=5000, max_test=2000, seed=42):
    from datasets import load_dataset

    random.seed(seed)
    ds = load_dataset("walzen/gigaword")

    cols = ds["train"].column_names
    print(f"Detected columns in walzen/gigaword: {cols}")

    doc_candidates = ["document", "text", "article", "src", "source"]
    sum_candidates = ["summary", "target", "headline", "tgt", "title"]
    doc_key = next((c for c in doc_candidates if c in cols), None)
    sum_key = next((c for c in sum_candidates if c in cols), None)
    if doc_key is None or sum_key is None:
        raise ValueError(
            f"Could not auto-detect document/summary columns among {cols}. "
            f"Inspect the dataset (e.g. print(ds['train'][0])) and update "
            f"doc_candidates/sum_candidates in load_gigaword_splits() accordingly."
        )
    print(f"Using '{doc_key}' as article/document field and '{sum_key}' as headline/summary field.")

    def to_pairs(split, limit):
        n = min(limit, len(split))
        pairs = []
        for i in range(n):
            ex = split[i]
            src = tokenize(ex[doc_key])
            tgt = tokenize(ex[sum_key])
            if len(src) == 0 or len(tgt) == 0:
                continue
            pairs.append((src, tgt))
        return pairs

    train_pairs = to_pairs(ds["train"], max_train)
    val_pairs = to_pairs(ds["validation"], max_val)
    test_pairs = to_pairs(ds["test"], max_test)
    return train_pairs, val_pairs, test_pairs
