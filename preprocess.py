import json
import os
from typing import List, Tuple

from data import (
    PAD,
    SOS,
    EOS,
    UNK,
    Vocab,
    build_vocab,
    load_gigaword_splits,
    tokenize,
)

# Dataset info for the report.
DATASET_NAME = "Gigaword"
DATASET_SOURCE = "https://huggingface.co/datasets/walzen/gigaword"
DATASET_CITATION = """
@inproceedings{napoles2012annotated,
  title={Annotated Gigaword},
  author={Napoles, Courtney and Gormley, Matthew R. and Van Durme, Benjamin},
  booktitle={Proceedings of the Joint Workshop on Automatic Knowledge Base
             Construction and Web-scale Knowledge Extraction (AKBC-WEKEX)},
  year={2012}
}
"""
DATASET_LICENSE = (
    "Gigaword is a corpus of news articles compiled by the Linguistic Data "
    "Consortium (LDC). The `walzen/gigaword` mirror on HuggingFace is a "
    "re-distribution for research/educational use. Users should verify the "
    "LDC license terms before commercial use. See: "
    "https://catalog.ldc.upenn.edu/LDC2012T21"
)


def dataset_info() -> dict:
    return {
        "name": DATASET_NAME,
        "source": DATASET_SOURCE,
        "citation": DATASET_CITATION.strip(),
        "license": DATASET_LICENSE,
    }


def _save_split(path: str, pairs: List[Tuple[List[str], List[str]]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for src, tgt in pairs:
            f.write(json.dumps({"source": src, "target": tgt}) + "\n")


def prepare_data(
    max_train: int = 200000,
    max_val: int = 5000,
    max_test: int = 2000,
    vocab_size: int = 30000,
    min_freq: int = 2,
    seed: int = 42,
    data_dir: str = "data",
) -> Tuple[List[Tuple[List[str], List[str]]],
           List[Tuple[List[str], List[str]]],
           List[Tuple[List[str], List[str]]],
           Vocab]:
    """Load data, reserve val/test splits, build vocab from train only.

    Returns (train_pairs, val_pairs, test_pairs, vocab).
    """
    # Reserve held-out val/test before model development.
    train_pairs, val_pairs, test_pairs = load_gigaword_splits(
        max_train=max_train, max_val=max_val, max_test=max_test, seed=seed
    )

    # Build vocab from TRAIN ONLY (no leakage).
    vocab = build_vocab(train_pairs, max_size=vocab_size, min_freq=min_freq)

    # Skip any val/test examples whose source also appears in train.
    train_src = {tuple(src) for src, _ in train_pairs}
    val_pairs = [p for p in val_pairs if tuple(p[0]) not in train_src]
    test_pairs = [p for p in test_pairs if tuple(p[0]) not in train_src]

    print(f"Dataset: {DATASET_NAME} ({DATASET_SOURCE})")
    print(f"License: {DATASET_LICENSE}")
    print(f"train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    print(f"vocab_size={len(vocab)} (built from train only)")
    print(f"leakage check: skipped val/test sources that overlap train")

    # Save held-out val/test splits to disk so they are reserved before training.
    os.makedirs(data_dir, exist_ok=True)
    _save_split(os.path.join(data_dir, "val.jsonl"), val_pairs)
    _save_split(os.path.join(data_dir, "test.jsonl"), test_pairs)

    return train_pairs, val_pairs, test_pairs, vocab


__all__ = [
    "PAD", "UNK", "SOS", "EOS",
    "Vocab",
    "tokenize",
    "dataset_info",
    "prepare_data",
]


if __name__ == "__main__":
    train, val, test, vocab = prepare_data()
    print("dataset_info:", dataset_info())
    print(
        f"Ready. frozen val/test saved to data/. train={len(train)} "
        f"val={len(val)} test={len(test)} vocab={len(vocab)}"
    )
