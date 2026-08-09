import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import build_vocab, collate_fn, HeadlineDataset, load_gigaword_splits, PAD, SOS, EOS
from model import Encoder, Decoder, Seq2Seq


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for src, src_lens, tgt, tgt_lens in loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
            loss = criterion(output[:, 1:].reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--hid_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--vocab_size", type=int, default=30000)
    ap.add_argument("--max_train", type=int, default=200000)
    ap.add_argument("--max_val", type=int, default=5000)
    ap.add_argument("--max_test", type=int, default=2000)
    ap.add_argument("--teacher_forcing", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading Gigaword splits ...")
    train_pairs, val_pairs, test_pairs = load_gigaword_splits(
        max_train=args.max_train, max_val=args.max_val, max_test=args.max_test, seed=args.seed)
    print(f"train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

    vocab = build_vocab(train_pairs, max_size=args.vocab_size)
    print(f"Vocab size: {len(vocab)}")

    train_ds = HeadlineDataset(train_pairs, vocab)
    val_ds = HeadlineDataset(val_pairs, vocab)

    pad_idx = vocab.stoi[PAD]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=lambda b: collate_fn(b, pad_idx))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, pad_idx))

    encoder = Encoder(len(vocab), args.emb_dim, args.hid_dim, dropout=args.dropout, pad_idx=pad_idx)
    decoder = Decoder(len(vocab), args.emb_dim, args.hid_dim, dropout=args.dropout, pad_idx=pad_idx)
    model = Seq2Seq(encoder, decoder, pad_idx, vocab.stoi[SOS], vocab.stoi[EOS], device).to(device)
    print(f"Trainable parameters: {count_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        total_loss = 0.0
        for i, (src, src_lens, tgt, tgt_lens) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, src_lens, tgt, teacher_forcing_ratio=args.teacher_forcing)
            loss = criterion(output[:, 1:].reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"  epoch {epoch} step {i+1}/{len(train_loader)} loss {loss.item():.4f}")

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        elapsed = time.time() - start
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"({elapsed:.1f}s, hardware={device})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "vocab_itos": vocab.itos,
                "args": vars(args),
            }, os.path.join(args.out_dir, "best_model.pt"))
            print("  -> saved new best checkpoint")

    print("Training complete.")


if __name__ == "__main__":
    main()
