import argparse
import json
import os

import torch

from data import Vocab, tokenize, PAD, SOS, EOS, load_split
from model import Encoder, Decoder, Seq2Seq


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt["args"]
    vocab = Vocab()
    vocab.itos = ckpt["vocab_itos"]
    vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}

    pad_idx = vocab.stoi[PAD]
    encoder = Encoder(len(vocab), args["emb_dim"], args["hid_dim"], dropout=0.0, pad_idx=pad_idx)
    decoder = Decoder(len(vocab), args["emb_dim"], args["hid_dim"], dropout=0.0, pad_idx=pad_idx)
    model = Seq2Seq(encoder, decoder, pad_idx, vocab.stoi[SOS], vocab.stoi[EOS], device).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab, args


def generate_headline(model, vocab, text, device, beam_width=1, max_len=20):
    tokens = tokenize(text)
    src_ids = vocab.encode(tokens) or [vocab.stoi["<unk>"]]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_lens = torch.tensor([len(src_ids)], dtype=torch.long)

    if beam_width <= 1:
        out_ids = model.greedy_decode(src, src_lens, max_len=max_len)[0].tolist()
    else:
        out_ids = model.beam_search_decode(src, src_lens, beam_width=beam_width, max_len=max_len)

    return " ".join(vocab.decode(out_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--beam_width", type=int, default=4)
    ap.add_argument("--out_file", type=str, default="predictions.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                     help="Only decode the first N test examples (e.g. 100, to match "
                          "the LLM baseline's evaluation set for a fair side-by-side).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, model_args = load_model(args.checkpoint, device)

    test_pairs = load_split(os.path.join(args.data_dir, "test.jsonl"))
    if args.limit is not None:
        test_pairs = test_pairs[: args.limit]
    print(f"Generating predictions for {len(test_pairs)} test examples "
          f"(limit={args.limit})")

    with open(args.out_file, "w") as f:
        for src_toks, tgt_toks in test_pairs:
            src_text = " ".join(src_toks)
            ref_text = " ".join(tgt_toks)
            pred = generate_headline(model, vocab, src_text, device, beam_width=args.beam_width)
            f.write(json.dumps({"source": src_text, "reference": ref_text, "prediction": pred}) + "\n")

    print(f"Wrote predictions to {args.out_file}")


if __name__ == "__main__":
    main()
