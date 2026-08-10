"""
llm_baseline.py
LLM baseline generator for headline generation (Gigaword) using Gemini free tier.
Loads the same frozen test.jsonl as the LSTM pipeline and writes output in the
same JSONL format expected by evaluate.py (fields: reference, prediction).
Includes retry-with-backoff for rate limits and safe handling of blocked/empty responses.
"""

import argparse
import json
import os
import re
import time

from data import load_frozen_splits
from google import genai

MODEL_NAME = "gemini-3.1-flash-lite"
FEW_SHOT_K = 5
SLEEP_BETWEEN_CALLS = 6.5   # ~9 RPM, safely under the 15 RPM free-tier cap
MAX_RETRIES = 5

PROMPT_VARIANTS = {
    "minimal": (
        "Write a short news headline for the following article sentence: "
        "{input_text}\nHeadline:"
    ),
    "structured": (
        "You are a headline writer. Given the article sentence below, output "
        "ONLY a concise news headline (max 10 words), no punctuation at the end, "
        "no explanation, no quotation marks.\n\nArticle: {input_text}\nHeadline:"
    ),
}


def detokenize(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:'\)\]])", r"\1", text)
    text = re.sub(r"([\(\[])\s+", r"\1", text)
    return text.strip()


def call_llm(client, prompt):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            text = response.text
            if text is None:
                return "", 0
            tokens = 0
            if getattr(response, "usage_metadata", None):
                tokens = (response.usage_metadata.prompt_token_count or 0) + \
                         (response.usage_metadata.candidates_token_count or 0)
            return text.strip(), tokens
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = 35 + attempt * 10
                print(f"Rate limited, waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
            print(f"Non-retryable error: {e}")
            return "", 0
    print("Max retries exceeded, skipping this example.")
    return "", 0


def build_few_shot_prefix(val_pairs, k):
    examples = val_pairs[:k]
    blocks = []
    for src_toks, tgt_toks in examples:
        blocks.append(f"Article: {detokenize(src_toks)}\nHeadline: {detokenize(tgt_toks)}")
    return "\n\n".join(blocks) + "\n\n"


def make_prompt(variant, input_text, few_shot_prefix=""):
    filled = PROMPT_VARIANTS[variant].format(input_text=input_text)
    return few_shot_prefix + filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--max_test", type=int, default=100)
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    val_pairs, test_pairs = load_frozen_splits(data_dir=args.data_dir)
    if args.max_test:
        test_pairs = test_pairs[:args.max_test]

    few_shot_prefix = build_few_shot_prefix(val_pairs, FEW_SHOT_K)
    os.makedirs(args.out_dir, exist_ok=True)

    total_tokens = 0
    total_calls = 0

    for variant in PROMPT_VARIANTS:
        for setting in ["zero_shot", "few_shot"]:
            prefix = few_shot_prefix if setting == "few_shot" else ""
            out_path = os.path.join(args.out_dir, f"llm_{variant}_{setting}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for i, (src_toks, tgt_toks) in enumerate(test_pairs):
                    document = detokenize(src_toks)
                    reference = detokenize(tgt_toks)
                    prompt = make_prompt(variant, document, prefix)
                    prediction, tokens = call_llm(client, prompt)
                    total_tokens += tokens
                    total_calls += 1
                    f.write(json.dumps({
                        "source": document,
                        "reference": reference,
                        "prediction": prediction,
                        "prompt_variant": variant,
                        "shot_setting": setting,
                        "model_name": MODEL_NAME,
                        "tokens_used": tokens,
                    }) + "\n")
                    f.flush()
                    if (i + 1) % 10 == 0:
                        print(f"{variant}/{setting}: {i+1}/{len(test_pairs)} done")
                    time.sleep(SLEEP_BETWEEN_CALLS)
            print(f"Wrote {out_path}")

    print(f"Done. Total calls={total_calls} total_tokens={total_tokens} cost=$0 (free tier)")


if __name__ == "__main__":
    main()