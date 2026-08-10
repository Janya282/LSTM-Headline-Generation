# LSTM Headline Generation

LSTM encoder decoder with attention, built from scratch in PyTorch, trained on Gigaword to generate headlines from articles!

## Setup

```
pip install -r requirements.txt
```

## Run it (3 hour version of complete training - produces results discussed in report)

```
python train.py --epochs 3 --max_train 50000 --max_val 2000 --max_test 500 --batch_size 64
python generate.py --checkpoint checkpoints/best_model.pt --beam_width 4
python evaluate.py --predictions predictions.jsonl
```
## Run it (2-3 min version used for demo video - results included in appendix)
```
python train.py --epochs 1 --max_train 2000 --max_val 500 --max_test 100 --batch_size 64
python generate.py --checkpoint checkpoints/best_model.pt --beam_width 4 --limit 100
python evaluate.py --predictions predictions.jsonl
```

Training saves the best model to `checkpoints/best_model.pt`. Generate makes headlines for the test set and saves them to `predictions.jsonl`. Evaluate prints ROUGE scores.

## Files

- `data.py` - loads Gigaword, tokenizes, builds vocab
- `model.py` - encoder, attention, decoder
- `train.py` - training loop
- `generate.py` - greedy/beam search decoding
- `evaluate.py` - ROUGE scoring

## Note on the dataset

- The `gigaword` dataset on HuggingFace seemed broken so I loaded `walzen/gigaword` instead - same data, different mirror. Its columns are named `article`/`summary`, which the code auto detects.
- Don't commit the checkpoint file (checkpoints/best_model.pt). It's over 100MB and GitHub will reject the push with a "file size limit" error. .gitignore already excludes the checkpoints/ folder, __pycache__/, and predictions.jsonl, so as long as you don't force add them, git will skip them automatically. Please and thank you!

## Test run so far

Trained 1 epoch on 5,000 examples on CPU (~5.5 min). Loss went from ~9 to ~7. ROUGE-1 was 0.065, ROUGE-2 was 0.008 - low because it's barely trained. Predictions were mostly repeating "#" (the token used for all numbers in this dataset) instead of real words.
