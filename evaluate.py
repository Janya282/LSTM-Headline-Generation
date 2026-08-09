import argparse
import json

from rouge_score import rouge_scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=str, required=True)
    args = ap.parse_args()

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = 0

    with open(args.predictions) as f:
        for line in f:
            ex = json.loads(line)
            scores = scorer.score(ex["reference"], ex["prediction"])
            for k in totals:
                totals[k] += scores[k].fmeasure
            n += 1

    print(f"Evaluated {n} examples")
    for k, v in totals.items():
        print(f"{k}: {v / n:.4f}")


if __name__ == "__main__":
    main()
