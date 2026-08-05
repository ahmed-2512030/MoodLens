"""Evaluate the deployed MoodLens model on EXTERNAL English emotion benchmarks.

The in-domain benchmark (ml/evaluate.py) scores models on GoEmotions — the data the
models were trained on. This script measures cross-dataset generalisation: it runs the
served model (the ensemble by default, via app.services.classifier) on independent
English emotion datasets, mapping their native labels to the Ekman space.

Each external dataset covers only a subset of the Ekman emotions, so the model's output
is restricted (masked) to the labels the dataset actually uses before argmax — a fair
test of discriminative ability on those emotions.

Run (from backend/):
    python -m ml.evaluate_external --dataset dair-ai/emotion
    python -m ml.evaluate_external --dataset tweet_eval --config emotion
"""
from __future__ import annotations

import argparse

from datasets import load_dataset

from app.core.emotions import EKMAN
from app.services.classifier import load_classifier
from ml.metrics import compute_metrics, save_confusion_matrix, save_metrics

# Map each benchmark's native label names to Ekman buckets.
LABEL_MAPS: dict[str, dict[str, str]] = {
    "dair-ai/emotion": {
        "sadness": "sadness", "joy": "joy", "love": "joy",
        "anger": "anger", "fear": "fear", "surprise": "surprise",
    },
    "tweet_eval/emotion": {
        "anger": "anger", "joy": "joy", "optimism": "joy", "sadness": "sadness",
    },
}


def _resolve_map(dataset: str, config: str | None) -> dict[str, str]:
    key = f"{dataset}/{config}" if config else dataset
    if key in LABEL_MAPS:
        return LABEL_MAPS[key]
    if dataset in LABEL_MAPS:
        return LABEL_MAPS[dataset]
    raise SystemExit(f"No label map for '{key}'. Add one to LABEL_MAPS.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset id, e.g. dair-ai/emotion")
    ap.add_argument("--config", default=None, help="HF dataset config, e.g. emotion")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="metrics")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    label_map = _resolve_map(args.dataset, args.config)
    ekman_index = {e: i for i, e in enumerate(EKMAN)}
    allowed = sorted({ekman_index[v] for v in label_map.values()})
    allowed_names = [EKMAN[i] for i in allowed]
    print(f"Ekman classes present in this benchmark: {allowed_names}")

    ds = load_dataset(args.dataset, args.config) if args.config else load_dataset(args.dataset)
    split = ds[args.split]
    names = split.features["label"].names  # native label names, indexed by int

    texts: list[str] = []
    y_true: list[int] = []
    for ex in split:
        native = names[ex["label"]]
        bucket = label_map.get(native)
        if bucket is None:  # label has no Ekman equivalent (e.g. ISEAR guilt) -> skip
            continue
        texts.append(ex["text"])
        y_true.append(ekman_index[bucket])
    print(f"Evaluating MoodLens on {len(texts)}/{len(split)} mappable rows …")

    clf = load_classifier()
    print(f"Model: {clf.model_name}")

    # Predict: restrict each row's scores to the dataset's Ekman subset, then argmax.
    y_pred: list[int] = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        for res in clf.classify(batch):
            scores = res["emotions"]
            best = max(allowed, key=lambda i: scores[EKMAN[i]])
            y_pred.append(best)
        print(f"  scored {min(start + args.batch_size, len(texts))}/{len(texts)}", end="\r")
    print()

    metrics = compute_metrics(y_true, y_pred, EKMAN, label_ids=allowed)
    print(f"\nAccuracy   : {metrics['accuracy']}")
    print(f"Macro-F1   : {metrics['macro_f1']}")
    print(f"Weighted-F1: {metrics['weighted_f1']}")

    tag = "external_" + (f"{args.dataset}_{args.config}" if args.config else args.dataset)
    tag = tag.replace("/", "_")
    save_metrics(metrics, f"{args.out_dir}/{tag}.json")
    save_confusion_matrix(
        y_true, y_pred, EKMAN, f"{args.out_dir}/{tag}_confusion.png", label_ids=allowed
    )


if __name__ == "__main__":
    main()
