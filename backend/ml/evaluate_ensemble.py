"""Evaluate an ENSEMBLE of GoEmotions-trained models on the Ekman-6 test set.

Motivation (see PROJECT_REPORT Findings #1, #5): no single model wins every class —
the baseline is strong on fear/neutral, SamLowe is stronger on disgust. This averages
each model's Ekman-space scores (sigmoid -> to_ekman) and argmaxes the mean, so their
per-class strengths combine. Scored on the SAME single-label test set as every other
model, so macro-F1 compares directly to the 0.663 baseline.

Run (from backend/):
    python -m ml.evaluate_ensemble
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.emotions import EKMAN, to_ekman
from ml.data import load_ekman_dataset
from ml.metrics import compute_metrics, save_confusion_matrix, save_metrics

DEFAULT_MODELS = [
    "bhadresh-savani/bert-base-go-emotion",
    "SamLowe/roberta-base-go_emotions",
]


@torch.inference_mode()
def _ekman_scores(model, tokenizer, texts, device, batch_size=32):
    """Return an (N, 7) tensor of Ekman-space scores for one model."""
    ekman_index = {e: i for i, e in enumerate(EKMAN)}
    id2label = model.config.id2label
    rows: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(device)
        probs = torch.sigmoid(model(**enc).logits).cpu().tolist()
        for row in probs:
            scored = {id2label[i]: p for i, p in enumerate(row)}
            ekman = to_ekman(scored)
            vec = [0.0] * len(EKMAN)
            for emo, score in ekman.items():
                vec[ekman_index[emo]] = score
            # normalise so each model contributes on the same scale before averaging
            total = sum(vec) or 1.0
            rows.append([v / total for v in vec])
        print(f"  scored {min(start + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return torch.tensor(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="metrics")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading + mapping GoEmotions …")
    ds = load_ekman_dataset()[args.split]
    texts, y_true = ds["text"], ds["label"]
    print(f"Evaluating ensemble of {len(args.models)} models on {len(texts)} rows …")

    summed = torch.zeros(len(texts), len(EKMAN))
    for name in args.models:
        print(f"Scoring {name} on {device} …")
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        summed += _ekman_scores(model, tok, texts, device)
        del model

    y_pred = summed.argmax(dim=-1).tolist()  # mean == sum for argmax purposes

    metrics = compute_metrics(y_true, y_pred, EKMAN)
    print(f"\nAccuracy   : {metrics['accuracy']}")
    print(f"Macro-F1   : {metrics['macro_f1']}")
    print(f"Weighted-F1: {metrics['weighted_f1']}")

    tag = "ensemble_" + "_".join(m.split("/")[-1] for m in args.models)
    save_metrics(metrics, f"{args.out_dir}/{tag}.json")
    save_confusion_matrix(y_true, y_pred, EKMAN, f"{args.out_dir}/{tag}_confusion.png")


if __name__ == "__main__":
    main()
