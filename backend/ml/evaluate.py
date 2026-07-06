"""Evaluate a model on the GoEmotions test set, in the Ekman-6 label space.

Two label spaces are supported so you can measure BOTH:

  --label-space goemotions   the off-the-shelf 28-label BERT. Its 28 outputs are
                             aggregated to Ekman via to_ekman() then argmax.
                             Gives you a BASELINE number with NO training.

  --label-space ekman        a 7-output model YOU fine-tuned (see train.py).
                             Outputs are already Ekman, so we argmax directly.

Run (from the backend/ directory):
    python -m ml.evaluate --model bhadresh-savani/bert-base-go-emotion \
        --label-space goemotions
    python -m ml.evaluate --model models/ekman-bert --label-space ekman
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.emotions import EKMAN, to_ekman
from ml.data import load_ekman_dataset
from ml.metrics import compute_metrics, save_confusion_matrix, save_metrics


@torch.inference_mode()
def _predict(model, tokenizer, texts, device, label_space, batch_size=32):
    preds: list[int] = []
    ekman_index = {e: i for i, e in enumerate(EKMAN)}
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(device)
        logits = model(**enc).logits

        if label_space == "goemotions":
            # 28 multi-label outputs -> sigmoid -> aggregate to Ekman -> argmax.
            probs = torch.sigmoid(logits).cpu().tolist()
            id2label = model.config.id2label
            for row in probs:
                scored = {id2label[i]: p for i, p in enumerate(row)}
                ekman = to_ekman(scored)
                preds.append(ekman_index[max(ekman, key=ekman.get)])
        else:
            # 7 single-label outputs -> argmax directly.
            preds.extend(logits.argmax(dim=-1).cpu().tolist())

        print(f"  scored {min(start + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path")
    ap.add_argument("--label-space", choices=["goemotions", "ekman"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="metrics")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device).eval()

    print("Loading + mapping GoEmotions …")
    ds = load_ekman_dataset()[args.split]
    texts, y_true = ds["text"], ds["label"]
    print(f"Evaluating on {len(texts)} rows …")

    y_pred = _predict(model, tokenizer, texts, device, args.label_space)

    metrics = compute_metrics(y_true, y_pred, EKMAN)
    print(f"\nAccuracy   : {metrics['accuracy']}")
    print(f"Macro-F1   : {metrics['macro_f1']}")
    print(f"Weighted-F1: {metrics['weighted_f1']}")

    tag = args.model.replace("/", "_").strip("._")
    save_metrics(metrics, f"{args.out_dir}/{tag}.json")
    save_confusion_matrix(y_true, y_pred, EKMAN, f"{args.out_dir}/{tag}_confusion.png")


if __name__ == "__main__":
    main()
