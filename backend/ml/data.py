"""Load GoEmotions and collapse it to the Ekman-6 (+neutral) label space.

GoEmotions ships 28 fine-grained, MULTI-label annotations. Our research question
is about the 6 Ekman emotions, and "accuracy" is cleanest to define for a
SINGLE-label problem. So we:

  1. Map each example's GoEmotions labels to their Ekman buckets.
  2. Keep only examples that map to exactly ONE Ekman bucket (single-label).

Dropping the genuinely-ambiguous multi-emotion rows is a deliberate, documented
simplification — state it in your methodology chapter and report how many rows
it removes (printed by `stats()` below).
"""
from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset

from app.core.emotions import EKMAN, GOEMOTIONS_TO_EKMAN

# Stable integer id for each Ekman label; the model's output neurons use this order.
EKMAN_LABEL2ID: dict[str, int] = {e: i for i, e in enumerate(EKMAN)}
EKMAN_ID2LABEL: dict[int, str] = {i: e for e, i in EKMAN_LABEL2ID.items()}


def _ekman_buckets(go_names: list[str], label_ids: list[int]) -> set[str]:
    """The set of distinct Ekman emotions an example's GoEmotions labels map to."""
    buckets = {GOEMOTIONS_TO_EKMAN.get(go_names[i]) for i in label_ids}
    buckets.discard(None)
    return buckets


def _convert(split, go_names: list[str]) -> Dataset:
    texts: list[str] = []
    labels: list[int] = []
    for ex in split:
        buckets = _ekman_buckets(go_names, ex["labels"])
        if len(buckets) != 1:  # skip empty (unmapped) and multi-emotion rows
            continue
        emo = next(iter(buckets))
        texts.append(ex["text"])
        labels.append(EKMAN_LABEL2ID[emo])
    return Dataset.from_dict({"text": texts, "label": labels})


def load_ekman_dataset() -> DatasetDict:
    """Return train/validation/test splits with `text` and integer `label`."""
    raw = load_dataset("go_emotions", "simplified")
    go_names = raw["train"].features["labels"].feature.names  # the 28 label names
    return DatasetDict({name: _convert(raw[name], go_names) for name in raw})


def stats(ds: DatasetDict) -> None:
    """Print class balance per split — useful for the dissertation and to spot
    imbalance (your risk table flags 'poor or imbalanced dataset')."""
    for split, data in ds.items():
        counts = {e: 0 for e in EKMAN}
        for label in data["label"]:
            counts[EKMAN[label]] += 1
        print(f"[{split}] {len(data)} rows -> {counts}")


if __name__ == "__main__":
    stats(load_ekman_dataset())
