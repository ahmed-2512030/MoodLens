"""Emotion classifier wrapper.

Loads the BERT/GoEmotions model ONCE and keeps it in memory. Loading inside
each request would re-read ~400MB of weights per call and make the API
unusable, so the app lifespan builds a single instance at startup.
"""
from __future__ import annotations

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from app.core.config import settings
from app.core.emotions import EKMAN, to_ekman


class EmotionClassifier:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def _raw_scores(self, texts: list[str]) -> list[dict[str, float]]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits
        # GoEmotions is multi-label -> sigmoid. Softmax would force a single
        # winner and distort the aggregated Ekman distribution.
        probs = torch.sigmoid(logits).cpu().tolist()
        return [
            {self.id2label[i]: float(p) for i, p in enumerate(row)}
            for row in probs
        ]

    def classify(self, texts: list[str]) -> list[dict]:
        results = []
        for fine in self._raw_scores(texts):
            ekman = to_ekman(fine)
            total = sum(ekman.values()) or 1.0
            ekman_norm = {k: v / total for k, v in ekman.items()}
            dominant = max(ekman_norm, key=ekman_norm.get)
            top_fine = sorted(fine.items(), key=lambda kv: kv[1], reverse=True)
            results.append(
                {
                    "dominant": dominant,
                    "emotions": {k: round(ekman_norm[k], 4) for k in EKMAN},
                    "fine_grained": [
                        {"label": lbl, "score": round(s, 4)}
                        for lbl, s in top_fine[: settings.top_k]
                    ],
                }
            )
        return results


# Set by the FastAPI lifespan on startup.
classifier: EmotionClassifier | None = None


def load_classifier() -> EmotionClassifier:
    global classifier
    if classifier is None:
        classifier = EmotionClassifier(settings.model_name)
    return classifier


def get_classifier() -> EmotionClassifier:
    if classifier is None:
        raise RuntimeError("Classifier not loaded")
    return classifier
