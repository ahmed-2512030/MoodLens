"""Emotion classifier wrapper.

Loads the GoEmotions model(s) ONCE and keeps them in memory. Loading inside each
request would re-read hundreds of MB of weights per call and make the API
unusable, so the app lifespan builds a single instance at startup.

Supports an ENSEMBLE of GoEmotions-trained models. The dissertation benchmark
(PROJECT_REPORT Finding #7) showed that averaging the Ekman-space scores of
`bhadresh-savani/bert-base-go-emotion` + `SamLowe/roberta-base-go_emotions` beats
either model alone (macro-F1 0.673 vs 0.663). Each member is normalised in Ekman
space, then the members are averaged — identical to `ml/evaluate_ensemble.py`, so
what we serve matches what we benchmarked.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.config import settings
from app.core.emotions import EKMAN, to_ekman


class _Member:
    """One model in the ensemble: tokenizer + weights + label map."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def raw_scores(self, texts: list[str]) -> list[dict[str, float]]:
        # 512 = the model's hard position-embedding ceiling (BERT/RoBERTa). Text
        # beyond this is still truncated — a single forward pass cannot see more.
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**enc).logits
        # GoEmotions is multi-label -> sigmoid. Softmax would force a single
        # winner and distort the aggregated Ekman distribution.
        probs = torch.sigmoid(logits).cpu().tolist()
        return [
            {self.id2label[i]: float(p) for i, p in enumerate(row)} for row in probs
        ]


class EmotionClassifier:
    def __init__(self, model_names: list[str]) -> None:
        if not model_names:
            raise ValueError("model_names must not be empty")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.members = [_Member(name, self.device) for name in model_names]
        self.is_ensemble = len(self.members) > 1
        self.model_name = (
            "ensemble(" + ", ".join(m.model_name for m in self.members) + ")"
            if self.is_ensemble
            else self.members[0].model_name
        )

    def classify(self, texts: list[str]) -> list[dict]:
        n = len(self.members)
        # Accumulate across members: Ekman scores (per-member normalised) and the
        # fine-grained GoEmotions scores (for the top-k explainability list).
        ekman_sum = [{e: 0.0 for e in EKMAN} for _ in texts]
        fine_sum: list[defaultdict[str, float]] = [defaultdict(float) for _ in texts]

        for member in self.members:
            for i, fine in enumerate(member.raw_scores(texts)):
                ekman = to_ekman(fine)
                total = sum(ekman.values()) or 1.0
                for emo in EKMAN:
                    ekman_sum[i][emo] += ekman[emo] / total  # normalise then average
                for label, score in fine.items():
                    fine_sum[i][label] += score

        results = []
        for i in range(len(texts)):
            ekman_avg = {e: ekman_sum[i][e] / n for e in EKMAN}
            total = sum(ekman_avg.values()) or 1.0
            ekman_norm = {e: ekman_avg[e] / total for e in EKMAN}
            dominant = max(ekman_norm, key=ekman_norm.get)
            top_fine = sorted(
                ((lbl, s / n) for lbl, s in fine_sum[i].items()),
                key=lambda kv: kv[1],
                reverse=True,
            )
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

    def classify_document(
        self, text: str, max_tokens: int = 500, max_chunks: int = 40
    ) -> dict:
        """Analyse text of any length by chunking past the 512-token limit.

        Returns the same shape as one `classify()` result, plus:
        - `chunk_count`: number of chunks scored (1 for short text).
        - `arc`: per-chunk Ekman vectors (0-100 scale) — the within-document
          emotional trajectory that feeds the frontend trend chart. None if one chunk.
        - `truncated_chunks`: True if the document exceeded `max_chunks` and the tail
          was dropped.

        Short text is a single chunk, so the result is byte-identical to
        `classify([text])[0]` — no behaviour change for normal input or benchmarks.
        """
        from app.services.chunking import split_into_chunks

        chunks = split_into_chunks(text, self.members[0].tokenizer, max_tokens=max_tokens)
        truncated = len(chunks) > max_chunks
        if truncated:
            chunks = chunks[:max_chunks]

        results = self.classify([c["text"] for c in chunks])

        if len(results) == 1:  # short text — preserve exact single-analysis behaviour
            return {**results[0], "chunk_count": 1, "arc": None, "truncated_chunks": False}

        # Headline = token-length-weighted mean of the per-chunk Ekman vectors, so
        # longer chunks count proportionally more toward the document's overall mood.
        weights = [max(c["n_tokens"], 1) for c in chunks]
        wsum = sum(weights) or 1.0
        emo = {e: 0.0 for e in EKMAN}
        fine_sum: defaultdict[str, float] = defaultdict(float)
        for r, w in zip(results, weights):
            for e in EKMAN:
                emo[e] += r["emotions"][e] * w
            for f in r["fine_grained"]:
                fine_sum[f["label"]] += f["score"] * w

        emo_avg = {e: emo[e] / wsum for e in EKMAN}
        total = sum(emo_avg.values()) or 1.0
        emo_norm = {e: emo_avg[e] / total for e in EKMAN}
        top_fine = sorted(
            ((lbl, s / wsum) for lbl, s in fine_sum.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )

        # Arc: one point per chunk, shaped exactly as the frontend TrendPoint
        # (bin label = char span, values on the 0-100 % scale).
        arc = []
        for idx, (r, c) in enumerate(zip(results, chunks)):
            point = {"bin": f"{c['start'] + 1}–{c['end']}", "index": idx}
            for e in EKMAN:
                point[e] = round(r["emotions"][e] * 100, 1)
            arc.append(point)

        return {
            "dominant": max(emo_norm, key=emo_norm.get),
            "emotions": {e: round(emo_norm[e], 4) for e in EKMAN},
            "fine_grained": [
                {"label": lbl, "score": round(s, 4)} for lbl, s in top_fine[: settings.top_k]
            ],
            "chunk_count": len(chunks),
            "arc": arc,
            "truncated_chunks": truncated,
        }


# Set by the FastAPI lifespan on startup.
classifier: EmotionClassifier | None = None


def load_classifier() -> EmotionClassifier:
    global classifier
    if classifier is None:
        classifier = EmotionClassifier(settings.model_names)
    return classifier


def get_classifier() -> EmotionClassifier:
    if classifier is None:
        raise RuntimeError("Classifier not loaded")
    return classifier
