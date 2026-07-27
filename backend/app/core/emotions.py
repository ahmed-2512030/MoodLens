"""GoEmotions (28 labels) -> Ekman 6 core emotions mapping.

Source: Demszky et al. (2020), GoEmotions. The six MoodLens classes are the
Ekman set used throughout the project: joy, anger, sadness, fear, surprise,
disgust. `neutral` is kept as a 7th bucket so probabilities always sum to 1.
"""

# The six emotions the project reports on, plus neutral.
EKMAN = ["joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"]

# Per-emotion colours — kept in sync with the frontend (lib/api.ts EMOTION_COLORS)
# so the PDF report and the web dashboard read as one visual system.
EMOTION_COLORS: dict[str, str] = {
    "joy": "#f59e0b",
    "anger": "#ef4444",
    "sadness": "#3b82f6",
    "fear": "#8b5cf6",
    "surprise": "#ec4899",
    "disgust": "#10b981",
    "neutral": "#94a3b8",
}

GOEMOTIONS_TO_EKMAN: dict[str, str] = {
    # joy
    "amusement": "joy",
    "excitement": "joy",
    "joy": "joy",
    "love": "joy",
    "desire": "joy",
    "optimism": "joy",
    "caring": "joy",
    "pride": "joy",
    "admiration": "joy",
    "gratitude": "joy",
    "relief": "joy",
    "approval": "joy",
    # anger
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    # sadness
    "sadness": "sadness",
    "disappointment": "sadness",
    "embarrassment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    # fear
    "fear": "fear",
    "nervousness": "fear",
    # surprise
    "surprise": "surprise",
    "realization": "surprise",
    "confusion": "surprise",
    "curiosity": "surprise",
    # disgust
    "disgust": "disgust",
    # neutral
    "neutral": "neutral",
}


def to_ekman(scored: dict[str, float]) -> dict[str, float]:
    """Aggregate fine-grained GoEmotions scores into the 6+neutral buckets."""
    out = {e: 0.0 for e in EKMAN}
    for label, score in scored.items():
        bucket = GOEMOTIONS_TO_EKMAN.get(label.lower())
        if bucket:
            out[bucket] += score
    return out
