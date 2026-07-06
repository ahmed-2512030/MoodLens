"""Build dashboard summary stats from a batch of classified items."""
from app.core.emotions import EKMAN


def summarize(items: list[dict]) -> dict:
    count = len(items)
    distribution = {e: 0 for e in EKMAN}
    totals = {e: 0.0 for e in EKMAN}

    for item in items:
        distribution[item["dominant"]] += 1
        for emo, score in item["emotions"].items():
            totals[emo] += score

    average = {
        e: round(totals[e] / count, 4) if count else 0.0 for e in EKMAN
    }
    return {
        "count": count,
        "distribution": distribution,
        "average_scores": average,
    }
