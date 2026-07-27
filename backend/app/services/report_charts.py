"""Render the dashboard's visualisations as PNGs for the PDF report.

Mirrors the frontend charts (components/EmotionCharts.tsx + lib/analytics.ts) so the
downloadable report shows the same pie / average / trend / keyword views the user sees
on screen. Pure functions returning PNG bytes; no web or model dependencies.
"""
from __future__ import annotations

import io
import math
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # render to a buffer, no display
import matplotlib.pyplot as plt

from app.core.emotions import EKMAN, EMOTION_COLORS


def _png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ---- Pie: dominant-emotion distribution -----------------------------------
def pie_png(distribution: dict[str, int]) -> io.BytesIO:
    items = [(e, distribution.get(e, 0)) for e in EKMAN if distribution.get(e, 0) > 0]
    labels = [e for e, _ in items]
    sizes = [v for _, v in items]
    colors = [EMOTION_COLORS[e] for e in labels]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda p: f"{p:.0f}%",
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax.set_title("Dominant emotion distribution", fontsize=11, weight="bold")
    ax.axis("equal")
    return _png(fig)


# ---- Bars: average emotion intensity --------------------------------------
def bars_png(average_scores: dict[str, float]) -> io.BytesIO:
    vals = [average_scores.get(e, 0.0) * 100 for e in EKMAN]
    colors = [EMOTION_COLORS[e] for e in EKMAN]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.bar(EKMAN, vals, color=colors)
    ax.set_ylabel("mean score (%)", fontsize=9)
    ax.set_title("Average emotion intensity", fontsize=11, weight="bold")
    ax.tick_params(axis="x", labelsize=9, rotation=30)
    ax.tick_params(axis="y", labelsize=8)
    return _png(fig)


# ---- Trend: emotion intensity across the document stream ------------------
def _build_trend(items, bins: int = 10) -> list[dict]:
    n = len(items)
    if not n:
        return []
    size = max(1, math.ceil(n / bins))
    points = []
    for start in range(0, n, size):
        chunk = items[start : start + size]
        point = {"bin": f"{start + 1}-{min(start + size, n)}"}
        for emo in EKMAN:
            point[emo] = (
                sum(it.emotions.get(emo, 0.0) for it in chunk) / len(chunk) * 100
            )
        points.append(point)
    return points


def trend_png(items) -> io.BytesIO:
    pts = _build_trend(items)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    if pts:
        x = [p["bin"] for p in pts]
        for emo in EKMAN:
            ax.plot(
                x, [p[emo] for p in pts], label=emo, color=EMOTION_COLORS[emo],
                linewidth=2, marker="o", markersize=3,
            )
        ax.set_xlabel("document range", fontsize=9)
        ax.set_ylabel("mean score (%)", fontsize=9)
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(fontsize=7, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.set_title("Emotion trend across documents", fontsize=11, weight="bold")
    return _png(fig)


# ---- Keyword -> emotion mapping (explainability) --------------------------
_STOPWORDS = set(
    (
        "a an the and or but if then so of to in on at for with without from by as is are was "
        "were be been being it its this that these those i you he she we they them his her our "
        "your their my me him us do does did done have has had having not no nor can could will "
        "would shall should may might must just really very too also about into over under up "
        "down out off than there here what which who whom when where why how all any some more "
        "most other such only own same one two get got like im ive dont cant"
    ).split()
)


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z']{2,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _build_keyword_map(items, top_n: int = 15) -> list[dict]:
    table: dict[str, dict] = {}
    for it in items:
        for word in _tokenize(it.text):
            stat = table.setdefault(
                word, {"word": word, "total": 0, "counts": defaultdict(int)}
            )
            stat["counts"][it.dominant] += 1
            stat["total"] += 1
    stats = [s for s in table.values() if s["total"] >= 2]
    for s in stats:
        s["dominant"] = max(EKMAN, key=lambda e: s["counts"].get(e, 0))
    stats.sort(key=lambda s: s["total"], reverse=True)
    return stats[:top_n]


def keyword_png(items) -> io.BytesIO:
    stats = _build_keyword_map(items)
    fig, ax = plt.subplots(figsize=(7, 4))
    if stats:
        stats = list(reversed(stats))  # largest at top in barh
        words = [s["word"] for s in stats]
        totals = [s["total"] for s in stats]
        colors = [EMOTION_COLORS[s["dominant"]] for s in stats]
        ax.barh(words, totals, color=colors)
        ax.set_xlabel("documents containing the word", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        # legend of the emotions actually present
        present = {s["dominant"] for s in stats}
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=EMOTION_COLORS[e])
            for e in EKMAN
            if e in present
        ]
        labels = [e for e in EKMAN if e in present]
        ax.legend(handles, labels, fontsize=7, ncol=4, loc="lower right")
    else:
        ax.text(0.5, 0.5, "Not enough repeated words to map", ha="center", fontsize=9)
        ax.axis("off")
    ax.set_title("Keyword to emotion mapping", fontsize=11, weight="bold")
    return _png(fig)
