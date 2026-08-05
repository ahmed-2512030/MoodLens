"""Split long text into token-bounded chunks for whole-document analysis.

The models cap at 512 tokens per forward pass, so a long document must be broken
into pieces, each scored independently. Splitting is sentence-aware (never cuts
mid-sentence unless a single sentence alone exceeds the budget) and packs whole
sentences greedily up to `max_tokens`. Token counts come from the real tokenizer,
so the budget is exact — char-based splitting would miscount for dense text.

Pure function, tokenizer injected: no model load, unit-testable.
"""
from __future__ import annotations

import re

# Sentence boundary: run of . ! ? followed by whitespace/end, OR a blank line.
_BOUNDARY = re.compile(r"[.!?]+(?=\s|$)|\n{2,}")


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, text) spans for each sentence, offsets into `text`."""
    sents: list[tuple[int, int, str]] = []
    prev = 0
    for m in _BOUNDARY.finditer(text):
        seg = text[prev : m.end()]
        if seg.strip():
            sents.append((prev, m.end(), seg))
        prev = m.end()
    if prev < len(text):
        seg = text[prev:]
        if seg.strip():
            sents.append((prev, len(text), seg))
    if not sents:  # no punctuation at all -> whole text is one "sentence"
        sents = [(0, len(text), text)]
    return sents


def split_into_chunks(
    text: str, tokenizer, max_tokens: int = 500, overlap: int = 1
) -> list[dict]:
    """Pack sentences into chunks of at most `max_tokens` tokens.

    - `max_tokens` leaves headroom under the model's 512 ceiling for special tokens.
    - `overlap` repeats the last N sentences of a chunk into the next one so an
      emotion spanning a boundary isn't lost. Set 0 to disable.

    Each chunk: {"text", "start", "end" (char offsets), "n_tokens"}.
    """
    sents = _split_sentences(text)
    counts = [len(tokenizer(s, add_special_tokens=False)["input_ids"]) for _, _, s in sents]

    chunks: list[dict] = []
    cur: list[int] = []  # sentence indices in the current chunk
    cur_tok = 0

    def emit(indices: list[int]) -> None:
        if not indices:
            return
        txt = "".join(sents[i][2] for i in indices).strip()
        if not txt:
            return
        chunks.append(
            {
                "text": txt,
                "start": sents[indices[0]][0],
                "end": sents[indices[-1]][1],
                "n_tokens": sum(counts[i] for i in indices),
            }
        )

    i = 0
    while i < len(sents):
        c = counts[i]

        # A single sentence longer than the budget: flush, then hard-split it by
        # token window (unavoidable — one sentence can't fit a forward pass).
        if c > max_tokens:
            emit(cur)
            cur, cur_tok = [], 0
            s_start, s_end, s_txt = sents[i]
            ids = tokenizer(s_txt, add_special_tokens=False)["input_ids"]
            n = len(ids) or 1
            for w in range(0, len(ids), max_tokens):
                piece = ids[w : w + max_tokens]
                cs = int(s_start + (w / n) * (s_end - s_start))
                ce = int(s_start + (min(w + max_tokens, len(ids)) / n) * (s_end - s_start))
                decoded = tokenizer.decode(piece).strip()
                if decoded:
                    chunks.append(
                        {"text": decoded, "start": cs, "end": ce, "n_tokens": len(piece)}
                    )
            i += 1
            continue

        # Adding this sentence would overflow -> close the chunk and seed overlap.
        if cur and cur_tok + c > max_tokens:
            emit(cur)
            cur = cur[-overlap:] if overlap > 0 else []
            cur_tok = sum(counts[j] for j in cur)
            if cur_tok + c > max_tokens:  # overlap seed too big -> drop it
                cur, cur_tok = [], 0

        cur.append(i)
        cur_tok += c
        i += 1

    emit(cur)

    if not chunks:  # pathological (e.g. whitespace-only) -> single passthrough chunk
        chunks = [{"text": text.strip() or text, "start": 0, "end": len(text), "n_tokens": 0}]
    return chunks
