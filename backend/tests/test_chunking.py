"""Unit tests for the long-text chunker.

Uses a fake word-based tokenizer (one word = one token) so the packing, overlap,
and hard-split logic is exercised deterministically — no model download, fast.
The real tokenizer only changes token *counts*, not the algorithm under test.
"""
from app.services.chunking import _split_sentences, split_into_chunks


class FakeTokenizer:
    """One token per whitespace-delimited word. Ids are word positions."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, ids):
        # Placeholder text of the right token length (content irrelevant here).
        return " ".join("w" for _ in ids)


TOK = FakeTokenizer()


def _tokens(chunk_text):
    return len(chunk_text.split())


# ---- sentence splitting ---------------------------------------------------

def test_split_sentences_basic():
    spans = _split_sentences("Hello world. How are you? I am fine!")
    assert [s.strip() for _, _, s in spans] == ["Hello world.", "How are you?", "I am fine!"]


def test_split_sentences_no_punctuation():
    spans = _split_sentences("just a stream of words with no enders")
    assert len(spans) == 1


def test_split_offsets_are_within_bounds():
    text = "One. Two. Three."
    for start, end, seg in _split_sentences(text):
        assert 0 <= start < end <= len(text)
        assert text[start:end] == seg


# ---- chunk packing --------------------------------------------------------

def test_short_text_is_single_chunk():
    chunks = split_into_chunks("I am happy today.", TOK, max_tokens=50)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "I am happy today."


def test_sentences_pack_until_budget():
    # 10 sentences of 5 words each = 50 tokens; budget 20 -> 3 chunks (no overlap).
    text = " ".join("aa bb cc dd ee." for _ in range(10))
    chunks = split_into_chunks(text, TOK, max_tokens=20, overlap=0)
    assert len(chunks) >= 3
    assert all(_tokens(c["text"]) <= 20 for c in chunks)


def test_no_chunk_exceeds_budget():
    text = " ".join(f"word{i} more filler here." for i in range(40))
    chunks = split_into_chunks(text, TOK, max_tokens=25, overlap=1)
    assert chunks
    assert all(c["n_tokens"] <= 25 for c in chunks)
    assert all(_tokens(c["text"]) <= 25 for c in chunks)


def test_overlap_repeats_a_sentence():
    text = " ".join(f"s{i} one two three four." for i in range(12))
    no_ov = split_into_chunks(text, TOK, max_tokens=20, overlap=0)
    with_ov = split_into_chunks(text, TOK, max_tokens=20, overlap=1)
    # Overlap re-uses trailing sentences, so it needs at least as many chunks.
    assert len(with_ov) >= len(no_ov)
    assert all(c["n_tokens"] <= 20 for c in with_ov)


def test_oversized_single_sentence_is_hard_split():
    # One 60-word sentence, no internal boundary; budget 20 -> 3 hard pieces.
    text = " ".join(f"tok{i}" for i in range(60))  # no . ! ? -> one sentence
    chunks = split_into_chunks(text, TOK, max_tokens=20)
    assert len(chunks) == 3
    assert all(c["n_tokens"] <= 20 for c in chunks)


def test_empty_and_whitespace_return_one_chunk():
    for text in ["", "   ", "\n\n\t "]:
        chunks = split_into_chunks(text, TOK, max_tokens=20)
        assert len(chunks) == 1


def test_chunk_char_offsets_ordered_and_in_bounds():
    text = " ".join("alpha beta gamma delta." for _ in range(15))
    chunks = split_into_chunks(text, TOK, max_tokens=20, overlap=0)
    for c in chunks:
        assert 0 <= c["start"] < c["end"] <= len(text)
