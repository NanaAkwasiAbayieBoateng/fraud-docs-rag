from rag_app.chunking import STRATEGIES, chunk_text, fixed_size, recursive, structure_aware

TEXT = (
    "Fraud guidance intro paragraph with several words in it.\n\n"
    "Second paragraph talks about scams. It has two sentences. "
    "And a third one here to make it longer.\n\n"
    "Third paragraph covers reporting and reimbursement rules for victims."
)


def test_strategies_all_return_nonempty_chunks():
    for name in STRATEGIES:
        chunks = chunk_text(TEXT, strategy=name, chunk_size=120, overlap=20)
        assert chunks, f"{name} returned no chunks"
        assert all(c.strip() for c in chunks)


def test_fixed_size_respects_chunk_size_approximately():
    chunks = fixed_size(TEXT, chunk_size=60, overlap=10)
    assert len(chunks) > 1
    # Windows may end early at a whitespace boundary but never overrun much.
    assert all(len(c) <= 60 for c in chunks)


def test_fixed_size_short_text_single_chunk():
    assert fixed_size("hello world", chunk_size=100) == ["hello world"]


def test_fixed_size_empty_text():
    assert fixed_size("   ") == []


def test_recursive_keeps_sentence_boundaries():
    chunks = recursive(TEXT, chunk_size=90, overlap=10)
    assert len(chunks) >= 2
    joined = " ".join(chunks)
    assert "Second paragraph talks about scams." in joined


def test_structure_aware_splits_on_headings():
    doc = "# Title\n\nIntro text.\n\n## Section A\n\nAlpha body.\n\n## Section B\n\nBeta body."
    chunks = structure_aware(doc, chunk_size=800)
    assert len(chunks) == 3
    assert chunks[1].startswith("Title > Section A")
    assert "Alpha body." in chunks[1]
    assert chunks[2].startswith("Title > Section B")


def test_structure_aware_without_headings_falls_back():
    chunks = structure_aware("plain text with no headings at all", chunk_size=800)
    assert chunks == ["plain text with no headings at all"]


def test_unknown_strategy_raises():
    try:
        chunk_text(TEXT, strategy="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown strategy")
