"""Three chunking strategies, compared by the eval harness.

Every strategy takes raw document text and returns a list of chunk strings.
Keeping the interface identical (text in, list[str] out) is what lets
``scripts/run_eval.py`` swap strategies without touching anything else.
"""

from __future__ import annotations

import re

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 100


def fixed_size(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[str]:
    """Sliding character window with overlap; breaks on whitespace when close."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Prefer to end at the last whitespace inside the window.
            cut = text.rfind(" ", start + chunk_size // 2, end)
            if cut > start:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + step, end - overlap) if end - overlap > start else end
        if start <= 0 or start >= len(text):
            break
    return chunks


_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_pieces(text: str, separators: list[str]) -> list[str]:
    """Recursively split text into small pieces using the separator list."""
    if not separators:
        return [text] if text else []
    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split_pieces(text, rest)
    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        if piece.strip():
            pieces.append(piece)
    out: list[str] = []
    for piece in pieces:
        if len(piece) > 400 and rest:
            out.extend(_split_pieces(piece, rest))
        else:
            out.append(piece)
    return out


def recursive(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[str]:
    """Split on paragraph/sentence boundaries, then pack pieces into chunks."""
    text = text.strip()
    if not text:
        return []
    pieces = _split_pieces(text, _SEPARATORS)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current.strip():
                chunks.append(current.strip())
            # Carry a tail of the previous chunk forward as overlap.
            tail = current[-overlap:] if overlap and current else ""
            current = (tail + piece) if tail else piece
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size].strip())
                current = current[chunk_size - overlap :]
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def structure_aware(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[str]:
    """One chunk per heading section (heading path prepended).

    Sections longer than ``chunk_size`` fall back to fixed-size windows,
    each still prefixed with the section heading so context survives.
    Documents without headings degrade to a single fixed-size pass.
    """
    lines = text.strip().splitlines()
    sections: list[tuple[str, str]] = []
    heading_stack: list[str] = []
    current_heading = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_heading, body))

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush()
            buffer = []
            level = len(m.group(1))
            title = m.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1] + [title]
            current_heading = " > ".join(heading_stack)
        else:
            buffer.append(line)
    flush()

    if not sections:
        return fixed_size(text, chunk_size, overlap)

    chunks: list[str] = []
    for heading, body in sections:
        prefix = f"{heading}\n" if heading else ""
        full = prefix + body
        if len(full) <= chunk_size:
            chunks.append(full)
        else:
            for piece in fixed_size(body, chunk_size, overlap):
                chunks.append(prefix + piece)
    return [c for c in chunks if c.strip()]


STRATEGIES = {
    "fixed_size": fixed_size,
    "recursive": recursive,
    "structure_aware": structure_aware,
}


def chunk_text(
    text: str,
    strategy: str = "fixed_size",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown chunking strategy {strategy!r}; choose from {sorted(STRATEGIES)}"
        )
    return STRATEGIES[strategy](text, chunk_size=chunk_size, overlap=overlap)
