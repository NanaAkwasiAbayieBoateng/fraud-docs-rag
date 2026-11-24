"""Document loading: turn common file formats into plain text.

Ingestion accepts the formats fraud/AML guidance actually arrives in —
Markdown and plain text, PDF, Word (.docx), Excel (.xlsx), and CSV —
and normalises each to text the chunker can work on. Tabular formats
(CSV, Excel) are rendered one row per line as ``header: value`` pairs,
and PDFs/Word docs keep their structure as ``##`` headings where the
format exposes it, so the structure-aware chunker still has something
to hold on to.

Legacy binary formats (.doc, .xls) are not supported — convert them to
.docx/.xlsx first; the libraries used here do not read them.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".docx", ".xlsx", ".csv"}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            parts.append(f"## Page {i + 1}\n\n{text}")
    return "\n\n".join(parts)


def _read_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _rows_to_text(headers: list[str], rows) -> list[str]:
    """Render data rows as 'header: value; header: value' lines."""
    lines = []
    for row in rows:
        cells = ["" if c is None else str(c).strip() for c in row]
        if not any(cells):
            continue
        pairs = [
            f"{headers[i]}: {value}" if i < len(headers) and headers[i] else value
            for i, value in enumerate(cells)
            if value
        ]
        if pairs:
            lines.append("; ".join(pairs))
    return lines


def _read_csv(path: Path) -> str:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(raw)))
    if not rows:
        return ""
    headers = [h.strip() for h in rows[0]]
    return "\n".join(_rows_to_text(headers, rows[1:]))


def _read_xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    parts = []
    for ws in wb.worksheets:
        rows = ws.iter_rows(values_only=True)
        first = next(rows, None)
        if first is None:
            continue
        headers = ["" if c is None else str(c).strip() for c in first]
        lines = _rows_to_text(headers, rows)
        if lines:
            parts.append(f"## Sheet: {ws.title}\n\n" + "\n".join(lines))
    wb.close()
    return "\n\n".join(parts)


EXTRACTORS = {
    ".md": _read_text_file,
    ".txt": _read_text_file,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".xlsx": _read_xlsx,
    ".csv": _read_csv,
}


def extract_text(path: str | Path) -> str:
    """Extract plain text from a supported document file."""
    path = Path(path)
    extractor = EXTRACTORS.get(path.suffix.lower())
    if extractor is None:
        raise ValueError(
            f"unsupported document format: {path.suffix or '(none)'} "
            f"(supported: {', '.join(sorted(SUPPORTED_SUFFIXES))})"
        )
    return extractor(path)
