"""Document-format extraction: one test per supported format.

Fixtures are generated with the same libraries that read them (docx,
openpyxl), written directly (csv/txt/md), or committed as a real file
(tests/fixtures/guidance.pdf, a two-page PDF) — extraction is tested
against genuine files, not mocks of the parsers.
"""

from pathlib import Path

import pytest

from rag_app.documents import extract_text
from rag_app.ingest import load_documents

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "guidance.pdf"


def test_txt_and_md_pass_through(tmp_path):
    for suffix in (".txt", ".md"):
        p = tmp_path / f"doc{suffix}"
        p.write_text("# Heading\n\nMoney mules move stolen money.", encoding="utf-8")
        assert "Money mules move stolen money." in extract_text(p)


def test_csv_rows_rendered_with_headers(tmp_path):
    p = tmp_path / "codes.csv"
    p.write_text("Code,Meaning\n10.4,Card-not-present fraud\n", encoding="utf-8")
    text = extract_text(p)
    assert "Code: 10.4" in text
    assert "Meaning: Card-not-present fraud" in text


def test_docx_paragraphs_and_tables(tmp_path):
    from docx import Document

    doc = Document()
    doc.add_paragraph("Account takeover warning signs include unexpected password resets.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Signal"
    table.rows[0].cells[1].text = "Unexpected MFA prompt"
    p = tmp_path / "guide.docx"
    doc.save(str(p))
    text = extract_text(p)
    assert "Account takeover warning signs" in text
    assert "Signal | Unexpected MFA prompt" in text


def test_xlsx_sheets_rendered_empty_sheet_skipped(tmp_path):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Reason codes"
    ws.append(["Code", "Meaning"])
    ws.append(["13.1", "Merchandise not received"])
    wb.create_sheet("Empty")
    p = tmp_path / "codes.xlsx"
    wb.save(str(p))
    text = extract_text(p)
    assert "## Sheet: Reason codes" in text
    assert "Code: 13.1" in text
    assert "Meaning: Merchandise not received" in text
    assert "## Sheet: Empty" not in text


def test_pdf_text_extracted_with_page_headings():
    text = extract_text(FIXTURE_PDF)
    assert "## Page 1" in text
    assert "Chargeback Fraud Guidance" in text
    assert "## Page 2" in text
    assert "3-D Secure liability shift" in text


def test_unsupported_format_raises(tmp_path):
    p = tmp_path / "legacy.doc"
    p.write_bytes(b"\xd0\xcf\x11\xe0")
    with pytest.raises(ValueError, match="unsupported document format"):
        extract_text(p)


def test_load_documents_skips_corrupt_supported_file(tmp_path, capsys):
    (tmp_path / "good.md").write_text("Money mules move stolen money.", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"this is not a pdf")
    docs = load_documents(tmp_path)
    assert [name for name, _ in docs] == ["good.md"]
    assert "skipping broken.pdf" in capsys.readouterr().err
