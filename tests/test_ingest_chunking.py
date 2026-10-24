from pathlib import Path

import pytest

from citebank.chunking import _split_table, chunk_documents
from citebank.ingest import parse_filename


def test_parse_filename():
    assert parse_filename(Path("dbs_2025.pdf")) == ("DBS", 2025)
    assert parse_filename(Path("OCBC-Annual-Report-2024.pdf")) == ("OCBC", 2024)
    assert parse_filename(Path("uob-ar25.pdf"))[0] == "UOB"
    assert parse_filename(Path("demo_2025.pdf")) == ("DEMO", 2025)


def test_tables_and_headings_are_found(docs):
    doc = next(d for d in docs if d.year == 2025)
    kinds = [b.kind for b in doc.blocks]
    assert "table" in kinds
    assert any(b.kind == "heading" and b.text == "Financial Highlights" for b in doc.blocks)
    table = next(b for b in doc.blocks if b.kind == "table")
    assert "4.57" in table.text and table.page == 2


def test_running_header_and_page_numbers_removed(docs):
    # "Demo Bank Annual Report 2025" is printed on every page and should be stripped as page furniture
    for doc in docs:
        assert not any(b.text.startswith(f"Demo Bank Annual Report {doc.year}") for b in doc.blocks)


def test_heading_comes_before_its_table(docs):
    doc = next(d for d in docs if d.year == 2025)
    page2 = [b.kind for b in doc.blocks if b.page == 2]
    assert page2.index("heading") < page2.index("table")


def test_structured_chunks_keep_table_with_caption_and_section(docs):
    chunks = chunk_documents(docs, "structured", 220, 40)
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 2
    for t in tables:
        assert t.section == "Financial Highlights"
        assert "table below summarises" in t.text
        assert t.text.startswith(f"[DEMO annual report {t.year} | page 2")


def test_chunks_never_span_pages(docs):
    # every word in a chunk body must appear on the page the chunk claims, otherwise its citation would be wrong
    page_words = {}
    for d in docs:
        for b in d.blocks:
            page_words.setdefault((d.source, b.page), set()).update(b.text.split())
    for strategy in ["fixed", "structured"]:
        for c in chunk_documents(docs, strategy, 30, 5):
            body = c.text.split("\n", 1)[1].split()
            assert set(body) <= page_words[(c.source, c.page)], (strategy, c.page, c.text[:80])


def test_fixed_chunk_overlap_and_validation(docs):
    with pytest.raises(ValueError):
        chunk_documents(docs, "fixed", 20, 20)
    with pytest.raises(ValueError):
        chunk_documents(docs, "semantic", 200, 20)


def test_chunk_ids_are_unique_and_stable(docs):
    a = [c.id for c in chunk_documents(docs, "structured", 220, 40)]
    b = [c.id for c in chunk_documents(docs, "structured", 220, 40)]
    assert a == b and len(set(a)) == len(a)


def test_large_table_split_repeats_header():
    rows = "\n".join(f"|row {i}|{i}|{i * 2}|" for i in range(40))
    md = "|Metric|2025|2024|\n|---|---|---|\n" + rows
    pieces = _split_table(md, size=30)
    assert len(pieces) > 1
    assert all(p.startswith("|Metric|2025|2024|\n|---|---|---|") for p in pieces)
    # every original row lands in exactly one piece
    assert sum(p.count("|row ") for p in pieces) == 40


def test_spread_page_reads_left_half_first(tmp_path):
    # a wide page with two printed pages side by side should not interleave their lines
    import pymupdf

    from citebank.ingest import load_pdf

    doc = pymupdf.open()
    page = doc.new_page(width=1200, height=800)
    for y in (100, 200, 300):
        page.insert_text((60, y), f"left line {y}", fontsize=10)
        page.insert_text((660, y), f"right line {y}", fontsize=10)
    path = tmp_path / "demo_2025.pdf"
    doc.save(path)

    texts = [b.text for b in load_pdf(path, extract_tables=False).blocks]
    assert texts == ["left line 100", "left line 200", "left line 300",
                     "right line 100", "right line 200", "right line 300"]
