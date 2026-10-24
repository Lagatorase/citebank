"""Turns annual report PDFs into clean, page-tagged blocks of headings, text and tables."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# file names like dbs_2025.pdf or ocbc-2024-annual-report.pdf tell me the bank and year
KNOWN_BANKS = {"dbs": "DBS", "ocbc": "OCBC", "uob": "UOB"}
YEAR_RE = re.compile(r"(20\d{2})")


@dataclass
class Block:
    # one piece of a page: a heading, a paragraph or a table already converted to markdown
    kind: str
    text: str
    page: int


@dataclass
class Document:
    # one annual report with everything I pulled out of it
    source: str
    bank: str
    year: int | None
    blocks: list[Block] = field(default_factory=list)
    page_count: int = 0


def parse_filename(path: Path) -> tuple[str, int | None]:
    # I read the bank from the file name, and fall back to the file stem if it's not one of the three banks
    stem = path.stem.lower()
    # for other files I use the first word of the name, e.g. demo_2025.pdf becomes DEMO
    fallback = re.split(r"[_\-\s\d]+", stem)[0].upper() or stem.upper()
    bank = next((label for key, label in KNOWN_BANKS.items() if key in stem), fallback)
    match = YEAR_RE.search(stem)
    return bank, int(match.group(1)) if match else None


def _normalise_line(line: str) -> str:
    # page numbers change on every page, so I swap digits out before comparing header and footer lines
    return re.sub(r"\d+", "#", line.strip().lower())


def _find_repeated_lines(page_lines: list[list[str]], threshold: float = 0.4) -> set[str]:
    # running headers and footers (like "DBS Annual Report 2025 | 47") show up on most pages and only add noise
    if len(page_lines) < 5:
        return set()
    counts = Counter()
    for lines in page_lines:
        # each line only counts once per page
        counts.update({_normalise_line(l) for l in lines if 0 < len(l.strip()) < 80})
    cutoff = threshold * len(page_lines)
    return {line for line, n in counts.items() if n >= cutoff}


def _inside(rect: tuple, bbox: tuple) -> bool:
    # checks whether a text block's centre sits inside a table area, so I don't read table cells twice
    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx0, by0, bx1, by1 = bbox
    return bx0 <= cx <= bx1 and by0 <= cy <= by1


def _is_spread(page: pymupdf.Page) -> bool:
    # a landscape page much wider than it is tall is two portrait pages side by side
    return page.rect.width > page.rect.height * 1.2


def _body_font_size(doc: pymupdf.Document, sample_pages: int = 40) -> float:
    # the most common font size (weighted by characters) is the body text size; headings are bigger than this
    sizes = Counter()
    for page in list(doc)[:sample_pages]:
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _table_to_markdown(table) -> str:
    # pymupdf can export straight to markdown; I fall back to building it myself if that fails
    try:
        md = table.to_markdown(clean=True)
        if md and md.strip():
            return md.strip()
    except Exception:
        pass
    rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in table.extract()]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def load_pdf(path: Path, extract_tables: bool = True) -> Document:
    """Reads one PDF into a Document of headings, paragraphs and tables, each tagged with its page number."""
    bank, year = parse_filename(path)
    doc = pymupdf.open(path)
    body_size = _body_font_size(doc)

    # first pass: collect raw lines from every page so I can spot repeated headers and footers
    raw_pages = []
    for page in doc:
        raw_pages.append([l for l in page.get_text("text").splitlines() if l.strip()])
    repeated = _find_repeated_lines(raw_pages)

    out = Document(source=path.name, bank=bank, year=year, page_count=len(doc))
    for page_index, page in enumerate(doc):
        page_no = page_index + 1  # people cite pages starting from 1, not 0

        # tables first, because financial figures in annual reports mostly live in tables
        tables: list[tuple[tuple, Block]] = []
        if extract_tables:
            try:
                for table in page.find_tables().tables:
                    md = _table_to_markdown(table)
                    # I skip tiny "tables" that are really just layout boxes
                    if md and md.count("\n") >= 2:
                        tables.append((tuple(table.bbox), Block("table", md, page_no)))
            except Exception:
                # some pages have odd layouts that break table detection; the text pass below still covers them
                tables = []
        table_boxes = [box for box, _ in tables]

        # some reports (DBS) put two printed pages side by side on one wide pdf page, so I read each half on its own
        # (pymupdf can merge lines from both halves into one block when they sit at the same height)
        r = page.rect
        half = (r.x0 + r.x1) / 2
        clips = [pymupdf.Rect(r.x0, r.y0, half, r.y1), pymupdf.Rect(half, r.y0, r.x1, r.y1)] if _is_spread(page) else [None]

        # then normal text, skipping anything that sits inside a table I already captured
        # I keep pymupdf's own block order, which follows multi-column layouts; sorting by height interleaves columns
        texts: list[tuple[tuple, Block]] = []
        for clip in clips:
            for b in page.get_text("dict", clip=clip)["blocks"]:
                if b.get("type") != 0 or any(_inside(b["bbox"], box) for box in table_boxes):
                    continue
                for kind, text in _split_block(b, body_size, repeated):
                    texts.append((b["bbox"], Block(kind, text, page_no)))

        # each table goes just before the first text below it in the same column, so headings come before their tables
        placed = [(float(i), block) for i, (_, block) in enumerate(texts)]
        for box, block in tables:
            after = next((i for i, (bb, _) in enumerate(texts)
                          if bb[1] >= box[1] and bb[0] < box[2] and bb[2] > box[0]), len(texts))
            placed.append((after - 0.5, block))
        # the sort is stable, so tables placed before the same text keep their top to bottom order
        placed.sort(key=lambda item: item[0])
        out.blocks.extend(block for _, block in placed)
    doc.close()
    return out


def _split_block(block: dict, body_size: float, repeated: set[str]) -> list[tuple[str, str]]:
    # a pdf "block" can mix a heading line with body lines, so I split it line by line by font size
    results: list[tuple[str, str]] = []
    current_kind, current_lines = None, []
    for line in block.get("lines", []):
        text = "".join(span["text"] for span in line["spans"]).strip()
        if not text or _normalise_line(text) in repeated:
            continue
        max_size = max(span["size"] for span in line["spans"])
        # a heading is noticeably larger than body text, short, and not just a number
        is_heading = max_size >= body_size * 1.25 and len(text) < 120 and not re.fullmatch(r"[\d.,%$()\s-]+", text)
        kind = "heading" if is_heading else "text"
        if kind != current_kind and current_lines:
            results.append((current_kind, _join(current_lines)))
            current_lines = []
        current_kind = kind
        current_lines.append(text)
    if current_lines:
        results.append((current_kind, _join(current_lines)))
    return results


def _join(lines: list[str]) -> str:
    # pdfs break words with hyphens at line ends, so I glue those back together
    text = " ".join(lines)
    text = re.sub(r"(\w)- (\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def load_folder(folder: Path, extract_tables: bool = True) -> list[Document]:
    """Loads every PDF in a folder, sorted so runs are repeatable."""
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {folder}. See the README for which reports to download.")
    return [load_pdf(p, extract_tables=extract_tables) for p in pdfs]
