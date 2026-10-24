"""Two chunking strategies so I can measure which one retrieves better."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from .ingest import Document


@dataclass
class Chunk:
    # the unit I embed, search over and cite
    id: str
    text: str
    bank: str
    year: int | None
    page: int
    section: str
    kind: str
    source: str

    def header(self) -> str:
        # this short label goes in front of every chunk so the embedding knows which report and section it's from
        year = self.year if self.year else "unknown year"
        section = f" | Section: {self.section}" if self.section else ""
        return f"[{self.bank} annual report {year} | page {self.page}{section}]"

    def to_dict(self) -> dict:
        return asdict(self)


def _make_id(source: str, page: int, text: str) -> str:
    # a stable id from the content, so the same chunk keeps the same id across rebuilds
    digest = hashlib.sha1(f"{source}|{page}|{text}".encode()).hexdigest()[:12]
    return f"{source.rsplit('.', 1)[0]}-p{page}-{digest}"


def _new_chunk(doc: Document, page: int, section: str, kind: str, body: str) -> Chunk:
    chunk = Chunk("", "", doc.bank, doc.year, page, section, kind, doc.source)
    # the stored text includes the header, which helps both embedding search and keyword search
    chunk.text = f"{chunk.header()}\n{body.strip()}"
    chunk.id = _make_id(doc.source, page, body)
    return chunk


def fixed_chunks(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Baseline: a sliding window of words over each page, ignoring headings and tables."""
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    chunks: list[Chunk] = []
    step = size - overlap
    # I group blocks by page so a chunk never spans two pages, which keeps page citations exact
    pages: dict[int, list[str]] = {}
    for block in doc.blocks:
        pages.setdefault(block.page, []).append(block.text)
    for page, texts in sorted(pages.items()):
        words = " ".join(texts).split()
        for start in range(0, max(len(words) - overlap, 1), step):
            window = words[start:start + size]
            if len(window) < 15:
                # very short leftovers are usually page furniture, not content
                continue
            chunks.append(_new_chunk(doc, page, "", "text", " ".join(window)))
    return chunks


def _split_table(md: str, size: int) -> list[str]:
    # big tables get split by rows, and I repeat the header row on every piece so each piece still makes sense alone
    lines = md.splitlines()
    if len(md.split()) <= size or len(lines) <= 3:
        return [md]
    header, divider, rows = lines[0], lines[1], lines[2:]
    pieces, current, current_words = [], [], 0
    for row in rows:
        row_words = len(row.split())
        if current and current_words + row_words > size:
            pieces.append("\n".join([header, divider, *current]))
            current, current_words = [], 0
        current.append(row)
        current_words += row_words
    if current:
        pieces.append("\n".join([header, divider, *current]))
    return pieces


def structured_chunks(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Layout-aware: tables stay whole, paragraphs are grouped under their heading, and the heading is kept as context."""
    chunks: list[Chunk] = []
    section = ""
    buffer: list[str] = []
    buffer_page = None
    # tracks whether the buffer holds anything beyond the carried-over overlap words
    has_new = False

    def flush():
        # turns the paragraphs collected so far into one chunk
        nonlocal buffer, has_new
        body = " ".join(buffer).strip()
        if has_new and buffer_page is not None and len(body.split()) >= 15:
            chunks.append(_new_chunk(doc, buffer_page, section, "text", body))
        # I keep the last few words as overlap so an answer split across two chunks isn't lost
        tail = body.split()[-overlap:] if (overlap and has_new) else []
        buffer = [" ".join(tail)] if tail else []
        has_new = False

    for block in doc.blocks:
        if block.kind == "heading":
            # a new heading starts a new section, so I close off the current chunk first
            flush()
            buffer = []
            section = block.text[:100]
            buffer_page = block.page
            continue

        if block.kind == "table":
            # a short line right before a table is usually its caption, so I attach it to the table instead of losing it
            caption = ""
            fresh_words = " ".join(buffer).split() if has_new else []
            if has_new and len(fresh_words) < 15 and buffer_page == block.page:
                caption = " ".join(fresh_words) + "\n"
                has_new = False
            flush()
            buffer = []
            # tables get their own chunks, tagged with the section they appear in
            for piece in _split_table(block.text, size):
                chunks.append(_new_chunk(doc, block.page, section, "table", caption + piece))
            buffer_page = block.page
            continue

        # a page change also closes the chunk, again to keep citations pointing at one page
        if buffer_page is not None and block.page != buffer_page:
            flush()
            buffer = []
        buffer_page = block.page

        words = block.text.split()
        # paragraphs longer than the chunk size get cut into windows on their own
        if len(words) > size:
            flush()
            buffer = []
            step = size - overlap
            for start in range(0, len(words), step):
                window = words[start:start + size]
                if len(window) >= 15:
                    chunks.append(_new_chunk(doc, block.page, section, "text", " ".join(window)))
            continue

        if has_new and sum(len(b.split()) for b in buffer) + len(words) > size:
            flush()
        buffer.append(block.text)
        has_new = True

    flush()
    return chunks


def chunk_documents(docs: list[Document], strategy: str, size: int, overlap: int) -> list[Chunk]:
    """Chunks every document with the chosen strategy and drops exact duplicates."""
    makers = {"fixed": fixed_chunks, "structured": structured_chunks}
    if strategy not in makers:
        raise ValueError(f"unknown chunk strategy '{strategy}', use one of {list(makers)}")
    seen, out = set(), []
    for doc in docs:
        for chunk in makers[strategy](doc, size, overlap):
            # duplicate ids mean identical text on the same page, which only wastes search slots
            if chunk.id not in seen:
                seen.add(chunk.id)
                out.append(chunk)
    return out
