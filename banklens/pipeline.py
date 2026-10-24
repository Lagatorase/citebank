"""The full question-answering flow: retrieve sources, prompt the LLM, then check and attach citations."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from .config import Settings, get_settings
from .embeddings import Embedder, get_embedder
from .indexer import load_or_build
from .llm import LLM, get_llm
from .prompts import REFUSAL, SYSTEM_PROMPT, build_user_prompt, chunk_body
from .retrieval import Hit, Reranker, Retriever

CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Citation:
    number: int
    bank: str
    year: int | None
    page: int
    section: str
    snippet: str


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    citations: list[Citation] = field(default_factory=list)
    # citation numbers the model made up that don't match any source; should always be empty
    invalid_citations: list[int] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    model: str = ""
    retrieval_s: float = 0.0
    generation_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def is_refusal(text: str) -> bool:
    # I match on the key part of the phrase, since models sometimes add a sentence around it
    # small models also reword the end, e.g. "in the provided annual reports", so I allow that too
    return bool(re.search(r"could not find this in the (provided )?annual reports", text.lower()))


def parse_citations(text: str, hits: list[Hit]) -> tuple[list[Citation], list[int]]:
    """Reads [n] markers from the answer and links each one back to the chunk it refers to."""
    numbers = sorted({int(n) for n in CITATION_RE.findall(text)})
    valid = [n for n in numbers if 1 <= n <= len(hits)]
    invalid = [n for n in numbers if n not in valid]
    citations = []
    for n in valid:
        c = hits[n - 1].chunk
        citations.append(Citation(n, c.bank, c.year, c.page, c.section, chunk_body(hits[n - 1])[:300]))
    return citations, invalid


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None, llm: LLM | None = None,
                 reranker: Reranker | None = None, auto_filter: bool = True, store=None):
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder(self.settings)
        self.store = store or load_or_build(self.settings, self.embedder)
        # the reranker is optional because it needs a second model download and adds latency
        if reranker is None and self.settings.use_reranker:
            reranker = Reranker(self.settings.rerank_model)
        self.retriever = Retriever(self.store, self.embedder, self.settings, reranker, auto_filter)
        self.llm = llm or get_llm(self.settings)

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        return self.retriever.search(question, k)

    def ask(self, question: str, k: int | None = None) -> Answer:
        t0 = time.perf_counter()
        hits = self.retrieve(question, k)
        retrieval_s = time.perf_counter() - t0
        sources = [{"bank": h.chunk.bank, "year": h.chunk.year, "page": h.chunk.page, "section": h.chunk.section,
                    "kind": h.chunk.kind, "score": round(h.score, 4), "text": chunk_body(h)} for h in hits]

        # with nothing retrieved there's no point asking the model, it could only guess
        if not hits:
            return Answer(question, REFUSAL, True, sources=sources, model=self.llm.name, retrieval_s=retrieval_s)

        result = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, hits))
        text = result.text.strip()
        citations, invalid = parse_citations(text, hits)
        return Answer(question, text, is_refusal(text), citations, invalid, sources, result.model,
                      retrieval_s, result.latency_s)
