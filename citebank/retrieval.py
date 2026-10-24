"""Finds the chunks most likely to answer a question: dense, keyword (BM25) or hybrid search, plus optional reranking."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .chunking import Chunk
from .config import Settings
from .embeddings import Embedder
from .store import VectorStore

# ways people refer to each bank in a question
BANK_ALIASES = {
    "DBS": ["dbs", "development bank of singapore"],
    "OCBC": ["ocbc", "oversea-chinese", "oversea chinese"],
    "UOB": ["uob", "united overseas bank"],
}


# banking acronyms that reports often spell out in full; keyword search can't link the two forms by itself
FINANCE_ACRONYMS = {
    "cet1": "common equity tier 1",
    "nim": "net interest margin",
    "roe": "return on equity",
    "roa": "return on assets",
    "npl": "non-performing loans",
    "npa": "non-performing assets",
    "lcr": "liquidity coverage ratio",
    "nsfr": "net stable funding ratio",
    "eps": "earnings per share",
    "aum": "assets under management",
    "esg": "environmental social governance",
    "rwa": "risk-weighted assets",
}


def expand_acronyms(query: str) -> str:
    # adds the spelled-out form next to any acronym, so "CET1" also matches "Common Equity Tier 1"
    extra = [full for short, full in FINANCE_ACRONYMS.items() if re.search(rf"\b{short}\b", query.lower())]
    return query + (" " + " ".join(extra) if extra else "")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    # which search produced it, which helps when I'm debugging bad answers
    source: str


def detect_filters(query: str, store: VectorStore) -> tuple[set[str], set[int]]:
    """Picks out bank names and years mentioned in the question, limited to ones that exist in the index."""
    q = query.lower()
    aliases = dict(BANK_ALIASES)
    # any other bank in the index (like the demo reports) matches by its own name
    for bank in store.banks():
        aliases.setdefault(bank, [bank.lower()])
    banks = {bank for bank, names in aliases.items() if bank in store.banks() and any(re.search(rf"\b{re.escape(n)}\b", q) for n in names)}
    years = {int(y) for y in re.findall(r"\b(20\d{2})\b", q)} & store.years()
    return banks, years


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merges ranked lists by position instead of raw score, since BM25 and cosine scores aren't on the same scale."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            # higher positions count for more; k=60 is the standard value from the original RRF paper
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: -item[1])


class Reranker:
    """Cross-encoder that reads the question and each chunk together, which is slower but more accurate than vectors."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        scores = self.model.predict([(query, h.chunk.text) for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda pair: -float(pair[1]))
        return [Hit(h.chunk, float(s), h.source + "+rerank") for h, s in ranked]


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, settings: Settings, reranker: Reranker | None = None, auto_filter: bool = True):
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker
        self.auto_filter = auto_filter

    def search(self, query: str, k: int | None = None, mode: str | None = None) -> list[Hit]:
        k = k or self.settings.top_k
        mode = mode or self.settings.retrieval_mode
        # with a reranker I fetch a wider net first, then let the reranker pick the best k
        pool = self.settings.candidate_k if self.reranker else k

        # if the question names a bank or year, I only search those reports
        allowed = None
        if self.auto_filter:
            banks, years = detect_filters(query, self.store)
            allowed = self.store.metadata_mask(banks, years)
            # if the filter removes everything, I fall back to searching all reports instead of returning nothing
            if allowed is not None and not allowed.any():
                allowed = None

        if mode == "dense":
            results = self.store.dense_search(self.embedder.embed_query(query), pool, allowed)
            hits = [Hit(self.store.chunks[i], s, "dense") for i, s in results]
        elif mode == "bm25":
            results = self.store.keyword_search(expand_acronyms(query), pool, allowed)
            hits = [Hit(self.store.chunks[i], s, "bm25") for i, s in results]
        elif mode == "hybrid":
            # dense catches paraphrases, bm25 catches exact terms like "CET1" or "2025", so I combine both.
            # each list goes deeper than k: if a chunk is first in one list but missing from the other's short list,
            # fusion would wrongly rank it below chunks that are mediocre in both
            depth = max(pool, self.settings.candidate_k)
            dense = self.store.dense_search(self.embedder.embed_query(query), depth, allowed)
            sparse = self.store.keyword_search(expand_acronyms(query), depth, allowed)
            fused = reciprocal_rank_fusion([[i for i, _ in dense], [i for i, _ in sparse]])[:pool]
            hits = [Hit(self.store.chunks[i], s, "hybrid") for i, s in fused]
        else:
            raise ValueError(f"unknown retrieval mode '{mode}', use dense, bm25 or hybrid")

        if self.reranker:
            hits = self.reranker.rerank(query, hits)
        return hits[:k]
