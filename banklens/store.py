"""A small on-disk index: chunk metadata in JSONL, vectors in a NumPy file, and BM25 rebuilt on load."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk

# common words that only add noise to keyword matching
STOPWORDS = set(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "what which who how did does do our we their they".split()
)


def tokenize(text: str) -> list[str]:
    # I keep numbers like 10.3 and 2025 intact because financial questions often hinge on them
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower().replace(",", ""))
    return [t for t in tokens if t not in STOPWORDS]


@dataclass
class VectorStore:
    chunks: list[Chunk]
    vectors: np.ndarray
    embedder_name: str

    def __post_init__(self):
        if len(self.chunks) != len(self.vectors):
            raise ValueError("chunk count and vector count don't match, rebuild the index")
        # bm25 is fast to build, so I rebuild it on load instead of saving it
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks]) if self.chunks else None

    # saving and loading

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        np.save(folder / "vectors.npy", self.vectors)
        (folder / "meta.json").write_text(json.dumps({"embedder": self.embedder_name, "count": len(self.chunks)}))

    @classmethod
    def load(cls, folder: Path) -> "VectorStore":
        if not (folder / "chunks.jsonl").exists():
            raise FileNotFoundError(f"No index at {folder}. Run: python -m banklens.cli build")
        with open(folder / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        vectors = np.load(folder / "vectors.npy")
        meta = json.loads((folder / "meta.json").read_text())
        return cls(chunks, vectors, meta["embedder"])

    # searching

    def dense_search(self, query_vector: np.ndarray, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        # vectors are unit length, so the dot product is the cosine similarity
        scores = self.vectors @ query_vector
        if allowed is not None:
            scores = np.where(allowed, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if np.isfinite(scores[i])]

    def keyword_search(self, query: str, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens or self.bm25 is None:
            return []
        scores = np.asarray(self.bm25.get_scores(tokens))
        if allowed is not None:
            scores = np.where(allowed, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        # a zero bm25 score means no query words matched at all, so it's not a real hit
        return [(int(i), float(scores[i])) for i in top if np.isfinite(scores[i]) and scores[i] > 0]

    def metadata_mask(self, banks: set[str] | None = None, years: set[int] | None = None) -> np.ndarray | None:
        # returns a true/false array marking which chunks match the bank and year filters
        if not banks and not years:
            return None
        mask = np.ones(len(self.chunks), dtype=bool)
        if banks:
            mask &= np.array([c.bank in banks for c in self.chunks])
        if years:
            mask &= np.array([c.year in years for c in self.chunks])
        return mask

    def banks(self) -> set[str]:
        return {c.bank for c in self.chunks}

    def years(self) -> set[int]:
        return {c.year for c in self.chunks if c.year}
