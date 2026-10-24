"""Embedding backends: a local model on the GPU, the OpenAI API, and a simple hashing one for offline tests."""

from __future__ import annotations

import hashlib
import re

import httpx
import numpy as np

from .config import Settings

# bge models search better when the query (but not the documents) gets this instruction in front
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _normalise(vectors: np.ndarray) -> np.ndarray:
    # unit-length vectors let me use a plain dot product as cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class Embedder:
    name: str = "base"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class LocalEmbedder(Embedder):
    """sentence-transformers model, which uses the NVIDIA GPU automatically when torch has CUDA."""

    def __init__(self, model_name: str):
        # I import here so the rest of the project still runs on machines without torch installed
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=32, show_progress_bar=len(texts) > 200, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([self.query_prefix + text])[0]


class OpenAIEmbedder(Embedder):
    """Calls the OpenAI embeddings endpoint in batches."""

    def __init__(self, model_name: str, api_key: str, base_url: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is empty, set it in .env or switch EMBED_BACKEND to local")
        self.name = model_name
        self.client = httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = []
        # the api accepts many inputs per call, so I batch to save time
        for start in range(0, len(texts), 128):
            resp = self.client.post("/embeddings", json={"model": self.name, "input": texts[start:start + 128]})
            resp.raise_for_status()
            out.extend(item["embedding"] for item in sorted(resp.json()["data"], key=lambda d: d["index"]))
        return _normalise(np.asarray(out, dtype=np.float32))


class HashEmbedder(Embedder):
    """Bag-of-words hashing trick. It's weak, but needs no model download, so the tests run anywhere."""

    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                # each word lands in a fixed bucket, so texts sharing words point in similar directions
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                vectors[row, bucket] += 1.0
        return _normalise(vectors)


def get_embedder(settings: Settings) -> Embedder:
    if settings.embed_backend == "local":
        return LocalEmbedder(settings.local_embed_model)
    if settings.embed_backend == "openai":
        return OpenAIEmbedder(settings.openai_embed_model, settings.openai_api_key, settings.openai_base_url)
    if settings.embed_backend == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown EMBED_BACKEND '{settings.embed_backend}'")
