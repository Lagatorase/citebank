"""Builds (or loads) the search index for a given set of settings."""

from __future__ import annotations

import time

from .chunking import chunk_documents
from .config import Settings
from .embeddings import Embedder, get_embedder
from .ingest import load_folder
from .store import VectorStore


def build_index(settings: Settings, embedder: Embedder | None = None, verbose: bool = True) -> VectorStore:
    """Parses the PDFs, chunks them, embeds every chunk and saves the index to disk."""
    start = time.perf_counter()
    docs = load_folder(settings.raw_dir)
    if verbose:
        for d in docs:
            tables = sum(b.kind == "table" for b in d.blocks)
            print(f"  read {d.source}: {d.page_count} pages, {len(d.blocks)} blocks, {tables} tables")

    chunks = chunk_documents(docs, settings.chunk_strategy, settings.chunk_size, settings.chunk_overlap)
    if verbose:
        print(f"  made {len(chunks)} chunks with the '{settings.chunk_strategy}' strategy")

    # I reuse an embedder if one is passed in, since loading the model is the slow part
    embedder = embedder or get_embedder(settings)
    vectors = embedder.embed_documents([c.text for c in chunks])

    store = VectorStore(chunks, vectors, embedder.name)
    store.save(settings.index_path())
    if verbose:
        print(f"  saved index to {settings.index_path()} in {time.perf_counter() - start:.1f}s")
    return store


def load_or_build(settings: Settings, embedder: Embedder | None = None, rebuild: bool = False) -> VectorStore:
    # loads an existing index for these settings, or builds one if it's missing
    path = settings.index_path()
    if not rebuild and (path / "chunks.jsonl").exists():
        return VectorStore.load(path)
    return build_index(settings, embedder)
