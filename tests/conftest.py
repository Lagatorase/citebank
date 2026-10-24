import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from banklens.chunking import chunk_documents  # noqa: E402
from banklens.config import Settings  # noqa: E402
from banklens.embeddings import HashEmbedder  # noqa: E402
from banklens.ingest import load_folder  # noqa: E402
from banklens.store import VectorStore  # noqa: E402


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory):
    # builds the two fictional demo reports once for the whole test run
    spec = importlib.util.spec_from_file_location("make_sample", ROOT / "scripts" / "make_sample_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    folder = tmp_path_factory.mktemp("raw")
    for year in module.FIGURES:
        module.build(year, folder)
    return folder


@pytest.fixture(scope="session")
def docs(sample_dir):
    return load_folder(sample_dir)


@pytest.fixture(scope="session")
def settings(sample_dir, tmp_path_factory):
    return Settings(raw_dir=sample_dir, index_dir=tmp_path_factory.mktemp("index"), embed_backend="hash",
                    use_reranker=False, llm_backend="echo", top_k=3)


@pytest.fixture(scope="session")
def store(docs):
    chunks = chunk_documents(docs, "structured", 220, 40)
    embedder = HashEmbedder()
    return VectorStore(chunks, embedder.embed_documents([c.text for c in chunks]), embedder.name)
