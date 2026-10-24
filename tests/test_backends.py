"""Checks the API request and response handling for each backend without calling any real service."""

import json
import sys
import types

import httpx
import numpy as np

from banklens.chunking import Chunk
from banklens.embeddings import BGE_QUERY_PREFIX, LocalEmbedder, OpenAIEmbedder
from banklens.llm import AnthropicLLM, OllamaLLM, OpenAILLM
from banklens.retrieval import Hit, Reranker


def _client(base_url, handler, headers=None):
    return httpx.Client(base_url=base_url, transport=httpx.MockTransport(handler), headers=headers or {})


def test_ollama_request_and_parse():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "Answer [1]"}, "prompt_eval_count": 50, "eval_count": 5})

    llm = OllamaLLM("http://x", "qwen2.5:7b")
    llm.client = _client("http://x", handler)
    out = llm.generate("sys", "user")
    assert out.text == "Answer [1]" and out.input_tokens == 50
    assert seen["messages"][0] == {"role": "system", "content": "sys"} and seen["stream"] is False


def test_openai_request_and_parse():
    def handler(request):
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hi"}}], "usage": {"prompt_tokens": 9, "completion_tokens": 1}})

    llm = OpenAILLM("key", "gpt-4o-mini", "http://x/v1")
    llm.client = _client("http://x/v1", handler)
    assert llm.generate("s", "u").output_tokens == 1


def test_anthropic_request_and_parse():
    def handler(request):
        body = json.loads(request.content)
        # the system prompt goes in its own field for this api, not in the messages list
        assert body["system"] == "s" and body["messages"] == [{"role": "user", "content": "u"}]
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Done [2]"}], "usage": {"input_tokens": 3, "output_tokens": 2}})

    llm = AnthropicLLM("key", "claude-haiku-5-5")
    llm.client = _client("https://api.anthropic.com", handler)
    assert llm.generate("s", "u").text == "Done [2]"


def test_openai_embedder_batches_and_normalises():
    calls = []

    def handler(request):
        inputs = json.loads(request.content)["input"]
        calls.append(len(inputs))
        # return in reverse order to check I sort by index
        data = [{"index": i, "embedding": [float(i + 1), 0.0]} for i in range(len(inputs))][::-1]
        return httpx.Response(200, json={"data": data})

    emb = OpenAIEmbedder("text-embedding-3-small", "key", "http://x/v1")
    emb.client = _client("http://x/v1", handler)
    vecs = emb.embed_documents([f"t{i}" for i in range(130)])
    assert calls == [128, 2] and vecs.shape == (130, 2)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def _fake_sentence_transformers(monkeypatch):
    # stands in for the real library so these tests don't need torch or a model download
    module = types.ModuleType("sentence_transformers")

    class FakeST:
        last = None

        def __init__(self, name):
            self.name = name

        def encode(self, texts, **kwargs):
            FakeST.last = texts
            return np.ones((len(texts), 4), dtype=np.float32) / 2

    class FakeCE:
        def __init__(self, name):
            pass

        def predict(self, pairs):
            # scores each chunk by how often "profit" appears, so the order is predictable
            return [text.count("profit") for _, text in pairs]

    module.SentenceTransformer, module.CrossEncoder = FakeST, FakeCE
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return FakeST


def test_local_embedder_adds_bge_prefix_to_queries_only(monkeypatch):
    fake = _fake_sentence_transformers(monkeypatch)
    emb = LocalEmbedder("BAAI/bge-small-en-v1.5")
    emb.embed_documents(["doc"])
    assert fake.last == ["doc"]
    emb.embed_query("q")
    assert fake.last == [BGE_QUERY_PREFIX + "q"]


def test_reranker_reorders(monkeypatch):
    _fake_sentence_transformers(monkeypatch)
    hits = [Hit(Chunk(str(i), text, "DBS", 2025, i, "", "text", "f"), 0.0, "hybrid")
            for i, text in enumerate(["income", "profit profit", "profit"])]
    ranked = Reranker("x").rerank("profit?", hits)
    assert [h.chunk.id for h in ranked] == ["1", "2", "0"]
    assert ranked[0].source == "hybrid+rerank"
