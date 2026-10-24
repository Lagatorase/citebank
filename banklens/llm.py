"""LLM backends behind one interface, so I can swap a local model for a hosted one with a single setting."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import httpx

from .config import Settings


@dataclass
class LLMResult:
    text: str
    model: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLM:
    name = "base"

    def generate(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 700) -> LLMResult:
        raise NotImplementedError


class OllamaLLM(LLM):
    """Talks to a local Ollama server, which runs the model on my own GPU."""

    def __init__(self, url: str, model: str):
        self.name = f"ollama:{model}"
        self.model = model
        # local models can be slow on the first call while they load into GPU memory, so the timeout is generous
        self.client = httpx.Client(base_url=url, timeout=300)

    def generate(self, system, user, temperature=0.0, max_tokens=700):
        start = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192},
        }
        try:
            resp = self.client.post("/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise RuntimeError("Can't reach Ollama. Start it, then run: ollama pull " + self.model) from exc
        resp.raise_for_status()
        data = resp.json()
        return LLMResult(data["message"]["content"], self.name, time.perf_counter() - start,
                         data.get("prompt_eval_count"), data.get("eval_count"))


class OpenAILLM(LLM):
    """Any OpenAI-compatible chat completions endpoint."""

    def __init__(self, api_key: str, model: str, base_url: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is empty, set it in .env")
        self.name = f"openai:{model}"
        self.model = model
        self.client = httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=120)

    def generate(self, system, user, temperature=0.0, max_tokens=700):
        start = time.perf_counter()
        resp = self.client.post("/chat/completions", json={
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return LLMResult(data["choices"][0]["message"]["content"], self.name, time.perf_counter() - start,
                         usage.get("prompt_tokens"), usage.get("completion_tokens"))


class AnthropicLLM(LLM):
    """Anthropic's Messages API."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is empty, set it in .env")
        self.name = f"anthropic:{model}"
        self.model = model
        self.client = httpx.Client(base_url="https://api.anthropic.com", timeout=120,
                                   headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"})

    def generate(self, system, user, temperature=0.0, max_tokens=700):
        start = time.perf_counter()
        resp = self.client.post("/v1/messages", json={
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        resp.raise_for_status()
        data = resp.json()
        text = "".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text")
        usage = data.get("usage", {})
        return LLMResult(text, self.name, time.perf_counter() - start, usage.get("input_tokens"), usage.get("output_tokens"))


class EchoLLM(LLM):
    """Offline stand-in for tests: quotes the first source it was given, or refuses if there are none."""

    name = "echo"

    def generate(self, system, user, temperature=0.0, max_tokens=700):
        match = re.search(r"\[1\][^\n]*\n(.+?)(?:\n\[2\]|\n\nQuestion:)", user, re.S)
        if not match:
            return LLMResult("I could not find this in the annual reports provided.", self.name, 0.0)
        snippet = " ".join(match.group(1).split()[:40])
        return LLMResult(f"{snippet} [1]", self.name, 0.0)


def get_llm(settings: Settings, backend: str | None = None) -> LLM:
    backend = backend or settings.llm_backend
    if backend == "ollama":
        return OllamaLLM(settings.ollama_url, settings.ollama_model)
    if backend == "openai":
        return OpenAILLM(settings.openai_api_key, settings.openai_model, settings.openai_base_url)
    if backend == "anthropic":
        return AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model)
    if backend == "echo":
        return EchoLLM()
    raise ValueError(f"unknown LLM_BACKEND '{backend}', use ollama, openai or anthropic")
