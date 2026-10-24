"""Command line entry point.

  python -m citebank.cli build              parse the PDFs in data/raw and build the index
  python -m citebank.cli ask "question"     answer one question with citations
  python -m citebank.cli search "question"  show the retrieved chunks only, without calling the LLM
"""

from __future__ import annotations

import argparse
import textwrap

from .config import get_settings
from .indexer import build_index


def _settings_from_args(args):
    # command line flags override .env, which is handy for quick comparisons
    changes = {}
    for flag, key in [("mode", "retrieval_mode"), ("llm", "llm_backend"), ("embed", "embed_backend"),
                      ("strategy", "chunk_strategy"), ("k", "top_k")]:
        value = getattr(args, flag, None)
        if value is not None:
            changes[key] = value
    if getattr(args, "no_rerank", False):
        changes["use_reranker"] = False
    return get_settings().with_overrides(**changes)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="citebank")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--mode", choices=["dense", "bm25", "hybrid"])
        p.add_argument("--embed", choices=["local", "openai", "hash"])
        p.add_argument("--strategy", choices=["fixed", "structured"])
        p.add_argument("--k", type=int)
        p.add_argument("--no-rerank", action="store_true")

    p_build = sub.add_parser("build", help="parse PDFs and build the index")
    add_common(p_build)

    p_ask = sub.add_parser("ask", help="answer a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--llm", choices=["ollama", "openai", "anthropic", "echo"])
    add_common(p_ask)

    p_search = sub.add_parser("search", help="show retrieved chunks without calling the LLM")
    p_search.add_argument("question")
    add_common(p_search)

    args = parser.parse_args(argv)
    settings = _settings_from_args(args)

    if args.command == "build":
        print("Building index...")
        build_index(settings)
        return

    # I import the pipeline here so "build" doesn't need the reranker or LLM settings to be valid
    from .pipeline import RAGPipeline

    if args.command == "search":
        from .llm import EchoLLM

        rag = RAGPipeline(settings, llm=EchoLLM())
        for n, hit in enumerate(rag.retrieve(args.question), start=1):
            c = hit.chunk
            print(f"\n[{n}] {c.bank} {c.year} p.{c.page} ({c.kind}, {hit.source}, score {hit.score:.3f}) {c.section}")
            print(textwrap.indent(textwrap.shorten(c.text, 400), "    "))
        return

    rag = RAGPipeline(settings)
    answer = rag.ask(args.question)
    print("\n" + answer.answer + "\n")
    for cite in answer.citations:
        print(f"  [{cite.number}] {cite.bank} {cite.year}, page {cite.page}  {cite.section}")
    if answer.invalid_citations:
        print(f"  warning: model cited sources that don't exist: {answer.invalid_citations}")
    print(f"\n  {answer.model} | retrieval {answer.retrieval_s:.2f}s | generation {answer.generation_s:.2f}s")


if __name__ == "__main__":
    main()
