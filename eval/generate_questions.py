"""Drafts extra eval questions with an LLM, one per sampled chunk, for me to check by hand.

  python -m eval.generate_questions --n 30 --llm anthropic

Generated questions come out with "verified": false. I check each answer against the PDF, fix the page if needed,
then copy the good ones into eval/questions.jsonl with "verified": true. LLM-written questions tend to reuse the
exact wording of the chunk, which makes retrieval look easier than it is, so I also reword some myself.
"""

from __future__ import annotations

import argparse
import json
import random
import re

from citebank.config import ROOT, get_settings
from citebank.embeddings import get_embedder
from citebank.indexer import load_or_build
from citebank.llm import get_llm

PROMPT = """Below is an excerpt from the {bank} annual report for {year}, page {page}.

{text}

Write one question that a bank analyst might ask, which this excerpt answers on its own. Prefer questions about specific
figures or clearly stated facts. Do not mention "the excerpt" or "the passage" in the question. Name the bank and year
in the question.

Reply with JSON only: {{"question": "...", "answer": "...", "type": "numeric" or "factual"}}"""


def sample_chunks(chunks, n: int, seed: int):
    # I favour chunks that contain figures, since those make the most checkable questions,
    # and spread the sample across banks so one long report doesn't dominate
    rng = random.Random(seed)
    useful = [c for c in chunks if len(c.text.split()) > 40]
    by_bank: dict[str, list] = {}
    for c in useful:
        by_bank.setdefault(c.bank, []).append(c)
    picked = []
    while len(picked) < n and any(by_bank.values()):
        for bank in list(by_bank):
            pool = by_bank[bank]
            if not pool:
                continue
            with_numbers = [c for c in pool if re.search(r"\d", c.text)] or pool
            choice = rng.choice(with_numbers)
            pool.remove(choice)
            picked.append(choice)
            if len(picked) >= n:
                break
    return picked


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--llm", choices=["ollama", "openai", "anthropic"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(ROOT / "eval" / "generated_questions.jsonl"))
    args = parser.parse_args(argv)

    settings = get_settings()
    store = load_or_build(settings, get_embedder(settings))
    llm = get_llm(settings, args.llm)

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for chunk in sample_chunks(store.chunks, args.n, args.seed):
            body = chunk.text.split("\n", 1)[-1]
            raw = llm.generate("You write evaluation questions. Reply with JSON only.",
                               PROMPT.format(bank=chunk.bank, year=chunk.year, page=chunk.page, text=body[:2500]),
                               temperature=0.3, max_tokens=250).text
            match = re.search(r"\{.*\}", raw, re.S)
            try:
                data = json.loads(match.group(0))
            except (AttributeError, json.JSONDecodeError):
                # a bad reply just gets skipped, I can always generate more
                continue
            item = {"id": f"gen-{chunk.id}", "question": data.get("question", ""), "type": data.get("type", "factual"),
                    "answer": data.get("answer", ""), "evidence": [{"bank": chunk.bank, "year": chunk.year, "page": chunk.page}],
                    "verified": False, "notes": "generated, check against the PDF"}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written} draft questions to {args.out}")


if __name__ == "__main__":
    main()
