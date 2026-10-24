"""Scoring functions for retrieval quality and answer quality."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from banklens.chunking import Chunk
from banklens.llm import LLM


@dataclass
class EvalItem:
    # one question in the eval set
    id: str
    question: str
    type: str
    answer: str | None = None
    # where the answer lives: a list of {"bank", "year", "page"}
    evidence: list[dict] = field(default_factory=list)
    verified: bool = False
    notes: str = ""

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"


def load_items(path, include_unverified: bool = False) -> list[EvalItem]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # blank lines and // comments are allowed so the file is easy to edit by hand
            if not line or line.startswith("//"):
                continue
            item = EvalItem(**json.loads(line))
            # unverified questions (like freshly generated ones) are skipped until I've checked them myself
            if item.verified or include_unverified:
                items.append(item)
    return items


# retrieval metrics


def is_relevant(chunk: Chunk, evidence: list[dict], page_tolerance: int = 0) -> bool:
    # a chunk counts as relevant if it comes from the right bank, year and page
    for e in evidence:
        if e.get("bank") and e["bank"] != chunk.bank:
            continue
        if e.get("year") and e["year"] != chunk.year:
            continue
        if abs(int(e["page"]) - chunk.page) <= page_tolerance:
            return True
    return False


def retrieval_scores(chunks: list[Chunk], evidence: list[dict], page_tolerance: int = 0) -> dict:
    """hit@k: found anything relevant. MRR: how high the first relevant chunk ranks. recall: share of evidence pages found."""
    flags = [is_relevant(c, evidence, page_tolerance) for c in chunks]
    first = next((rank for rank, ok in enumerate(flags, start=1) if ok), None)
    found = sum(any(is_relevant(c, [e], page_tolerance) for c in chunks) for e in evidence)
    return {
        "hit": 1.0 if first else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "recall": found / len(evidence) if evidence else 0.0,
    }


# answer metrics


def _numbers(text: str) -> set[str]:
    # pulls out figures like 10.3, 18,400 or 14.2% and normalises them (no commas, no trailing zeros)
    out = set()
    for raw in re.findall(r"\d[\d,]*(?:\.\d+)?", text or ""):
        n = raw.replace(",", "")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.add(n)
    return out


def number_match(answer: str, reference: str) -> float | None:
    """Share of the figures in the reference answer that also appear in the model's answer."""
    ref = _numbers(reference)
    # years alone aren't the "answer", so I ignore 4-digit numbers that look like years if other figures exist
    core = {n for n in ref if not re.fullmatch(r"20\d{2}", n)} or ref
    if not core:
        return None
    got = _numbers(answer)
    return sum(n in got for n in core) / len(core)


JUDGE_PROMPT = """You are grading an answer produced by a question-answering system over bank annual reports.

Question: {question}

Reference answer (written by a human who checked the report): {reference}

Sources the system was given:
{sources}

System answer: {answer}

Grade two things:
1. faithful: true if every factual claim in the system answer is supported by the sources above, false if any claim is not.
2. correct: "correct" if the system answer matches the reference answer, "partial" if it is incomplete or partly wrong, "incorrect" if it is wrong or missing.

Reply with JSON only, like: {{"faithful": true, "correct": "correct", "reason": "one short sentence"}}"""


def judge(llm: LLM, question: str, reference: str, answer: str, sources: list[dict]) -> dict:
    """Uses an LLM as a grader. It's not perfect, so I spot-check its reasons in the results file."""
    source_text = "\n\n".join(f"[{n}] {s['bank']} {s['year']} p.{s['page']}: {s['text'][:1200]}" for n, s in enumerate(sources, 1))
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, sources=source_text, answer=answer)
    raw = llm.generate("You are a strict, careful grader. Reply with JSON only.", prompt, temperature=0.0, max_tokens=200).text
    # models sometimes wrap json in extra text, so I grab the first {...} block
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    correct = str(data.get("correct", "")).lower()
    return {
        "faithful": bool(data.get("faithful")) if "faithful" in data else None,
        "correct": {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}.get(correct),
        "judge_reason": data.get("reason", raw[:200] if not data else ""),
    }
