"""Runs every experiment in a config file against the question set and writes a comparison report.

  python -m eval.run_eval                                   retrieval + answers for all experiments
  python -m eval.run_eval --retrieval-only                  fast: no LLM calls, retrieval metrics only
  python -m eval.run_eval --judge anthropic                 also grade answers with an LLM judge
  python -m eval.run_eval --questions eval/demo_questions.jsonl --config eval/demo_experiments.yaml
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import yaml

from citebank.config import ROOT, get_settings
from citebank.embeddings import get_embedder
from citebank.indexer import load_or_build
from citebank.llm import get_llm
from citebank.pipeline import RAGPipeline
from citebank.retrieval import Reranker

from .metrics import judge, load_items, number_match, retrieval_scores

RESULTS_DIR = ROOT / "eval" / "results"


def _mean(values):
    # averages only the values that apply (some metrics don't apply to every question type)
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def _settings_for(base, exp: dict):
    # each experiment is just a set of overrides on top of the .env settings
    overrides = {k: v for k, v in exp.items() if k not in {"name", "auto_filter"}}
    # folders come in from the yaml as text, so I resolve them against the project root like config.py does
    for key in ("raw_dir", "index_dir"):
        if key in overrides:
            path = Path(overrides[key])
            overrides[key] = path if path.is_absolute() else ROOT / path
    return base.with_overrides(**overrides)


def run_experiment(exp: dict, items, base, caches, retrieval_only: bool, judge_llm, page_tolerance: int):
    settings = _settings_for(base, exp)

    # loading models is slow, so I cache embedders, rerankers and indexes across experiments
    embed_key = (settings.embed_backend, settings.local_embed_model, settings.openai_embed_model)
    if embed_key not in caches["embed"]:
        caches["embed"][embed_key] = get_embedder(settings)
    embedder = caches["embed"][embed_key]

    index_key = str(settings.index_path())
    if index_key not in caches["index"]:
        caches["index"][index_key] = load_or_build(settings, embedder)
    store = caches["index"][index_key]

    reranker = None
    if settings.use_reranker:
        if settings.rerank_model not in caches["rerank"]:
            caches["rerank"][settings.rerank_model] = Reranker(settings.rerank_model)
        reranker = caches["rerank"][settings.rerank_model]

    llm = None if retrieval_only else get_llm(settings)
    rag = RAGPipeline(settings, embedder=embedder, llm=llm or get_llm(settings, "echo"), reranker=reranker,
                      auto_filter=exp.get("auto_filter", True), store=store)

    rows = []
    for item in items:
        row = {"experiment": exp["name"], "id": item.id, "type": item.type, "question": item.question}
        if retrieval_only:
            t0 = time.perf_counter()
            hits = rag.retrieve(item.question)
            row["retrieval_s"] = time.perf_counter() - t0
            chunks = [h.chunk for h in hits]
        else:
            ans = rag.ask(item.question)
            row.update({"answer": ans.answer, "refused": ans.refused, "retrieval_s": ans.retrieval_s,
                        "generation_s": ans.generation_s, "invalid_citations": ans.invalid_citations,
                        "cited_pages": [(c.bank, c.year, c.page) for c in ans.citations]})
            # the answer already lists what was retrieved, so I score that instead of searching again
            chunks = _chunks_from_sources(ans.sources)

        row["retrieved"] = [(c.bank, c.year, c.page) for c in chunks]
        if item.answerable and item.evidence:
            row.update(retrieval_scores(chunks, item.evidence, page_tolerance))

        if not retrieval_only:
            if item.answerable:
                # refusing a question the reports do answer is a failure too
                row["false_refusal"] = 1.0 if row["refused"] else 0.0
                row["number_match"] = number_match(row["answer"], item.answer or "") if not row["refused"] else 0.0
                row["citation_valid"] = 0.0 if row["invalid_citations"] else 1.0
                cited = [{"bank": b, "year": y, "page": p} for b, y, p in row["cited_pages"]]
                row["citation_hit"] = 1.0 if any(
                    retrieval_scores([_FakeChunk(**c)], item.evidence, page_tolerance)["hit"] for c in cited) else 0.0
                if judge_llm and not row["refused"]:
                    row.update(judge(judge_llm, item.question, item.answer or "", row["answer"], ans.sources))
            else:
                # for unanswerable questions, the right behaviour is to refuse
                row["correct_refusal"] = 1.0 if row["refused"] else 0.0
        rows.append(row)
    return rows


class _FakeChunk:
    # lets me reuse the relevance check on cited pages, which only need bank, year and page
    def __init__(self, bank, year, page):
        self.bank, self.year, self.page = bank, year, page


def _chunks_from_sources(sources):
    # maps the answer's source list back to chunk-like objects for scoring
    return [_FakeChunk(s["bank"], s["year"], s["page"]) for s in sources]


def summarise(rows: list[dict]) -> dict:
    keys = ["hit", "mrr", "recall", "false_refusal", "correct_refusal", "number_match", "citation_valid",
            "citation_hit", "faithful", "correct", "retrieval_s", "generation_s"]
    out = {}
    for k in keys:
        vals = [float(r[k]) if isinstance(r.get(k), bool) else r.get(k) for r in rows]
        out[k] = _mean(vals)
    return out


def _fmt(value, pct=True):
    if value is None:
        return "-"
    return f"{value * 100:.0f}%" if pct else f"{value:.2f}"


def write_report(experiments, summaries, all_rows, args, n_items) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = RESULTS_DIR / f"rows_{stamp}.jsonl"
    with open(rows_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    lines = [f"# CiteBank evaluation report ({stamp})", "",
             f"Questions: {n_items} | page tolerance: {args.page_tolerance} | mode: {'retrieval only' if args.retrieval_only else 'full'}"
             + (f" | judge: {args.judge}" if args.judge else ""), ""]
    lines += ["## Retrieval", "", "| Experiment | Chunking | Search | Rerank | Auto filter | Bank split | Hit@k | MRR | Recall@k | Avg retrieval (ms) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    base = get_settings()
    for exp, s in zip(experiments, summaries):
        st = _settings_for(base, exp)
        lines.append(f"| {exp['name']} | {st.chunk_strategy} | {st.retrieval_mode} | {'yes' if st.use_reranker else 'no'} | "
                     f"{'yes' if exp.get('auto_filter', True) else 'no'} | {'yes' if st.multi_bank_split else 'no'} | {_fmt(s['hit'])} | {_fmt(s['mrr'], False)} | "
                     f"{_fmt(s['recall'])} | {s['retrieval_s'] * 1000:.0f} |")
    if not args.retrieval_only:
        lines += ["", "## Answers", "", "| Experiment | LLM | Correct refusals | False refusals | Number match | Citation valid | Cites right page | Faithful | Correct | Avg generation (s) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for exp, s in zip(experiments, summaries):
            st = _settings_for(base, exp)
            model = {"ollama": st.ollama_model, "openai": st.openai_model, "anthropic": st.anthropic_model}.get(st.llm_backend, st.llm_backend)
            lines.append(f"| {exp['name']} | {st.llm_backend}: {model} | {_fmt(s['correct_refusal'])} | {_fmt(s['false_refusal'])} | "
                         f"{_fmt(s['number_match'])} | {_fmt(s['citation_valid'])} | {_fmt(s['citation_hit'])} | "
                         f"{_fmt(s['faithful'])} | {_fmt(s['correct'])} | {_fmt(s['generation_s'], False)} |")

    # error analysis: which questions the best experiment still gets wrong
    best_i = max(range(len(summaries)), key=lambda i: (summaries[i]["mrr"] or 0))
    best = experiments[best_i]["name"]
    misses = [r for r in all_rows if r["experiment"] == best and r.get("hit") == 0.0]
    lines += ["", f"## Retrieval misses for the best experiment ({best})", ""]
    if not misses:
        lines.append("None. Every answerable question had a relevant chunk in the top k.")
    for r in misses:
        lines.append(f"- **{r['id']}**: {r['question']}  \n  retrieved: {r['retrieved']}")
    lines += ["", f"Per-question results: `{rows_path.relative_to(ROOT)}`"]

    report = RESULTS_DIR / f"report_{stamp}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    parser.add_argument("--config", default=str(ROOT / "eval" / "experiments.yaml"))
    parser.add_argument("--only", nargs="*", help="run only these experiment names")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--judge", choices=["ollama", "openai", "anthropic"], help="grade answers with this LLM")
    parser.add_argument("--include-unverified", action="store_true")
    parser.add_argument("--page-tolerance", type=int, default=0, help="count chunks within this many pages as relevant")
    args = parser.parse_args(argv)

    items = load_items(args.questions, args.include_unverified)
    if not items:
        raise SystemExit("No verified questions found. Check eval/questions.jsonl and mark checked ones as verified.")
    experiments = yaml.safe_load(Path(args.config).read_text())["experiments"]
    if args.only:
        experiments = [e for e in experiments if e["name"] in args.only]
    # retrieval-only runs skip experiments that only differ by LLM, since their retrieval is identical
    if args.retrieval_only:
        seen, unique = set(), []
        for e in experiments:
            key = tuple(sorted((k, str(v)) for k, v in e.items() if k not in {"name", "llm_backend"}))
            if key not in seen:
                seen.add(key)
                unique.append(e)
        experiments = unique

    base = get_settings()
    judge_llm = get_llm(base, args.judge) if args.judge and not args.retrieval_only else None
    caches = {"embed": {}, "index": {}, "rerank": {}}
    summaries, all_rows = [], []
    for exp in experiments:
        print(f"running {exp['name']} on {len(items)} questions...")
        rows = run_experiment(exp, items, base, caches, args.retrieval_only, judge_llm, args.page_tolerance)
        all_rows += rows
        summaries.append(summarise(rows))
        s = summaries[-1]
        print(f"  hit@k {_fmt(s['hit'])}  MRR {_fmt(s['mrr'], False)}  recall {_fmt(s['recall'])}")

    report = write_report(experiments, summaries, all_rows, args, len(items))
    print(f"\nreport written to {report}")


if __name__ == "__main__":
    main()
