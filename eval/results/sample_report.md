# CiteBank evaluation report (20261024_190512)

Questions: 32 | page tolerance: 0 | mode: full

## Retrieval

| Experiment | Chunking | Search | Rerank | Auto filter | Bank split | Hit@k | MRR | Recall@k | Avg retrieval (ms) |
|---|---|---|---|---|---|---|---|---|---|
| bm25_fixed | fixed | bm25 | no | yes | no | 96% | 0.80 | 79% | 10 |
| dense_fixed | fixed | dense | no | yes | no | 77% | 0.60 | 57% | 32 |
| dense_structured | structured | dense | no | yes | no | 92% | 0.67 | 66% | 19 |
| hybrid_structured | structured | hybrid | no | yes | no | 88% | 0.69 | 69% | 40 |
| hybrid_rerank | structured | hybrid | yes | yes | no | 96% | 0.82 | 79% | 495 |
| hybrid_rerank_nofilter | structured | hybrid | yes | no | no | 96% | 0.78 | 78% | 478 |
| hybrid_rerank_split | structured | hybrid | yes | yes | yes | 100% | 0.86 | 80% | 517 |

## Answers

| Experiment | LLM | Correct refusals | False refusals | Number match | Citation valid | Cites right page | Faithful | Correct | Avg generation (s) |
|---|---|---|---|---|---|---|---|---|---|
| bm25_fixed | ollama: qwen2.5:3b | 100% | 12% | 70% | 100% | 73% | - | - | 1.04 |
| dense_fixed | ollama: qwen2.5:3b | 100% | 19% | 62% | 100% | 62% | - | - | 1.08 |
| dense_structured | ollama: qwen2.5:3b | 83% | 23% | 58% | 100% | 58% | - | - | 0.85 |
| hybrid_structured | ollama: qwen2.5:3b | 83% | 8% | 73% | 100% | 73% | - | - | 0.82 |
| hybrid_rerank | ollama: qwen2.5:3b | 83% | 8% | 81% | 100% | 73% | - | - | 0.82 |
| hybrid_rerank_nofilter | ollama: qwen2.5:3b | 83% | 4% | 73% | 100% | 65% | - | - | 0.82 |
| hybrid_rerank_split | ollama: qwen2.5:3b | 83% | 8% | 78% | 100% | 77% | - | - | 0.69 |

## Retrieval misses for the best experiment (hybrid_rerank_split)

None. Every answerable question had a relevant chunk in the top k.

## What changed since the previous run

Two changes went in together, so their effects can't be separated:

- **Empty answer retry.** An answer that is empty or only citation markers gets one more LLM call at temperature 0.3, and becomes the refusal phrase if it is still empty. It fired 5 times across the seven experiments. It rescued 2 answers (uob-2025-appreciation in dense_structured, dbs-geopolitics in hybrid_rerank_nofilter) and turned 3 into refusals.
- **Profit label rule.** The prompt now says to use the version of a figure the question asks for when a source gives several (for example profit before tax and net profit).

Retrieval is unchanged. Number match rose in most experiments (bm25_fixed 60% to 70%, hybrid_structured 63% to 73%, hybrid_rerank 78% to 81%), while cites right page fell in several (hybrid_rerank 81% to 73%, hybrid_rerank_split 85% to 77%). Most of these moves are one to three questions, which is within noise for 26 answerable questions.

## Did the two target answers get fixed?

- **dbs-2025-retention** ("In DBS's 2025 annual report, how large is the retention award in deferred remuneration relative to the main award?"): no. In hybrid_rerank and hybrid_rerank_split the model again replied with only `[1]`, the retry gave the same, and the pipeline now refuses instead of showing a bare marker. The previous run counted the bare `[1]` as citing the right page, so part of the drop in cites right page for those two experiments is the score becoming more honest.
- **big3-2025-profit** ("Which of DBS, OCBC and UOB reported the highest net profit for 2025?"): no. hybrid_rerank_split still answers "DBS reported the highest net profit for 2025 at SGD 13.1 billion [1]". The bank is right and DBS p9 is an evidence page, but SGD 13.1 billion is profit before tax; the net profit on the same page is SGD 11.0 billion. Without the split, hybrid_rerank now answers instead of refusing, but credits DBS with UOB's $4.7 billion.

## Hand-written vs drafted questions (answerable only)

21 questions were written and checked by hand (15 answerable). 11 were drafted with `eval/generate_questions.py` and checked against the PDF text.

| Experiment | Hand-written (15): MRR | Number match | Cites right page | Drafted (11): MRR | Number match | Cites right page |
|---|---|---|---|---|---|---|
| bm25_fixed | 0.70 | 62% | 73% | 0.94 | 82% | 73% |
| dense_fixed | 0.54 | 54% | 53% | 0.68 | 73% | 73% |
| dense_structured | 0.60 | 46% | 47% | 0.76 | 73% | 73% |
| hybrid_structured | 0.62 | 67% | 67% | 0.79 | 82% | 82% |
| hybrid_rerank | 0.72 | 73% | 67% | 0.95 | 91% | 82% |
| hybrid_rerank_nofilter | 0.65 | 59% | 53% | 0.95 | 91% | 82% |
| hybrid_rerank_split | 0.78 | 68% | 73% | 0.95 | 91% | 82% |

The drafted questions are easier: each comes from a single passage and tends to reuse its wording, so retrieval finds the page more often (MRR 0.95 against 0.72 with reranking). They barely separate the reranked experiments, so the hand-written questions are the better guide to which choices matter.

## What the bank split changed

`hybrid_rerank_split` is `hybrid_rerank` with `MULTI_BANK_SPLIT` on, and the only question that names more than one bank is big3-2025-profit (evidence: DBS p9, DBS p12, OCBC p13, UOB p36).

- Without the split: sources were OCBC p99, OCBC p9, OCBC p24, OCBC p9 and UOB p93.
- With the split: sources were two pages from each of DBS, OCBC and UOB (six instead of five), with DBS p9 first. No OCBC or UOB evidence page was retrieved, so its recall is 1 of 4 pages.
- This question accounts for the retrieval differences from hybrid_rerank: Hit@k 96% to 100%, MRR 0.82 to 0.86.

## Remaining failures (hybrid_rerank_split)

Every answerable question had an evidence page in the top 5, so these all come from the model:

- **Refused although the page was retrieved:** ocbc-2025-nim and dbs-2025-retention (2 of 26 answerable questions).
- **Wrong figure:** big3-2025-profit quotes profit before tax as net profit.
- **Answered an unanswerable question:** unans-card ("Which UOB credit card should I sign up for?") gets a card recommendation instead of a refusal.
- **Vague summaries:** dbs-2025-ai, uob-sustainability and dbs-geopolitics give general answers that miss half or more of the reference figures and don't cite an evidence page.
- **Citation on the wrong page:** dbs-2024-workflows gives the right figure (80%) but cites a source that isn't the evidence page.

## Not run

The hosted model experiment (hybrid_rerank_api, same retrieval as hybrid_rerank_split) and the LLM judge were not run, because they need an Anthropic API key.

Per-question results are saved next to each report as `rows_<timestamp>.jsonl` (not committed).
