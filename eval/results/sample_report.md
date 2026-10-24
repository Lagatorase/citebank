# CiteBank evaluation report (20261024_184454)

Questions: 32 | page tolerance: 0 | mode: full

## Retrieval

| Experiment | Chunking | Search | Rerank | Auto filter | Bank split | Hit@k | MRR | Recall@k | Avg retrieval (ms) |
|---|---|---|---|---|---|---|---|---|---|
| bm25_fixed | fixed | bm25 | no | yes | no | 96% | 0.80 | 79% | 11 |
| dense_fixed | fixed | dense | no | yes | no | 77% | 0.60 | 57% | 37 |
| dense_structured | structured | dense | no | yes | no | 92% | 0.67 | 66% | 16 |
| hybrid_structured | structured | hybrid | no | yes | no | 88% | 0.69 | 69% | 29 |
| hybrid_rerank | structured | hybrid | yes | yes | no | 96% | 0.82 | 79% | 491 |
| hybrid_rerank_nofilter | structured | hybrid | yes | no | no | 96% | 0.78 | 78% | 478 |
| hybrid_rerank_split | structured | hybrid | yes | yes | yes | 100% | 0.86 | 80% | 517 |

## Answers

| Experiment | LLM | Correct refusals | False refusals | Number match | Citation valid | Cites right page | Faithful | Correct | Avg generation (s) |
|---|---|---|---|---|---|---|---|---|---|
| bm25_fixed | ollama: qwen2.5:3b | 100% | 15% | 60% | 100% | 69% | - | - | 0.98 |
| dense_fixed | ollama: qwen2.5:3b | 100% | 19% | 60% | 100% | 58% | - | - | 1.09 |
| dense_structured | ollama: qwen2.5:3b | 100% | 19% | 58% | 100% | 62% | - | - | 0.86 |
| hybrid_structured | ollama: qwen2.5:3b | 83% | 12% | 63% | 100% | 69% | - | - | 0.80 |
| hybrid_rerank | ollama: qwen2.5:3b | 83% | 8% | 78% | 100% | 81% | - | - | 0.82 |
| hybrid_rerank_nofilter | ollama: qwen2.5:3b | 83% | 8% | 73% | 100% | 73% | - | - | 0.71 |
| hybrid_rerank_split | ollama: qwen2.5:3b | 83% | 4% | 78% | 100% | 85% | - | - | 0.66 |

## Retrieval misses for the best experiment (hybrid_rerank_split)

None. Every answerable question had a relevant chunk in the top k.

## Hand-written vs drafted questions (answerable only)

21 questions were written and checked by hand (15 answerable). 11 were drafted with `eval/generate_questions.py` and checked against the PDF text.

| Experiment | Hand-written (15): MRR | Number match | Cites right page | Drafted (11): MRR | Number match | Cites right page |
|---|---|---|---|---|---|---|
| bm25_fixed | 0.70 | 58% | 73% | 0.94 | 64% | 64% |
| dense_fixed | 0.54 | 50% | 47% | 0.68 | 73% | 73% |
| dense_structured | 0.60 | 46% | 47% | 0.76 | 73% | 82% |
| hybrid_structured | 0.62 | 55% | 60% | 0.79 | 73% | 82% |
| hybrid_rerank | 0.72 | 68% | 73% | 0.95 | 91% | 91% |
| hybrid_rerank_nofilter | 0.65 | 67% | 67% | 0.95 | 82% | 82% |
| hybrid_rerank_split | 0.78 | 68% | 80% | 0.95 | 91% | 91% |

The scores on the hand-written questions are identical to the previous run, so the overall rise comes from the drafted questions being easier.

## What the bank split changed

`hybrid_rerank_split` is `hybrid_rerank` with `MULTI_BANK_SPLIT` on, and the only question that names more than one bank is big3-2025-profit (evidence: DBS p9, DBS p12, OCBC p13, UOB p36).

- Without the split: sources were OCBC p99, OCBC p9, OCBC p24, OCBC p9 and UOB p93, and the model refused.
- With the split: sources were two pages from each of DBS, OCBC and UOB (six instead of five), with DBS p9 first, and the model answered "DBS reported the highest net profit for 2025 at SGD 13.1 billion [1]", citing DBS p9. DBS is the right bank and p9 is an evidence page, but SGD 13.1 billion is DBS's profit before tax; its net profit on the same page is SGD 11.0 billion.
- This one question accounts for all of the differences from hybrid_rerank: Hit@k 96% to 100%, MRR 0.82 to 0.86, cites right page 81% to 85%, false refusals 8% to 4%. No OCBC or UOB evidence page was retrieved, so its recall is 1 of 4 pages.

The hosted API experiment (hybrid_rerank_api) and the LLM judge were not run because no Anthropic API key was set.

Per-question results are saved next to each report as `rows_<timestamp>.jsonl` (not committed).
