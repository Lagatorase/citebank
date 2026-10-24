# CiteBank evaluation report (20261024_143702)

Questions: 21 | page tolerance: 0 | mode: full

## Retrieval

| Experiment | Chunking | Search | Rerank | Auto filter | Hit@k | MRR | Recall@k | Avg retrieval (ms) |
|---|---|---|---|---|---|---|---|---|
| bm25_fixed | fixed | bm25 | no | yes | 87% | 0.69 | 68% | 9 |
| dense_fixed | fixed | dense | no | yes | 80% | 0.54 | 49% | 39 |
| dense_structured | structured | dense | no | yes | 87% | 0.60 | 56% | 22 |
| hybrid_structured | structured | hybrid | no | yes | 87% | 0.62 | 62% | 29 |
| hybrid_rerank | structured | hybrid | yes | yes | 93% | 0.72 | 68% | 501 |
| hybrid_rerank_nofilter | structured | hybrid | yes | no | 93% | 0.65 | 66% | 487 |

## Answers

| Experiment | LLM | Correct refusals | False refusals | Number match | Citation valid | Cites right page | Faithful | Correct | Avg generation (s) |
|---|---|---|---|---|---|---|---|---|---|
| bm25_fixed | ollama: qwen2.5:3b | 100% | 13% | 58% | 100% | 73% | - | - | 0.86 |
| dense_fixed | ollama: qwen2.5:3b | 100% | 20% | 50% | 100% | 47% | - | - | 1.15 |
| dense_structured | ollama: qwen2.5:3b | 100% | 27% | 46% | 100% | 47% | - | - | 0.89 |
| hybrid_structured | ollama: qwen2.5:3b | 83% | 13% | 55% | 100% | 60% | - | - | 0.87 |
| hybrid_rerank | ollama: qwen2.5:3b | 83% | 13% | 68% | 100% | 73% | - | - | 0.90 |
| hybrid_rerank_nofilter | ollama: qwen2.5:3b | 83% | 7% | 67% | 100% | 67% | - | - | 0.72 |

## Retrieval misses for the best experiment (hybrid_rerank)

- **big3-2025-profit**: Which of DBS, OCBC and UOB reported the highest net profit for 2025?  
  retrieved: [('OCBC', 2025, 99), ('OCBC', 2025, 9), ('OCBC', 2025, 24), ('OCBC', 2025, 9), ('UOB', 2025, 93)]

Per-question results are saved next to each report as `rows_<timestamp>.jsonl` (not committed).
