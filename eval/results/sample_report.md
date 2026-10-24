# CiteBank evaluation report (20261024_173938)

Questions: 21 | page tolerance: 0 | mode: full

## Retrieval

| Experiment | Chunking | Search | Rerank | Auto filter | Bank split | Hit@k | MRR | Recall@k | Avg retrieval (ms) |
|---|---|---|---|---|---|---|---|---|---|
| bm25_fixed | fixed | bm25 | no | yes | no | 93% | 0.70 | 69% | 8 |
| dense_fixed | fixed | dense | no | yes | no | 80% | 0.54 | 49% | 39 |
| dense_structured | structured | dense | no | yes | no | 87% | 0.60 | 55% | 16 |
| hybrid_structured | structured | hybrid | no | yes | no | 87% | 0.62 | 62% | 26 |
| hybrid_rerank | structured | hybrid | yes | yes | no | 93% | 0.72 | 68% | 499 |
| hybrid_rerank_nofilter | structured | hybrid | yes | no | no | 93% | 0.65 | 66% | 481 |
| hybrid_rerank_split | structured | hybrid | yes | yes | yes | 100% | 0.78 | 69% | 541 |

## Answers

| Experiment | LLM | Correct refusals | False refusals | Number match | Citation valid | Cites right page | Faithful | Correct | Avg generation (s) |
|---|---|---|---|---|---|---|---|---|---|
| bm25_fixed | ollama: qwen2.5:3b | 100% | 13% | 58% | 100% | 73% | - | - | 1.08 |
| dense_fixed | ollama: qwen2.5:3b | 100% | 20% | 50% | 100% | 47% | - | - | 1.13 |
| dense_structured | ollama: qwen2.5:3b | 100% | 27% | 46% | 100% | 47% | - | - | 0.88 |
| hybrid_structured | ollama: qwen2.5:3b | 83% | 13% | 55% | 100% | 60% | - | - | 0.87 |
| hybrid_rerank | ollama: qwen2.5:3b | 83% | 13% | 68% | 100% | 73% | - | - | 0.86 |
| hybrid_rerank_nofilter | ollama: qwen2.5:3b | 83% | 7% | 67% | 100% | 67% | - | - | 0.74 |
| hybrid_rerank_split | ollama: qwen2.5:3b | 83% | 7% | 68% | 100% | 80% | - | - | 0.69 |

## Retrieval misses for the best experiment (hybrid_rerank_split)

None. Every answerable question had a relevant chunk in the top k.

Per-question results: `eval\results\rows_20261024_173938.jsonl`
