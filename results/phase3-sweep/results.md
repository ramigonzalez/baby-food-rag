# Phase 3 retrieval sweep

Not medical advice. Retrieval only. No answer was generated.

The best configuration is the highest recall@10 on all 32 answerable questions. There is no held-out split. If a hybrid uses a BM25 analyzer other than the phase 2 suffix list, that analyzer was chosen on these same questions.

Hardware: Intel(R) Xeon(R) Processor, 4 CPUs, device cpu. No GPU.
Latency is the mean online time per answerable query after one warmup query.
It does not include model load or embedding the corpus.

Hit rule: same source_id and abs(retrieved pdf_page - gold pdf_page) <= 1; one retrieved page credits one gold page
Hybrid fuses the top 50 chunks from each side with reciprocal rank fusion k 60.
The reranker, when it ran, reorders the top 30 hybrid chunks.

## All configurations

| Config | Family | Recall@5 | Recall@10 | MRR | Latency ms |
| --- | --- | ---: | ---: | ---: | ---: |
| hybrid-baseline | hybrid | 0.5052 | 0.7083 | 0.4077 | 121.6 |
| rerank-m3-top30 | rerank | 0.6302 | 0.6875 | 0.5686 | 11874.1 |
| hybrid-snowball-stop-e5 | hybrid | 0.4688 | 0.5781 | 0.3732 | 117.7 |
| dense-e5-page | dense | 0.4323 | 0.5729 | 0.4191 | 120.4 |
| bm25-snowball-stop | bm25 | 0.3490 | 0.4948 | 0.3669 | 1.0 |
| dense-e5 | dense | 0.3958 | 0.4688 | 0.3976 | 120.0 |
| bm25-snowball | bm25 | 0.2760 | 0.4635 | 0.3466 | 2.5 |
| bm25-folded-stop | bm25 | 0.3281 | 0.4062 | 0.3145 | 0.7 |
| bm25-suffix-page | bm25 | 0.2812 | 0.3750 | 0.2781 | 0.9 |
| bm25-suffix | bm25 | 0.2188 | 0.3385 | 0.3161 | 1.9 |
| bm25-plain | bm25 | 0.2500 | 0.3385 | 0.2883 | 2.0 |
| bm25-folded | bm25 | 0.3125 | 0.3385 | 0.2925 | 2.0 |
| dense-minilm-page | dense | 0.1406 | 0.2292 | 0.1232 | 5.4 |
| dense-minilm | dense | 0.0781 | 0.1979 | 0.1792 | 13.2 |

## Answerable real questions only

| Config | n | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| hybrid-baseline | 8 | 0.2917 | 0.3542 | 0.1645 |
| bm25-snowball-stop | 8 | 0.1667 | 0.2917 | 0.1321 |
| dense-e5-page | 8 | 0.1667 | 0.2917 | 0.2018 |
| dense-minilm | 8 | 0.1667 | 0.2708 | 0.2062 |
| dense-minilm-page | 8 | 0.1667 | 0.2708 | 0.1310 |
| hybrid-snowball-stop-e5 | 8 | 0.1042 | 0.2292 | 0.1371 |
| bm25-snowball | 8 | 0.0000 | 0.2083 | 0.0752 |
| rerank-m3-top30 | 8 | 0.1875 | 0.1875 | 0.2233 |
| dense-e5 | 8 | 0.1250 | 0.1250 | 0.1621 |
| bm25-folded-stop | 8 | 0.0000 | 0.0625 | 0.0521 |
| bm25-suffix | 8 | 0.0000 | 0.0417 | 0.0477 |
| bm25-plain | 8 | 0.0000 | 0.0417 | 0.0514 |
| bm25-folded | 8 | 0.0000 | 0.0417 | 0.0539 |
| bm25-suffix-page | 8 | 0.0000 | 0.0000 | 0.0554 |

## Best against the hybrid baseline

Best by recall@10: `hybrid-baseline` (0.7083). Compared config `rerank-m3-top30` (0.6875) minus baseline `hybrid-baseline` (0.7083) is -0.0208, 95% bootstrap interval [-0.1302, 0.0885] (10000 paired resamples, seed 0). best is the hybrid baseline; the interval is the next config minus that baseline

No other configuration beat the hybrid baseline. The difference for rerank-m3-top30 is not significant: the 95% interval includes 0.

Hybrid baseline minus `bm25-suffix` recall@10 is 0.3698, 95% interval [0.2292, 0.5208].

## Not run

- `BAAI/bge-m3`: Dense BGE-M3 is not in fastembed 0.9.1, the latest release on PyPI. This VM has no GPU and torch is not installed, so the model was not loaded.

Whole-page dense embeddings are truncated at 512 tokens by the embedding model.
Labels were made by an agent, on this same question set. n is 32.
