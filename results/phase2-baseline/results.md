# Phase 2 retrieval baseline

Not medical advice. This file is retrieval only. No answer was generated.

Answerable questions: 32 of 37.
Unanswerable questions are excluded from the means: 5.

A hit is a retrieved page with the same `source_id` whose PDF page is
within 1 of a gold PDF page (exact page or ±1).
Chunks from the same source and PDF page count once, at the first time
that page appears. One retrieved page credits one gold page, the closest
one still unmatched, so neighboring gold pages are not both scored from
a single hit. Recall@k is the mean, over answerable questions, of
the share of that question's gold pages hit in the top k pages.
MRR is the mean of 1/rank of the first hit, or 0 when there is no hit.

Hybrid fuses the top 50 BM25 chunks with the top 50 dense chunks by reciprocal rank fusion (k 60). Embedder: `intfloat/multilingual-e5-large`.

## Tolerance 1

| Retriever | Recall@5 | Recall@10 | MRR | n |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 0.2188 | 0.3385 | 0.3161 | 32 |
| dense | 0.3958 | 0.4688 | 0.3976 | 32 |
| hybrid | 0.5052 | 0.7083 | 0.4077 | 32 |

## Exact page (tolerance 0)

| Retriever | Recall@5 | Recall@10 | MRR | n |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 0.1927 | 0.3125 | 0.2748 | 32 |
| dense | 0.3490 | 0.4219 | 0.3168 | 32 |
| hybrid | 0.4792 | 0.6406 | 0.3586 | 32 |

Gold pages were chosen by an agent reading the extracted text, not by a
health professional. `needs_review` questions are included in these means.
