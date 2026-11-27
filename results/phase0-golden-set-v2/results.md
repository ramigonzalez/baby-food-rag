# Phase 2 retrieval baseline

Not medical advice. This file is retrieval only. No answer was generated.

Split: `all`.
Answerable questions: 100 of 115.
Unanswerable questions are excluded from the means: 15.

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
| bm25 | 0.2583 | 0.4017 | 0.2701 | 100 |
| dense | 0.4583 | 0.5667 | 0.3835 | 100 |
| hybrid | 0.4433 | 0.5733 | 0.3903 | 100 |

### By split

| Retriever | Split | Origin | Recall@5 | Recall@10 | MRR | n |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| bm25 | dev | all | 0.2861 | 0.3917 | 0.3113 | 60 |
| bm25 | test | all | 0.2167 | 0.4167 | 0.2083 | 40 |
| dense | dev | all | 0.4917 | 0.5972 | 0.4199 | 60 |
| dense | test | all | 0.4083 | 0.5208 | 0.3289 | 40 |
| hybrid | dev | all | 0.4889 | 0.6278 | 0.4314 | 60 |
| hybrid | test | all | 0.3750 | 0.4917 | 0.3287 | 40 |

### By origin

| Retriever | Split | Origin | Recall@5 | Recall@10 | MRR | n |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| bm25 | all | real | 0.0000 | 0.0417 | 0.0477 | 8 |
| bm25 | all | plan | 0.4167 | 0.4583 | 0.6435 | 12 |
| bm25 | all | synthetic | 0.2604 | 0.4292 | 0.2363 | 80 |
| dense | all | real | 0.1250 | 0.1250 | 0.1621 | 8 |
| dense | all | plan | 0.4722 | 0.6250 | 0.6840 | 12 |
| dense | all | synthetic | 0.4896 | 0.6021 | 0.3606 | 80 |
| hybrid | all | real | 0.2917 | 0.3542 | 0.1645 | 8 |
| hybrid | all | plan | 0.5694 | 0.8611 | 0.6369 | 12 |
| hybrid | all | synthetic | 0.4396 | 0.5521 | 0.3759 | 80 |

### By split and origin

| Retriever | Split | Origin | Recall@5 | Recall@10 | MRR | n |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| bm25 | dev | real | 0.0000 | 0.0833 | 0.0580 | 4 |
| bm25 | dev | plan | 0.4792 | 0.5417 | 0.7054 | 8 |
| bm25 | dev | synthetic | 0.2778 | 0.3924 | 0.2667 | 48 |
| bm25 | test | real | 0.0000 | 0.0000 | 0.0373 | 4 |
| bm25 | test | plan | 0.2917 | 0.2917 | 0.5196 | 4 |
| bm25 | test | synthetic | 0.2344 | 0.4844 | 0.1907 | 32 |
| dense | dev | real | 0.0000 | 0.0000 | 0.0372 | 4 |
| dense | dev | plan | 0.5208 | 0.6875 | 0.6406 | 8 |
| dense | dev | synthetic | 0.5278 | 0.6319 | 0.4150 | 48 |
| dense | test | real | 0.2500 | 0.2500 | 0.2869 | 4 |
| dense | test | plan | 0.3750 | 0.5000 | 0.7708 | 4 |
| dense | test | synthetic | 0.4323 | 0.5573 | 0.2789 | 32 |
| hybrid | dev | real | 0.3333 | 0.3333 | 0.1500 | 4 |
| hybrid | dev | plan | 0.6042 | 0.8958 | 0.6679 | 8 |
| hybrid | dev | synthetic | 0.4826 | 0.6076 | 0.4155 | 48 |
| hybrid | test | real | 0.2500 | 0.3750 | 0.1791 | 4 |
| hybrid | test | plan | 0.5000 | 0.7917 | 0.5750 | 4 |
| hybrid | test | synthetic | 0.3750 | 0.4688 | 0.3166 | 32 |

## Exact page (tolerance 0)

| Retriever | Recall@5 | Recall@10 | MRR | n |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 0.1750 | 0.3083 | 0.1855 | 100 |
| dense | 0.4033 | 0.5017 | 0.3303 | 100 |
| hybrid | 0.3750 | 0.4817 | 0.3127 | 100 |

### By split

| Retriever | Split | Origin | Recall@5 | Recall@10 | MRR | n |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| bm25 | dev | all | 0.1806 | 0.2861 | 0.2035 | 60 |
| bm25 | test | all | 0.1667 | 0.3417 | 0.1585 | 40 |
| dense | dev | all | 0.4000 | 0.4972 | 0.3478 | 60 |
| dense | test | all | 0.4083 | 0.5083 | 0.3041 | 40 |
| hybrid | dev | all | 0.4139 | 0.5111 | 0.3412 | 60 |
| hybrid | test | all | 0.3167 | 0.4375 | 0.2700 | 40 |

Gold pages were chosen by an agent reading the extracted text, not by a
health professional. `needs_review` questions are included in these means.
