# Phase 1a: embedder comparison

Not medical advice. Retrieval only. No answer was generated and no chat model
was called.

## Pre-registered decision rule

Written and committed before any comparison number was computed:

> te3-large becomes the default embedder only if, on dev, the 95% paired CI of the hybrid recall@10 difference excludes 0 in its favour, and its median query latency is under 1 second. Otherwise e5 stays.

The rule is applied to the dev split only. The test split is run once afterwards
for both embedders, as the held-out confirmation, and does not change the decision.

## Setup

Variable: the embedding model. `e5` is `intfloat/multilingual-e5-large` (local, ONNX through fastembed, with `query: ` and `passage: ` prefixes), index `data/index`. `te3-large` is `openai/text-embedding-3-large` through the OpenRouter embeddings endpoint (3072 dimensions, no prefixes), index `data/index-te3-large`.

Fixed: golden set v2 (`eval/set.yaml`), 180/40 word chunks (the run stops unless both indexes hold the same chunks), the suffix BM25 (k1 1.5, b 0.75), and hybrid reciprocal rank fusion (k 60) of the top 50 BM25 and dense chunks.

A hit is the same `source_id` and a PDF page within 1 of a gold page; one retrieved page credits one gold page. Recall@k and MRR are as in phase 2. nDCG@10 uses binary gains on pages under the same one-credit rule. Ctx recall is the share of gold pages covered by the pages of the top 8 chunks from `index.search`, the call `baby-food ask` makes; only the hybrid row is what the chat sends today.

Paired bootstrap: te3-large minus e5 on per-question recall@10, 10,000 resamples of the questions, seed 0, 2.5 and 97.5 percentiles.

Latency: one `embed_queries` call per answerable question, after one untimed warmup. e5 runs on this CPU with the model already loaded. te3-large bypasses the cache, so each call is a real HTTPS request to OpenRouter from this machine.

Hardware: Apple M3, 8 CPUs, Darwin 24.6.0. e5 runs on the CPU (ONNX); no GPU is used.

## Dev (n = 60)

| Embedder | Retriever | Recall@5 | Recall@10 | MRR | nDCG@10 | Ctx recall | n |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| e5 | dense | 0.4917 | 0.5972 | 0.4199 | 0.4274 | 0.5556 | 60 |
| te3-large | dense | 0.6722 | 0.7361 | 0.6398 | 0.6098 | 0.7028 | 60 |
| e5 | hybrid | 0.4889 | 0.6278 | 0.4314 | 0.4385 | 0.5556 | 60 |
| te3-large | hybrid | 0.4750 | 0.6000 | 0.4891 | 0.4596 | 0.5222 | 60 |

### By origin

| Embedder | Retriever | Origin | Recall@5 | Recall@10 | MRR | nDCG@10 | Ctx recall | n |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| e5 | dense | real | 0.0000 | 0.0000 | 0.0372 | 0.0000 | 0.0000 | 4 |
| te3-large | dense | real | 0.1667 | 0.1667 | 0.3852 | 0.1913 | 0.1667 | 4 |
| e5 | hybrid | real | 0.3333 | 0.3333 | 0.1500 | 0.1554 | 0.3333 | 4 |
| te3-large | hybrid | real | 0.0833 | 0.1667 | 0.2993 | 0.1543 | 0.1667 | 4 |
| e5 | dense | plan | 0.5208 | 0.6875 | 0.6406 | 0.5710 | 0.6875 | 8 |
| te3-large | dense | plan | 0.7500 | 0.8542 | 0.7125 | 0.6614 | 0.8542 | 8 |
| e5 | hybrid | plan | 0.6042 | 0.8958 | 0.6679 | 0.6260 | 0.7917 | 8 |
| te3-large | hybrid | plan | 0.7917 | 0.8542 | 0.7604 | 0.6899 | 0.8542 | 8 |
| e5 | dense | synthetic | 0.5278 | 0.6319 | 0.4150 | 0.4391 | 0.5799 | 48 |
| te3-large | dense | synthetic | 0.7014 | 0.7639 | 0.6489 | 0.6361 | 0.7222 | 48 |
| e5 | hybrid | synthetic | 0.4826 | 0.6076 | 0.4155 | 0.4309 | 0.5347 | 48 |
| te3-large | hybrid | synthetic | 0.4549 | 0.5938 | 0.4597 | 0.4466 | 0.4965 | 48 |

### te3-large minus e5, recall@10 (paired bootstrap)

| Retriever | e5 | te3-large | Difference | 95% CI | Excludes 0 |
| --- | ---: | ---: | ---: | --- | --- |
| dense | 0.5972 | 0.7361 | +0.1389 | [+0.0528, +0.2250] | yes |
| hybrid | 0.6278 | 0.6000 | -0.0278 | [-0.1139, +0.0583] | no |

### Query-embedding latency

| Embedder | Median ms | Mean ms | p90 ms | Max ms | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| e5 | 48.8 | 48.3 | 60.0 | 67.1 | 60 |
| te3-large | 524.3 | 541.3 | 603.5 | 776.2 | 60 |

## Decision (rule applied to dev)

e5 stays the default embedder: the dev hybrid recall@10 interval [-0.1139, 0.0583] does not exclude 0 in te3-large's favour.

## Test, held-out confirmation (n = 40)

| Embedder | Retriever | Recall@5 | Recall@10 | MRR | nDCG@10 | Ctx recall | n |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| e5 | dense | 0.4083 | 0.5208 | 0.3289 | 0.3366 | 0.4958 | 40 |
| te3-large | dense | 0.7000 | 0.8042 | 0.5823 | 0.6076 | 0.7375 | 40 |
| e5 | hybrid | 0.3750 | 0.4917 | 0.3287 | 0.3347 | 0.4375 | 40 |
| te3-large | hybrid | 0.5250 | 0.6625 | 0.4445 | 0.4656 | 0.6000 | 40 |

### By origin

| Embedder | Retriever | Origin | Recall@5 | Recall@10 | MRR | nDCG@10 | Ctx recall | n |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| e5 | dense | real | 0.2500 | 0.2500 | 0.2869 | 0.2500 | 0.2500 | 4 |
| te3-large | dense | real | 0.5000 | 0.7500 | 0.4292 | 0.5040 | 0.6250 | 4 |
| e5 | hybrid | real | 0.2500 | 0.3750 | 0.1791 | 0.2123 | 0.3750 | 4 |
| te3-large | hybrid | real | 0.5000 | 0.6250 | 0.4021 | 0.4473 | 0.6250 | 4 |
| e5 | dense | plan | 0.3750 | 0.5000 | 0.7708 | 0.5130 | 0.5000 | 4 |
| te3-large | dense | plan | 0.5000 | 0.7917 | 0.5903 | 0.5602 | 0.5000 | 4 |
| e5 | hybrid | plan | 0.5000 | 0.7917 | 0.5750 | 0.5567 | 0.5000 | 4 |
| te3-large | hybrid | plan | 0.3750 | 0.7917 | 0.5857 | 0.5356 | 0.6667 | 4 |
| e5 | dense | synthetic | 0.4323 | 0.5573 | 0.2789 | 0.3253 | 0.5260 | 32 |
| te3-large | dense | synthetic | 0.7500 | 0.8125 | 0.6005 | 0.6265 | 0.7812 | 32 |
| e5 | hybrid | synthetic | 0.3750 | 0.4688 | 0.3166 | 0.3223 | 0.4375 | 32 |
| te3-large | hybrid | synthetic | 0.5469 | 0.6510 | 0.4321 | 0.4592 | 0.5885 | 32 |

### te3-large minus e5, recall@10 (paired bootstrap)

| Retriever | e5 | te3-large | Difference | 95% CI | Excludes 0 |
| --- | ---: | ---: | ---: | --- | --- |
| dense | 0.5208 | 0.8042 | +0.2833 | [+0.1667, +0.4125] | yes |
| hybrid | 0.4917 | 0.6625 | +0.1708 | [+0.0583, +0.2958] | yes |

### Query-embedding latency

| Embedder | Median ms | Mean ms | p90 ms | Max ms | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| e5 | 44.6 | 46.0 | 55.5 | 68.7 | 40 |
| te3-large | 536.3 | 538.8 | 601.1 | 658.4 | 40 |

The test split was run once, after the dev decision. It is reported as a
check and does not change the decision.

- dense: dev [+0.0528, +0.2250] (te3-large higher), test [+0.1667, +0.4125] (te3-large higher). The two splits agree.
- hybrid: dev [-0.1139, +0.0583] (includes 0), test [+0.0583, +0.2958] (te3-large higher). The two splits do not agree.

## Cost (te3-large; e5 is local)

| Item | Calls | Tokens | USD at the listed price | USD reported by OpenRouter |
| --- | ---: | ---: | ---: | ---: |
| Index build (from the index meta.json) | 25 | 421,637 | $0.0548 | $0.0548 |
| dev run: queries and timing calls | 62 | 2,650 | $0.0003 | $0.0003 |
| test run: queries and timing calls | 42 | 1,664 | $0.0002 | $0.0002 |
| Total | | 425,951 | $0.0554 | $0.0554 |

Listed price: $0.13 per million input tokens. A rerun with the same chunks
and questions reads every vector from `data/cache/embeddings.sqlite`; only the
uncached timing calls are billed again.

## Limitations

- Gold pages and the synthetic questions were written by an agent; the owner
  reviewed a sample of 20 and no clinician has checked them.
- Real parent questions are 4 per split, too few to read on their own.
- te3-large latency is network time from this machine to OpenRouter and the
  provider behind it. It varies with the connection and the hour.
- Cost uses the API's token counts times the listed price, not an invoice.
