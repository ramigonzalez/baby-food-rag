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

## Numbers

Not run yet. This file holds the rule above, committed before the comparison.
