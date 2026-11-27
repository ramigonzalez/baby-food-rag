# Pre-registration: phase 1c, te3-large dense against the current default

Written on 2026-11-27, after phase 1a (`results/phase1-embedders/`) and before any question in the evaluation set below exists.

## Why

Phase 1a's rule compared hybrid with e5 against hybrid with te3-large, and e5 stayed. Outside that rule, dense retrieval with te3-large looked better than the current default (hybrid with e5): +0.108 recall@10 on dev [−0.008, +0.228] and +0.313 on test. Both splits have now been seen, so neither can confirm that comparison. This file fixes the comparison and the rule before new data is collected.

## Comparison

- **Candidate:** dense retrieval, `openai/text-embedding-3-large`, index built from the same chunks (180 words, 40 overlap).
- **Baseline:** the current default, hybrid (suffix BM25 + `intfloat/multilingual-e5-large`, RRF k 60, pool 50).
- Everything else is unchanged.

## Evaluation set

A new split, `fresh`, of questions written after this date and never used to choose anything:
- the owner's new real parent questions (target about 20);
- about 20 new synthetic questions under the golden set v2 rules (paraphrased; overlap with the gold quote no higher than the real questions').

Each question gets gold pages and verbatim quotes in the usual way, with `split: fresh`. The `fresh` questions are scored once, for this comparison only.

## Metric and rule

- **Primary metric:** recall@10 at page tolerance 1, the same definition as phase 1a.
- **Rule:** dense te3-large becomes the default retriever only if the paired bootstrap 95% CI (10,000 resamples, seed 0) of recall@10 (candidate minus baseline) on `fresh` excludes 0 in the candidate's favour, and its median query latency is under 1 second. Otherwise the default stays hybrid e5.
- **Also reported, not used by the rule:** recall@5, MRR, nDCG@10, context-budget recall, and results by origin (real vs synthetic).

## Known limits

With about 40 questions the margin is wide (roughly ±0.16 worst case), so an inconclusive result is likely. An inconclusive result keeps the default.
