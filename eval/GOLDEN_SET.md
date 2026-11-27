**Not medical advice.** This file describes a retrieval test set. The labels were written by an agent, not by a health professional.

# Golden set v2

`eval/set.yaml` grew from 37 to 115 questions: 100 answerable and 15 unanswerable. The 37 original questions keep their ids, wording, and gold pages. Every question now has two more fields: `split` (`dev` or `test`) and `human_verified` (`false` for all of them).

Why: 32 answerable questions were too few to split, and the phase 3 sweep picked its winner on the same 32 it reported. The synthetic questions also scored much higher than the real ones (hybrid recall@10 0.71 overall, 0.35 on the 8 answerable real questions), because a question written while reading the page reuses its words.

## Counts

Answerable, by cohort and split. The cohort is the id prefix: `real-`, `plan-`, `syn-` (synthetic, phase 2), `syn2-` (synthetic, this version).

| Cohort | Origin | dev | test | total |
| --- | --- | ---: | ---: | ---: |
| real | real | 4 | 4 | 8 |
| plan | plan | 8 | 4 | 12 |
| synthetic-old (`syn-`) | synthetic | 8 | 4 | 12 |
| synthetic-new (`syn2-`) | synthetic | 40 | 28 | 68 |
| total | | 60 | 40 | 100 |

Unanswerable: 2 real (1 dev, 1 test), 3 plan (2 dev, 1 test), 10 new `unans2-` (6 dev, 4 test). Total 9 dev, 6 test.

Answerable, by the source of the first gold page:

| Source | dev | test | total | of which `syn2-` |
| --- | ---: | ---: | ---: | ---: |
| MS-2019 | 32 | 22 | 54 | 25 |
| SBP-MAN-2024 | 12 | 9 | 21 | 21 |
| ASBAI-SBP-2025 | 10 | 6 | 16 | 14 |
| SBP-BLW-2017 | 6 | 3 | 9 | 8 |

Gold status: 88 `verified`, 12 `needs_review` (6 old, 6 new). `python scripts/golden_set_stats.py` prints these tables.

## How the questions were made

- **real** (10): questions a parent asked (`eval/perguntas_reais.yaml`).
- **plan** (15): questions from the case plan (`eval/questions.yaml`).
- **synthetic-old** (12): written by an agent in phase 2.
- **synthetic-new** (68 answerable, 10 unanswerable): written by an agent for this version. The agent read the text that `extract_pdf` returns for each page (the same text the index uses), picked a passage a parent could ask about, wrote the question in colloquial Brazilian Portuguese, and copied a verbatim quote from that page. Topics: allergens, texture and choking, age of introduction, sugar, honey and salt, milk and breastfeeding, meal routine, BLW, ultra-processed foods, hygiene, and quantities. Some questions have a second or third gold page when another core guide covers the same point.

The new unanswerable questions are things a parent would plausibly type into the chat: a medicine dose (simethicone drops, nasal saline), a product choice (diaper cream, teething gel, thermometer), a deworming schedule, and topics the guides do not cover (pools, flights, walkers, ear piercing). Each was checked by searching the extracted text of the four core PDFs for its key terms. Where a term did appear (`assadura` in ASBAI, `gengiva` and `verminose` in MS 2019, `febre`), the page was read and it does not answer the question. That is a keyword search and a read of the hits, not a read of all 575 pages. Supplement doses were left out on purpose: the SBP manual does give iron and vitamin D doses, so such a question would not be unanswerable, and the project does not dose medicines.

## Paraphrase rule and overlap

Rule: a new question must not copy the page's wording. A key noun such as `mel`, `ovo`, or `amendoim` is allowed, as in the real questions.

Measure: `golden.quote_overlap`, the share of a question's content tokens that also appear in its gold quotes. Content tokens come from the phase 2 BM25 tokenizer (accents kept, suffix stem) after dropping the stopword list in `analyze.py` and tokens shorter than 3 characters.

| Cohort | n | mean | median | p75 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| real | 8 | 0.108 | 0.139 | 0.167 | 0.250 |
| plan | 12 | 0.187 | 0.211 | 0.250 | 0.333 |
| synthetic-old | 12 | 0.172 | 0.167 | 0.250 | 0.375 |
| synthetic-new | 68 | 0.067 | 0.087 | 0.125 | 0.222 |

Any draft above 0.25, the highest real value, was rewritten before it went in. For example, `Bisnaguinha e pão de forma no lanche, pode?` (0.40, both nouns are in the quote) became `Aquele pãozinho de pacote que criança adora, pode no lanche?` (0.00). Ten drafts were above 0.25 and were rewritten this way (the highest was 0.60); two more were dropped when the list was trimmed. A few drafts at or below 0.25 were reworded as well.

Two caveats. The number depends on quote length, and the real quotes are short phrases. And the new questions were written to avoid the page's words, so they may be harder to match lexically than what parents actually type. Overlap measures vocabulary only, not whether the question is natural.

## Gold labels

`verified` means the quote was found, as an exact substring, in the extracted text of that PDF page. `tests/test_gold_quotes.py` checks every verified quote against its PDF page after whitespace normalization; it skips when the PDFs are not downloaded. The older index test (`test_gold_quotes_are_on_the_cited_page`) checks all quotes, verified or not, against the local index; it passes on this set.

`needs_review` means the quote is on the page but the label itself is uncertain:

| id | reason |
| --- | --- |
| real-01 | The guide lists strawberry and tomato as foods; it does not explain the grandmother's ban. |
| real-03 | No order of fruits is given; the nearest text says every fruit may be offered. |
| real-06 | The page says foods shape the palate; it does not discuss a fruit-only diet. |
| real-08 | The page says keep foods separate; it does not name potato with banana. |
| real-10 | Broad question; the pages are the clearest rules on salt, frying, and seasoning. |
| syn-07 | MS 2019 and the SBP manual differ on cow's milk before 1 year. |
| syn2-024 | Vegetarian baby: MS 2019 accepts it with follow-up and B12; the SBP manual says children should not be vegetarian. |
| syn2-025 | Bottle sterilizing: chlorine soak of at least 1 hour (MS 2019) against 15 minutes (SBP manual). |
| syn2-029 | Honey after 1 year: SBP manual says no honey in year 1 and no sugar of any kind, honey included, before 2; MS 2019 says not before 2. |
| syn2-031 | First-day amount: 1 to 2 teaspoons and 2 to 3 tablespoons (both SBP manual), about 1 dessert spoon per group (MS 2019). |
| syn2-048 | Breastfeeding and allergy: ASBAI/SBP 2025 says protection is unproven; MS 2019 says it protects. |
| syn2-067 | Formula-fed baby at 4 months: the SBP BLW guide says 6 months for all; the SBP manual allows 4 or 6. |

These rows are the first candidates for expert review. Gold pages are also not exhaustive: another page may answer a question too, and a retriever that finds it gets no credit.

## Human review

On 2026-11-27 the owner reviewed 20 items drawn with seed 20261127 from the verified rows: 14 `syn2-`, 3 older answerable, and 3 `unans2-`. The reviewer is not a clinician. For each item they checked that the question reads like a parent's, that the quote is on the linked PDF page, and that it answers the question.

- All 20 items are `human_verified: true`. No gold page was wrong.
- `syn2-042`: the quote now includes the table header, so the unit (litres/day) is visible.
- `syn2-003`, `syn2-017`, `syn2-039`, `syn2-058`: the gold page is right, but the quote is only part of the answer. The rest is on the same page, so recall is unaffected. The review note in each row lists what a complete answer needs.
- `plan-12`: on a second look both gold pages stand, and the quotes now hold the full sentences. The first quote keeps a stray "3" that the extractor inserted mid-sentence (likely a figure label).

The quote is a pointer to the evidence. It is not a reference answer. Scoring whether a generated answer is complete needs its own label: the key facts the answer must contain.

## Split

`golden.assign_splits` with seed `20261127`. Strata are `(answerable, origin, cohort, first gold source)`. Inside a stratum, questions are ordered by `sha256("20261127:<id>")`. The strata are walked in sorted order, and a question goes to dev whenever that keeps the running dev count at ceil(3/5 of the questions seen), with one counter for answerable and one for unanswerable questions. The result is 60/40 overall and close to 60/40 inside each cohort and source.

The split was computed once and written into `eval/set.yaml`; the stored field is what counts. Running `assign_splits` again after adding questions can move existing ones, so give a new question its own split and leave the old ones alone.

Use dev to choose and test to report once:

```bash
baby-food sweep --split dev --out results/<run>
baby-food eval --split test --out results/<run>
```

`--split all`, the default, scores every question. The new code run on the 37-question set from commit `0c8a40a` reproduces `results/phase2-baseline/` exactly.

## Baseline on v2

`baby-food eval --out results/phase0-golden-set-v2` (page tolerance 1, same index and settings as phase 2):

| Retriever | Split | n | Recall@5 | Recall@10 | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| bm25 | dev | 60 | 0.2861 | 0.3917 | 0.3113 |
| bm25 | test | 40 | 0.2167 | 0.4167 | 0.2083 |
| dense | dev | 60 | 0.4917 | 0.5972 | 0.4199 |
| dense | test | 40 | 0.4083 | 0.5208 | 0.3289 |
| hybrid | dev | 60 | 0.4889 | 0.6278 | 0.4314 |
| hybrid | test | 40 | 0.3750 | 0.4917 | 0.3287 |

Recall@10 by cohort, all splits:

| Retriever | real (8) | plan (12) | synthetic-old (12) | synthetic-new (68) |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 0.0417 | 0.4583 | 0.4167 | 0.4314 |
| dense | 0.1250 | 0.6250 | 0.5417 | 0.6127 |
| hybrid | 0.3542 | 0.8611 | 0.7917 | 0.5098 |

The cohort table comes from `python scripts/golden_set_stats.py eval/set.yaml results/phase0-golden-set-v2/per_question.csv`, which also splits each cohort by dev and test.

What these numbers show, without a significance test:

- The new synthetic questions score lower than the old synthetic and plan questions with hybrid (0.51 against 0.79 and 0.86) and are still above the 8 real ones (0.35).
- Over all 100 questions, dense and hybrid are close (recall@10 0.567 and 0.573). On the new synthetic questions, hybrid is below dense (0.51 against 0.61). The phase 2 hybrid lead came mostly from questions that share words with the page.
- Hybrid recall@10 is 0.63 on dev and 0.49 on test. With 40 test questions, a gap this size can be noise.
- Real questions are 4 dev and 4 test. That is too few to read on their own.

## Limitations

- An agent wrote the synthetic questions and chose every gold page and quote. The owner reviewed a sample of 20 (see Human review). No clinician has checked them. Treat `needs_review` rows as candidates for expert review.
- 80 of the 100 answerable questions are synthetic. The real-question row is the most honest number and the smallest.
- Quotes were checked against extracted text, not against the rendered page.
- Gold pages are not exhaustive, so recall can understate a retriever that finds a different page that also answers.
