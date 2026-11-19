**Not medical advice.** This project searches official infant-feeding guides and returns the source and the page. It does not diagnose, prescribe, dose a medicine, or replace a pediatrician.

# baby-food-rag

Parents ask concrete questions about honey, grapes, allergens, and texture. The answers sit in official guides, and the guides do not all say the same thing. Phase 1 downloads those PDFs, extracts a page at a time, and searches the chunks. Every hit carries `source_id`, year, country, and page, and the citation is a link `<url>#page=N`. `N` is the PDF page a viewer opens. There is no answer generation yet.

## Sources

The default index is the Brazilian core. Uruguay is optional (`country: UY`) and stays out of that index. The contrast set is in the manifest with `role: contrast` and is not indexed unless you pass `--include contrast`. Full fields, licenses, and checksums are in [`sources/manifest.yaml`](sources/manifest.yaml).

| id | document | year | country |
| --- | --- | --- | --- |
| [MS-2019](https://www.gov.br/saude/pt-br/composicao/saps/promocao-da-saude/guias-alimentares/publicacoes/guia_da_crianca_2019.pdf/@@download/file) | Ministério da Saúde, Guia Alimentar para Crianças Brasileiras Menores de 2 Anos | 2019 | BR |
| [ASBAI-SBP-2025](https://www.sbp.com.br/fileadmin/user_upload/AAAI_V9_n1_a_03.pdf) | ASBAI and SBP, Atualização em Alergia Alimentar 2025 | 2025 | BR |
| [SBP-MAN-2024](https://www.sbp.com.br/fileadmin/user_upload/sbp/2024/outubro/31/24607e-ManAlim-OrientAlim_Lactente_ao_adl_na_escola-gest.pdf) | SBP, Manual de Alimentação, 5th ed. | 2024 | BR |
| [SBP-BLW-2017](https://www.sbp.com.br/fileadmin/user_upload/2017/05/Nutrologia-AlimCompl-Metodo-BLW.pdf) | SBP, A Alimentação Complementar e o Método BLW | 2017 | BR |
| [UY-MSP](https://www.gub.uy/ministerio-salud-publica/sites/ministerio-salud-publica/files/documentos/publicaciones/Gu%C3%ADa%20de%20alimentaci%C3%B3n%20complementaria%20para%20ni%C3%B1os%20de%20entre%206%20y%2024%20meses.pdf) | MSP Uruguay, Guía de alimentación complementaria (optional) | 2017 | UY |

Contrast, not in the default index: [ESPGHAN 2017](https://www.espghan.org/dam/jcr:3d960daa-e2f3-499f-9df9-2da682976cec/2017%20Complementary_Feeding__A_Position_Paper_by_the.21.pdf), [NIAID 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5226648/) (DOI [10.1016/j.jaci.2016.10.010](https://doi.org/10.1016/j.jaci.2016.10.010)), [ASCIA 2026 summary](https://www.allergy.org.au/images/pcc/ASCIA_HP_Guidelines_Infant_Feeding_for_Food_Allergy_Prevention_2026.pdf), [AAP 2019](https://doi.org/10.1542/peds.2019-0281). The AAP row is the DOI only. A third-party PDF is not downloaded.

The PDFs are not in git. `data/raw/` and `*.pdf` are gitignored. MS 2019 is CC BY-NC-SA 4.0. SBP, ASBAI, ESPGHAN, and AAP are all rights reserved. Do not commit the files.

## Download and index

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download.py
baby-food index
baby-food search "pode dar mel?" -k 3
baby-food stats
```

`scripts/download.py` is the catalog `download.sh` rewritten in Python: same official URLs, same user agent, sha256 checked when the manifest has one, and a failed URL does not stop the others. On 2026-11-19, `MS-RES-2021` and `MS-EVID-2025` (`bvsms.saude.gov.br`) returned HTTP 403. They are reference texts, not part of the default index. NIAID is not fetched by default. AAP is never fetched.

`baby-food index` writes `data/index/`, which is gitignored. Chunks are a recursive split of 180 words with 40 words of overlap, taken from [rag-eval-harness](https://github.com/ramigonzalez/rag-eval-harness). A chunk does not cross a page. In the 2019 MS guide the printed page is the PDF page plus 1, so the grape note is printed page 84 and the link uses `#page=83`.

Default index built here:

```
embedder: intfloat/multilingual-e5-large
chunk_words: 180  overlap: 40

source                 role       country  pdf_pages with_text   chunks
MS-2019                core       BR             265       253      527
ASBAI-SBP-2025         core       BR              92        92      447
SBP-MAN-2024           core       BR             212       209      561
SBP-BLW-2017           core       BR               6         6       26
total                                            575               1561
```

Twelve MS pages and three SBP Manual pages have no text layer (covers and full-bleed sheets), so `with_text` is lower than `pdf_pages`.

Dense search uses `intfloat/multilingual-e5-large` (1024 dimensions, ONNX through fastembed), with the `passage:` and `query:` prefixes that model expects. `bge-m3` is not in the fastembed build this project installs. BM25 uses k1 1.5 and b 0.75. Tokens keep Portuguese accents, and a short suffix list makes `cortar` match `cortadas`. `--retriever hybrid` fuses BM25 and dense with reciprocal rank fusion (k 60). `--embedder hash` builds a fake dense index for tests, without downloading the model.

## Example searches

Command output, not a rewrite.

```
baby-food search "pode dar mel?" -k 3 --retriever bm25
```

```
query: pode dar mel?
retriever: bm25
hits: 3

1. score=9.5718  MS-2019  year=2019  country=BR  printed_page=104  pdf_page=103
   https://www.gov.br/saude/pt-br/composicao/saps/promocao-da-saude/guias-alimentares/publicacoes/guia_da_crianca_2019.pdf/@@download/file#page=103
   ...ingrediente nos 2 primeiros anos de vida contribui para a formação de hábitos alimentares mais saudáveis. Pode oferecer mel? Apesar de o mel ser um produto natural, não é recomendado oferecer esse alimento à criança menor de 2 anos. São dois os motivos: o mel contém os mesmos componentes do açúcar, o que já justifica evitá-lo. Além disso, há risco de contami...

2. score=7.8261  MS-2019  year=2019  country=BR  printed_page=183  pdf_page=182
   https://www.gov.br/saude/pt-br/composicao/saps/promocao-da-saude/guias-alimentares/publicacoes/guia_da_crianca_2019.pdf/@@download/file#page=182
   Figura 56 – Cozinhando com frango em uma única panela. 1. Alimentos que serão cozidos. 2. Panela refogando alho e cebola - Em uma panela aquecida, 2 refogue com pouco óleo os temperos de sua preferência, como alho, cebola, cebolinha, talo de salsa etc. 3. Adicionando o frango ou a carne - Refogue a carne, pode ser em cubos pequenos ou médios, até que fiqu...

3. score=7.2412  MS-2019  year=2019  country=BR  printed_page=223  pdf_page=222
   https://www.gov.br/saude/pt-br/composicao/saps/promocao-da-saude/guias-alimentares/publicacoes/guia_da_crianca_2019.pdf/@@download/file#page=222
   ...r dificuldade de aceitação dos alimentos in natura ou minimamente processados. Não inclua na alimentação da criança nem mel, nem açúcar de qualquer tipo (mascavo, demerara, cristal ou refinado [“branco”], rapadura, melaço), nem ofereça preparações ou produtos prontos que contenham algum desses ingredientes (como biscoitos, bolos, iogurtes, entre outros). Os...
```

```
baby-food search "como cortar uva" -k 3 --retriever bm25
```

```
query: como cortar uva
retriever: bm25
hits: 3

1. score=10.1337  MS-2019  year=2019  country=BR  printed_page=84  pdf_page=83
   https://www.gov.br/saude/pt-br/composicao/saps/promocao-da-saude/guias-alimentares/publicacoes/guia_da_crianca_2019.pdf/@@download/file#page=83
   ...mentos. Elas podem ser raspadas ou amassadas com garfo e/ou ofertadas em pedaços para que a criança possa segurar com suas próprias mãos. Frutas pequenas ou com caroços, como uva e acerola, exigem maior atenção, pois podem provocar engasgo. Nesses casos, devem ser retiradas as sementes e cortadas em pedaços bem pequenos (dividir cada fruta em 2 a 4 pedaços).

2. score=8.8250  MS-2019  year=2019  country=BR  printed_page=82  pdf_page=81
   https://www.gov.br/saude/pt-br/composicao/saps/promocao-da-saude/guias-alimentares/publicacoes/guia_da_crianca_2019.pdf/@@download/file#page=81
   mastigar os mais duros, mesmo ralados. Conforme a criança vai crescendo, os alimentos podem ser oferecidos crus e ralados, como cenoura e beterraba, ou em tirinhas, para serem pegos com as mãos pelas crianças. Higienizar alimentos in natura é fácil. Veja na seção “Cozinhar em casa”. Figura 21 – Exemplos de alimentos do grupo de legumes e verduras. 2 1 1. ...

3. score=8.4619  MS-2019  year=2019  country=BR  printed_page=68  pdf_page=67
   https://www.gov.br/saude/pt-br/composicao/saps/promocao-da-saude/guias-alimentares/publicacoes/guia_da_crianca_2019.pdf/@@download/file#page=67
   Exemplos de alimentos ultraprocessados: refrigerantes; pós para refresco; bebidas adoçadas prontas para o consumo (chá, concentrado de guaraná, açaí, uva e outros sabores), bebidas prontas à
```

```
baby-food search "pode dar mel?" -k 3 --retriever dense
```

```
query: pode dar mel?
retriever: dense
hits: 3

1. score=0.8363  MS-2019  year=2019  country=BR  printed_page=104  pdf_page=103
   https://www.gov.br/saude/pt-br/composicao/saps/promocao-da-saude/guias-alimentares/publicacoes/guia_da_crianca_2019.pdf/@@download/file#page=103
   ...ingrediente nos 2 primeiros anos de vida contribui para a formação de hábitos alimentares mais saudáveis. Pode oferecer mel? Apesar de o mel ser um produto natural, não é recomendado oferecer esse alimento à criança menor de 2 anos. São dois os motivos: o mel contém os mesmos componentes do açúcar, o que já justifica evitá-lo. Além disso, há risco de contami...

2. score=0.8234  MS-2019  year=2019  country=BR  printed_page=138  pdf_page=137
   https://www.gov.br/saude/pt-br/composicao/saps/promocao-da-saude/guias-alimentares/publicacoes/guia_da_crianca_2019.pdf/@@download/file#page=137
   Se não tiver leite materno armazenado, oferecer outros alimentos, que podem ser fruta, almoço, jantar e/ou fórmulas infantis ou outros leites, dependendo do tempo que a mãe ficar longe da criança, do seu desenvolvimento e da aceitação por ela dos demais alimentos.

3. score=0.8229  MS-2019  year=2019  country=BR  printed_page=61  pdf_page=60
   https://www.gov.br/saude/pt-br/composicao/saps/promocao-da-saude/guias-alimentares/publicacoes/guia_da_crianca_2019.pdf/@@download/file#page=60
   Figura 16 – Leite materno oferecido na colher. Se você tiver dificuldade para retirar, conservar ou ofertar o leite materno à criança, procure apoio na Unidade Básica de Saúde, na maternidade, no Banco de Leite Humano mais próximo de você ou no local onde a criança está sendo ou vai ser acompanhada. Se você quiser doar leite materno para outros bebês, pro...
```

The e5 tokenizer receives the prefixes. A check on this index decoded `query: como cortar uva` as `▁que ry : ▁como ▁cortar ▁uva` and a passage as starting with `passage:`. The stored embeddings were already built that way, so the index was not rebuilt. Dense rank of MS 2019 printed page 84 for `como cortar uva` is 18 before that check and 18 after it (score 0.7942; the top hit scored 0.8088).

## Evaluation

`eval/set.yaml` is the labeled set. It has 37 questions: 10 real parent questions, 15 from the case plan, and 12 synthetic questions written in the same colloquial style. Five are `answerable: false` and are not scored: two of the real questions (whether strawberry is an allergen, and the daily protein intake) and the three plan questions about a drug dose, a commercial brand, and fever medicine. Six answerable questions are `needs_review` because the wording is broader than the page or the guides do not say the same thing. The other answerable questions are `verified`: the quote was found on that PDF page.

A hit is the same `source_id` and a PDF page within 1 of a gold page (the exact page or ±1). `--page-tolerance 0` scores the exact page only. Recall@k is the mean share of a question's gold pages found in the top k pages. MRR is the mean of 1/rank of the first hit. These numbers are from `baby-food eval` and are also in `results/phase2-baseline/`.

| Retriever | Recall@5 | Recall@10 | MRR | n |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 0.2292 | 0.3490 | 0.3161 | 32 |
| dense | 0.3958 | 0.4792 | 0.3976 | 32 |
| hybrid | 0.5156 | 0.7083 | 0.4077 | 32 |

Exact page, same run:

| Retriever | Recall@5 | Recall@10 | MRR | n |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 0.1927 | 0.3125 | 0.2748 | 32 |
| dense | 0.3490 | 0.4219 | 0.3168 | 32 |
| hybrid | 0.4792 | 0.6406 | 0.3586 | 32 |

The gold pages were chosen by an agent reading the extracted text, not by a health professional. The synthetic questions were written by that agent. `needs_review` rows are included in the means. This is not a measure of answer quality, because nothing generates an answer.

## Harness

[rag-eval-harness](https://github.com/ramigonzalez/rag-eval-harness) is a reference, not a dependency. Its chunk is `(chunk_id, doc_id, text, index)` over a whole document, and its tokenizer drops accents, so a page citation cannot be produced by calling it. Three pieces are adapted, with the MIT notice in the modules: the recursive word splitter, Okapi BM25, and reciprocal rank fusion. Judge v2 is `raglab.generation.judge_claims(..., version="v2")`. Nothing in this phase calls it.

## Roadmap

1. **Corpus and search.** Done. Download, page text, chunk metadata, BM25 and dense search, tests, CI. No generated answers.
2. **Evaluation questions and a retrieval baseline.** The labeled set and the BM25, dense, and hybrid numbers are in `eval/set.yaml` and `results/phase2-baseline/`. No generated answers and no judge.
3. **Compared configurations.** Not started. Other embeddings, a multilingual reranker, judge v2, abstention on the unanswerable questions, and mixing countries or sources.
4. **Chat UI.** Not started. A short answer, a clickable page, a fixed warning, and a weekly menu where each item cites the rule it follows.
5. **Public write-up.** Not started. A results table only after those measurements exist.

## Tests

GitHub Actions (`.github/workflows/ci.yml`) runs ruff and pytest on Python 3.11. The tests build their own short PDFs and use the hash embedder. They do not download guides or the embedding model. If the MS 2019 and ASBAI/SBP 2025 PDFs are already under `data/raw/`, extra tests check printed page 84 (grapes: seeds removed, cut into 2 to 4 pieces), printed page 104 (honey), and that the ASBAI sentence on introduction order is not mixed across the two columns.
