# Phase 1b generation evaluation

Not medical advice. Numbers below come from this run only.

Generator: `openai/gpt-4o-mini`. Judge: `anthropic/claude-haiku-5.5`. Split: `all`. Questions: 115.
Mean answer latency: 2.27 s.

## Tokens and cost

| Role | Model | Calls | Input tokens | Output tokens | of which reasoning | USD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| generator | `openai/gpt-4o-mini` | 115 | 255456 | 13535 | 0 | $0.0464 |
| judge | `anthropic/claude-haiku-5.5` | 260 | 371507 | 131624 | 91782 | $0.1030 |
| total | | 375 | 626963 | 145159 | 91782 | $0.1494 |

OpenRouter reported $0.1494 for these calls.

Estimate before the run: $0.2342.

USD is the measured tokens times the OpenRouter rate in `generate.py`.

## Scores

| Group | n | Faithfulness | Hallucination | Citation | Relevance | Completeness | Abstention acc. | False abstentions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 115 | 0.8187 | 0.1813 | 1.0000 | 0.9295 | 0.3847 (n 25) | 0.8667 | 22 |
| dev | 69 | 0.8251 | 0.1749 | 1.0000 | 0.8958 | 0.4363 (n 17) | 1.0000 | 12 |
| test | 46 | 0.8084 | 0.1916 | 1.0000 | 0.9833 | 0.2750 (n 8) | 0.6667 | 10 |
| real | 10 | 0.7889 | 0.2111 | 1.0000 | 0.8333 | 0.1500 (n 8) | 0.5000 | 5 |
| plan | 15 | 0.8610 | 0.1390 | 1.0000 | 0.9167 | 0.1667 (n 2) | 0.6667 | 0 |
| synthetic | 90 | 0.8120 | 0.1880 | 1.0000 | 0.9365 | 0.5389 (n 15) | 1.0000 | 17 |

Faithfulness is the share of extracted claims the retrieved context supports, and hallucination rate is 1 minus faithfulness. Citation correctness is the share of [source p.N] marks that match a retrieved page. Abstention accuracy is the share of the unanswerable questions that abstained. A partial answer is not an abstention: those scores still apply, and a claim that the guides do not cover a point is supported when the retrieved text never states that point. Relevance is 0.5 when only part of the question is covered. Completeness is the share of a question's key facts the answer states, on the questions that have key facts; an abstention covers none. Faithfulness, citation correctness, relevance, and completeness are means over answerable questions.
