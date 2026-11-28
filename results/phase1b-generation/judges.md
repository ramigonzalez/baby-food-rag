Items scored by every run: 30.

| Run | Judge | Temperature | Seed | Faithfulness | Completeness | Relevance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| haiku-1 | `anthropic/claude-haiku-5.5` | not sent | not sent | 0.8096 | 0.5042 | 0.9167 |
| haiku-2 | `anthropic/claude-haiku-5.5` | not sent | not sent | 0.7938 | 0.4625 | 0.9333 |
| gemini-1 | `google/gemini-2.5-flash` | 0.0 | 7 | 0.8163 | 0.6083 | 0.9500 |
| gemini-2 | `google/gemini-2.5-flash` | 0.0 | 7 | 0.8080 | 0.6083 | 0.9500 |

| Pair | Kind | Faithfulness MAD | Faithfulness changed > 0.1 | Completeness MAD | Relevance MAD |
| --- | --- | ---: | ---: | ---: | ---: |
| haiku-1 vs haiku-2 | same judge | 0.0357 (n 30) | 20% | 0.0417 (n 8) | 0.0500 |
| haiku-1 vs gemini-1 | cross judge | 0.0854 (n 30) | 40% | 0.1042 (n 8) | 0.0667 |
| haiku-1 vs gemini-2 | cross judge | 0.0829 (n 30) | 40% | 0.1042 (n 8) | 0.0667 |
| haiku-2 vs gemini-1 | cross judge | 0.0832 (n 30) | 37% | 0.1458 (n 8) | 0.0833 |
| haiku-2 vs gemini-2 | cross judge | 0.0768 (n 30) | 37% | 0.1458 (n 8) | 0.0833 |
| gemini-1 vs gemini-2 | same judge | 0.0301 (n 30) | 17% | 0.0000 (n 8) | 0.0000 |

Cross-judge mean over 4 pairs: faithfulness MAD 0.0821, completeness MAD 0.1250.

MAD is the mean absolute difference per item. No human labels exist yet, so these numbers show stability and agreement, not which judge is right.
