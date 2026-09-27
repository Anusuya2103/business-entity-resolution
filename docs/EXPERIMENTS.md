# Experiment log

Every decision below was measured against the **train** ground truth
(`train_ground_truth.tsv`), never guessed. Reproduce each table with the
matching `src/eval_*.py` script.

## 0. Ground-truth structure (the single most useful fact)

| property | value |
|---|---|
| rows | 2,206,821 |
| singletons (empty truth) | 123,247 (5.58%) |
| max cluster size | **11** |
| mean cluster size (non-empty) | 3.67 |

## 1. Why the first "fast" approach was thrown away

An earlier pass added ~1.75M pairs with string heuristics (name-gate,
cluster propagation, promotion by support). They looked convincing:

| heuristic | share that was name-exact | share with address agreement |
|---|---|---|
| promo / propagate adds | 92.7% | ~1% |
| model's own v1 approvals | 97.5% | 31% |

But calibrated against train GT:

| condition | P(pair is true) |
|---|---|
| name equal **and** address equal | 100% |
| name equal + address Jaccard ≥ 0.67 | 94% |
| address equal | 81% |
| **name equal + different address** | **2.4%** |

The heuristic adds lived almost entirely in that last row — name-twins in
different cities (a common India pattern). Re-scoring all 1.75M through the
model left only **5,575** survivors (0.3%). This was rolled back; it is not
part of the submission.

## 2. Trim policy — oversized clusters

Predicted cluster size vs. train GT distribution showed the tail was badly
over-merged (32,242 entities at ≥12, where train has **zero**).

Measured on 1,020 train entities whose blocking already produced ≥12
candidates (their true size averages only **3.5**, max 9):

| policy | macro F<sub>0.5</sub> | Δ vs no trim |
|---|---|---|
| **addr + cap3** | **0.6262** | **+0.1269** |
| addr + cap5 | 0.6167 | +0.1174 |
| threshold 0.999 | 0.5851 | +0.0858 |
| cap3 | 0.5610 | +0.0617 |
| cap11 | 0.5025 | +0.0032 |
| no trim | 0.4993 | — |
| addr only (no cap) | 0.4137 | −0.0856 |

Repeat on the 5–11 band (1,489 entities, true size avg 3.66):

| policy | macro F<sub>0.5</sub> | Δ |
|---|---|---|
| **addr + cap5** | **0.6740** | **+0.1040** |
| addr + cap3 | 0.6682 | +0.0982 |
| addr + cap11 | 0.6531 | +0.0830 |
| no trim | 0.5701 | — |

**Conclusion:** *address agreement is the dominant signal*; the cap is
secondary. Both bands were trimmed with address-first ordering.

## 3. Threshold sweep — is 0.99 worth it?

Macro F<sub>0.5</sub> by resulting cluster size, on a train sample:

| threshold | band-1 | band-2 | band-3+ |
|---|---|---|---|
| 0.980 | 0.5934 | 0.6694 | 0.6198 |
| 0.990 | 0.6216 | 0.7274 | 0.6822 |
| **0.995** | **0.6245** | **0.7371** | **0.6952** |

and the decisive number — precision of pairs in the near-miss band:

| band | precision |
|---|---|
| [0.990, 0.995) | **0.2125** |

0.2125 is far below the ~0.57 break-even, so a 0.99 pass was rejected.
Confirmed independently by a probe over 1.15M rejected candidates, of which
only 0.49% fell in [0.990, 0.995).

## 4. Are the top-up additions real?

Added pairs only, bucketed by how many matches the entity had beforehand:

| base band | entities | base precision | **add precision** | f05 before | f05 after | Δ |
|---|---|---|---|---|---|---|
| 0 (empty) | 242 | — | 0.658 | 0.0000 | 0.1459 | **+0.1459** |
| 1 | 505 | 0.889 | 0.714 | 0.6437 | 0.6683 | **+0.0246** |
| 2 | 545 | 0.866 | 0.697 | 0.7585 | 0.7653 | +0.0068 |
| 3–4 | 676 | 0.829 | 0.727 | 0.7874 | 0.7923 | +0.0049 |
| 5+ | 294 | **0.390** | 0.623 | 0.5723 | 0.5739 | +0.0017 |

Every band improves, so each round was merged only after this check passed.
Note the 0.390 base precision in the 5+ band — that is what motivated
extending the trim from ≥12 down to 5.

## 5. Final submission

```
matching_results.tsv : 1,732,544 rows (89,112 empty, 1,643,432 non-empty)
pairs                : 4,985,660
max cluster size     : 5
subset check         : 4,985,660 matched pairs, 0 missing from candidates
result               : PASS
```

## Bugs caught along the way

- **`pandas.update()` replaced match lists instead of unioning them.** An
  early merge silently dropped every pre-existing match for 3,138 entities.
  Replaced by `merge_adds.py`, which unions base-first and never loses a pair.
- **A checker bug of my own:** an early "row order mismatch" alarm compared
  `candidate_pairs.tsv` (country-grouped order) against `test_source1.tsv`
  order. The official validator matches by **id**, not position — fixed.
- **The official validator OOMs** on a 16 GB host (two full id→set maps over
  a 120M-pair candidate file). A streaming replica with identical rules is in
  `validate_stream.py`.
