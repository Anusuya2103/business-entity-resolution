# Business Entity Resolution — ML Challenge 2026

Linking **Source 1** business records to their true duplicates across
**Source 2** and **Source 3**, scored by **macro-averaged F<sub>0.5</sub>**.

Final submission: [`submission/matching_results.tsv`](submission/matching_results.tsv)
(1,643,432 non-empty rows, 4,985,660 pairs, validator **PASS**).

---

## The metric drives every decision

F<sub>0.5</sub> is computed **per Source 1 entity** and then averaged over all
1,732,544 entities, so two facts dominate the design:

1. **Completing a half-finished cluster beats almost everything.** Adding one
   *correct* member to an entity that currently has 1 match is worth
   **+0.186** F<sub>0.5</sub>; to an entity with 2 matches, **+0.099**; with 3
   matches, only **+0.067**.
2. **One false merge is expensive.** That same correct member is worth
   **−0.259** if it is wrong. Break-even precision ≈ **57%**, which is why
   every added pair in this pipeline is gated by the LightGBM model at
   **p ≥ 0.995** rather than by string heuristics.

## Pipeline

```
data/ (challenge data, gitignored)
   │
   ├─ 1. block_exact.py ────────────► candidate_pairs.tsv   (exact name / compact /
   │                                                          postal-code blocking)
   │  2. predict_fast.py ───────────► matching_results.tsv  (LightGBM @ p≥0.995)
   │
   ├─ 3. delta_tok_band.py ─────────► round{4..7}_cand.tsv (rare-token top-up for
   │                                                          under-matched S1)
   │  4. predict_fast.py ───────────► round{4..7}_matches.tsv
   │  5. merge_adds.py ─────────────► union new pairs (never replace lists)
   │  6. union_cands.py ────────────► keep candidate file a superset
   │
   └─ 7. apply_trim.py ────────────► trim impossible oversized clusters
                                       (address-agreement first, then cap)
```

### What each step contributed

| Stage | Effect | Pairs |
|---|---|---|
| Round 4 — band-1 top-up | 100,972 entities completed | +178,417 |
| Round 5 — band-2 top-up | 136,141 entities completed | +232,186 |
| Round 6 — band-3/4 top-up | 159,840 entities completed | +247,707 |
| Round 7 — band-5/6 top-up | 38,769 entities completed | +56,121 |
| Trim oversized clusters | removed impossible false positives | −1,859,836 |

## What the data taught us (all measured on train GT)

- **Train clusters cap at 11 members, average 3.67.** The raw pipeline
  predicted 32,242 entities with ≥12 matches (median **51**) — provably
  impossible, and costly under a precision-heavy metric.
- **Trim policy is worth choosing carefully.** On 1,020 held-out train
  entities, address-agreement-first with cap 3 scored **0.6262** vs **0.5025**
  for a naive cap-11 and **0.4993** for no trim. On the 5–11 band,
  `addr+cap5` scored **0.6740** (+0.104/entity).
- **Threshold 0.995 is correct, and 0.99 would have hurt.** Pairs in the
  [0.99, 0.995) band are only **21%** correct — far below the 57% break-even.
- **Model-approved top-ups are genuine.** Added pairs are 0.66–0.73 precise
  and improve F<sub>0.5</sub> in *every* band, including the emptiest.

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the full result tables.

## Reproduce

```bash
python -m pip install -r requirements.txt

cd src
# 1-2. blocking + first-pass scoring
python block_exact.py   --split test --out ../../candidate_pairs.tsv
python predict_fast.py  --candidates ../../candidate_pairs.tsv \
                        --out ../../matching_results.tsv --threshold 0.995 --workers 6
# 3-6. under-matched band top-ups, scored and unioned
python delta_tok_band.py --matching ../../matching_results.tsv \
                         --out ../../round4_cand.tsv --cur-min 1 --cur-max 1 --k 20
python predict_fast.py  --candidates ../../round4_cand.tsv \
                         --out ../../round4_matches.tsv --threshold 0.995 --workers 6
python merge_adds.py    --base ../../matching_results.tsv \
                        --add ../../round4_matches.tsv --out ../../matching_results.tsv
python union_cands.py   --add ../../round4_cand.tsv
# 7. trim impossible clusters
python apply_trim.py    --matching ../../matching_results.tsv \
                        --out ../../matching_results.tsv --min-size 5 --cap 5
```

Validate against the official checker:

```bash
cd student_resource
python utils/validate_submission.py \
  --matching ../output/matching_results.tsv \
  --candidate ../output/candidate_pairs.tsv --test-dir dataset/test
```

> **Note:** the official validator holds two full id→set maps in RAM and
> needs >16 GB with a 120M-pair candidate file. For a 16 GB host, use the
> streaming replica (identical checks, ~1 GB peak) in
> `docs/validate_stream.py`.

## Layout

```
src/         pipeline + evaluation scripts (12 modules)
artifacts/   models_pilot_us/lgb_model.txt (6.6 MB, LightGBM text format)
submission/  matching_results.tsv  <- the deliverable (86.6 MB)
docs/        experiment log, raw eval JSON, streaming validator
```

The `artifacts/models_pilot_us/` path is deliberate: the scripts resolve the
model as `../artifacts/models_pilot_us/lgb_model.txt` relative to `src/`, so
the layout matches the original challenge workspace and the scripts run
unmodified. Pass `--data` and `--out` explicitly (as in the examples above);
they default to the original workspace root.

## Requirements & licensing

Python 3.11. `lightgbm` (MIT), `rapidfuzz` (MIT), `pandas`/`numpy`/`scipy`
(BSD), `scikit-learn` (BSD), `jellyfish` (BSD), `faiss-cpu` (MIT) — all
compatible with the challenge's model-license rule.
