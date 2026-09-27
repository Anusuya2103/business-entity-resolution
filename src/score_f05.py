"""Step 6: F0.5 scorer on validation (same macro formula as leaderboard)."""
import argparse

import pandas as pd

from utils import parse_id_list


def f05(p, r):
    if p + r == 0:
        return 0.0
    if p == 0 or r == 0:
        return 0.0
    return 1.25 * p * r / (0.25 * p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()
    pred = {}
    for chunk in pd.read_csv(args.pred, sep="\t", dtype=str,
                             keep_default_na=False, chunksize=200000):
        for r in chunk.itertuples(index=False):
            col = ("matched_entity_ids" if "matched_entity_ids" in chunk.columns
                   else "candidate_entity_ids")
            pred[r.source1_entity_id] = set(parse_id_list(getattr(r, col)))
    fs, n = [], 0
    for chunk in pd.read_csv(args.gt, sep="\t", dtype=str,
                             keep_default_na=False, chunksize=200000):
        for r in chunk.itertuples(index=False):
            if r.source1_entity_id not in pred:
                continue
            n += 1
            t = set(parse_id_list(r.matched_entity_ids))
            p = pred[r.source1_entity_id]
            if not t and not p:
                fs.append(1.0)
            elif not t or not p:
                fs.append(0.0)
            else:
                tp = len(t & p)
                fs.append(f05(tp / len(p), tp / len(t)))
    import numpy as np
    print(f"[score] entities={n} macro_F0.5={np.mean(fs):.4f}", flush=True)


if __name__ == "__main__":
    main()
