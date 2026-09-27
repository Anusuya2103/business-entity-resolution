"""Apply a trim policy to oversized predicted clusters.

Train GT caps clusters at 11 members, but the test pipeline predicts 32,225
entities with >=12 matches (median 51, p90 112). Under macro F0.5 those are
provably over-predicted: a cluster with 51 predictions and 11 true members
scores f05=0.26, while the same 11 members alone score 1.0.

This re-scores just those pairs (cheap: ~1.9M pairs) and keeps the best
CAP members per entity, optionally preferring address-agreeing pairs.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lightgbm as lgb
import numpy as np
import pandas as pd

from features_a import pair_features
from utils import extract_pin, norm_address, norm_name

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
DEF_DATA = os.path.join(ROOT, "student_resource", "dataset")
DEF_MODEL = os.path.abspath(os.path.join(
    SRC_DIR, "..", "artifacts", "models_pilot_us", "lgb_model.txt"))


def load_norm(path, need):
    look = {}
    for chunk in pd.read_csv(path, sep="\t", dtype=str,
                             keep_default_na=False, chunksize=200000):
        chunk = chunk[chunk["entity_id"].isin(need)]
        for r in chunk.itertuples(index=False):
            look[r.entity_id] = (norm_name(r.business_name),
                                 norm_address(r.business_address),
                                 r.country,
                                 extract_pin(r.business_address))
    return look


def addr_agree(na1, na2):
    if not na1 or not na2:
        return False
    if na1 == na2:
        return True
    t1, t2 = set(na1.split()), set(na2.split())
    return len(t1 & t2) / max(len(t1 | t2), 1) >= 0.67


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", default=os.path.join(
        ROOT, "output", "matching_results.tsv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=DEF_DATA)
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--min-size", type=int, default=12)
    ap.add_argument("--cap", type=int, default=11)
    ap.add_argument("--addr-first", type=int, default=1)
    ap.add_argument("--ids-from", default=None,
                    help="file with source1_entity_id column; trim exactly "
                         "these entities (overrides --min-size)")
    args = ap.parse_args()

    t0 = time.time()
    big = {}
    if args.ids_from:
        want = set()
        for ch in pd.read_csv(args.ids_from, sep="\t", dtype=str,
                              keep_default_na=False, chunksize=200000):
            col = ("source1_entity_id" if "source1_entity_id" in ch.columns
                   else ch.columns[0])
            want.update(ch[col].tolist())
        for ch in pd.read_csv(args.matching, sep="\t", dtype=str,
                              keep_default_na=False, chunksize=200000):
            for s1, ids in zip(ch["source1_entity_id"],
                               ch["matched_entity_ids"]):
                if ids and s1 in want:
                    big[s1] = ids.split(",")
        print(f"[trim] ids_from={len(want)} matched={len(big)}", flush=True)
    else:
        for ch in pd.read_csv(args.matching, sep="\t", dtype=str,
                              keep_default_na=False, chunksize=200000):
            for s1, ids in zip(ch["source1_entity_id"],
                               ch["matched_entity_ids"]):
                if not ids:
                    continue
                if ids.count(",") + 1 >= args.min_size:
                    big[s1] = ids.split(",")
    n_pairs_in = sum(len(v) for v in big.values())
    print(f"[trim] oversized={len(big)} pairs={n_pairs_in} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if not big:
        print("[trim] nothing to trim", flush=True)
        return

    need = set(big)
    for v in big.values():
        need.update(v)
    norms = {}
    for f in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv"):
        norms.update(load_norm(os.path.join(args.data, "test", f), need))
    print(f"[trim] norms={len(norms)} ({time.time()-t0:.0f}s)", flush=True)

    model = lgb.Booster(model_file=args.model)
    probs = {}
    batch = []

    def flush():
        if not batch:
            return
        pr = model.predict(np.asarray([pair_features(*b[1])
                                       for b in batch], dtype=np.float32),
                           num_threads=4)
        for b, p in zip(batch, pr):
            probs[b[0]] = float(p)
        batch.clear()

    for s1, mem in big.items():
        a = norms.get(s1)
        if a is None:
            continue
        for c in mem:
            b = norms.get(c)
            if b is None:
                continue
            batch.append(((s1, c), (a[0], a[1], a[2], a[3],
                                    b[0], b[1], b[2], b[3],
                                    c[:3] == "S2-")))
            if len(batch) >= 50000:
                flush()
    flush()
    print(f"[trim] scored={len(probs)} ({time.time()-t0:.0f}s)", flush=True)

    new = {}
    n_kept = n_drop = n_addr_kept = 0
    for s1, mem in big.items():
        a = norms.get(s1)
        if a is None:
            continue
        cand = []
        for c in mem:
            if (s1, c) not in probs:
                continue
            b = norms[c]
            cand.append((probs[(s1, c)], addr_agree(a[1], b[1]), c))
        cand.sort(key=lambda x: (-x[1], -x[0]) if args.addr_first
                  else (-x[0],))
        keep = [x[2] for x in cand[:args.cap]]
        n_addr_kept += sum(1 for x in cand[:args.cap] if x[1])
        if keep:
            new[s1] = ",".join(keep)
        n_kept += len(keep)
        n_drop += len(mem) - len(keep)
    print(f"[trim] kept_pairs={n_kept} dropped={n_drop} "
          f"addr_agree_kept={n_addr_kept} ({time.time()-t0:.0f}s)", flush=True)

    tmp = args.out + ".tmp"
    n_rows = n_non = 0
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for ch in pd.read_csv(args.matching, sep="\t", dtype=str,
                              keep_default_na=False, chunksize=100000):
            for row in ch.itertuples(index=False):
                s1 = row.source1_entity_id
                ids = new.get(s1, row.matched_entity_ids)
                n_rows += 1
                if ids:
                    n_non += 1
                f.write(s1 + "\t" + ids + "\n")
    os.replace(tmp, args.out)
    print(f"[trim] wrote {args.out} rows={n_rows} nonempty={n_non} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
