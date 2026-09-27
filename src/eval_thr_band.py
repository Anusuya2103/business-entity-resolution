"""Train-side: is a 0.99 threshold better than 0.995 for UNDER-MATCHED S1?

The near-miss probe found ~0.5% of band-1/2 candidates score in
[0.990, 0.995). Under macro F0.5 adding a correct member to a 1-match
entity is worth +0.195 but a wrong one costs -0.259, so the break-even
precision is ~57%. This measures macro F0.5 for band-1/2 entities at
several thresholds to decide whether a 0.99 pass is worth ~150k pairs.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lightgbm as lgb
import numpy as np
import pandas as pd

from block_exact import compact, read_base
from features_a import FEAT_COLS, pair_features
from utils import extract_pin, norm_address, norm_name, parse_id_list


def f05(p, r):
    if p + r == 0 or p == 0 or r == 0:
        return 0.0
    return 1.25 * p * r / (0.25 * p + r)


def load_sample(args):
    t0 = time.time()
    rng = np.random.RandomState(args.seed)
    by_c = defaultdict(list)
    for chunk in pd.read_csv(
            os.path.join(args.data, "train/train_source1.tsv"),
            sep="\t", dtype=str, keep_default_na=False,
            usecols=["entity_id", "country"], chunksize=200000):
        for r in chunk.itertuples(index=False):
            by_c[r.country or "<EMPTY>"].append(r.entity_id)
    keep = set()
    for c in sorted(by_c):
        ids = by_c[c]
        for i in rng.permutation(len(ids))[:args.n_per_country]:
            keep.add(ids[i])
    by_c.clear()
    s1 = {}
    for chunk in pd.read_csv(
            os.path.join(args.data, "train/train_source1.tsv"),
            sep="\t", dtype=str, keep_default_na=False, chunksize=200000):
        hit = chunk[chunk["entity_id"].isin(keep)]
        for r in hit.itertuples(index=False):
            s1[r.entity_id] = (r.business_name, r.business_address,
                               r.country or "<EMPTY>")
    gt = {}
    for chunk in pd.read_csv(
            os.path.join(args.data, "train/train_ground_truth.tsv"),
            sep="\t", dtype=str, keep_default_na=False, chunksize=200000):
        hit = chunk[chunk["source1_entity_id"].isin(keep)]
        for r in hit.itertuples(index=False):
            gt[r.source1_entity_id] = parse_id_list(r.matched_entity_ids)
    print(f"[thr] sample={len(keep)} gt={len(gt)} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return s1, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../../student_resource/dataset")
    ap.add_argument("--model",
                    default="../artifacts/models_pilot_us/lgb_model.txt")
    ap.add_argument("--n-per-country", type=int, default=2000)
    ap.add_argument("--max-entities", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="../artifacts/eval_thr_band.json")
    args = ap.parse_args()

    t0 = time.time()
    s1, gt = load_sample(args)
    model = lgb.Booster(model_file=args.model)
    THRS = [0.95, 0.98, 0.99, 0.995]
    # s1 -> list[(score, cid)]
    scored = defaultdict(list)
    for ctry in sorted({v[2] for v in s1.values()}):
        t1 = time.time()
        s2 = read_base(os.path.join(args.data, "train/train_source2.tsv"), ctry)
        s3 = read_base(os.path.join(args.data, "train/train_source3.tsv"), ctry)
        n2 = len(s2["ids"])
        base = {k: s2[k] + s3[k] for k in s2}
        is2 = [True] * n2 + [False] * len(s3["ids"])
        del s2, s3
        nm, cm, pm = defaultdict(list), defaultdict(list), defaultdict(list)
        for i, name in enumerate(base["names"]):
            x = norm_name(name)
            if x:
                nm[x].append(i)
            x = compact(x)
            if x:
                cm[x].append(i)
            if base["pins"][i]:
                pm[base["pins"][i]].append(i)
        print(f"[thr] {ctry}: base={len(base['ids'])} "
              f"({time.time()-t1:.0f}s)", flush=True)

        sel = []
        for s1_id, (nm_s, ad_s, cc) in s1.items():
            if cc != ctry:
                continue
            x = norm_name(nm_s)
            inds = set(nm.get(x, ())) | set(cm.get(compact(x), ()))
            pin = extract_pin(ad_s)
            if pin:
                inds |= set(pm.get(pin, ()))
            if inds:
                sel.append((s1_id, x, ad_s, cc, list(inds)[:args.max_entities]))
        print(f"[thr] {ctry}: entities={len(sel)} "
              f"({time.time()-t1:.0f}s)", flush=True)

        Xb, Kb = [], []

        def flush():
            if not Xb:
                return
            pr = model.predict(np.array(Xb, dtype=np.float32))
            for (s1_id, cid), p in zip(Kb, pr):
                scored[s1_id].append((float(p), cid))
            Xb.clear()
            Kb.clear()

        for s1_id, nn1, ad_s, cc1, inds in sel:
            na1 = norm_address(ad_s)
            pin1 = extract_pin(ad_s)
            for i in inds:
                Xb.append(pair_features(nn1, na1, cc1, pin1,
                                        base["names"][i],
                                        norm_address(base["addrs"][i]),
                                        base["countries"][i], base["pins"][i],
                                        is2[i]))
                Kb.append((s1_id, base["ids"][i]))
                if len(Xb) >= args.batch:
                    flush()
        flush()
        print(f"[thr] {ctry}: cum={sum(len(v) for v in scored.values())} "
              f"({time.time()-t1:.0f}s)", flush=True)
        del base, nm, cm, pm

    # ---- band-wise macro F0.5 + pair precision per threshold ----
    out = {}
    for T in THRS:
        band_scores = defaultdict(list)
        for s1_id, pairs in scored.items():
            t = gt.get(s1_id, [])
            if not t:
                continue
            tset = set(t)
            sel_pairs = [(p, c) for p, c in pairs if p >= T]
            k = len(sel_pairs)
            if k == 0:
                continue
            tp = sum(1 for p, c in sel_pairs if c in tset)
            # current size after threshold, bucket 1 / 2 / 3+
            b = k if k <= 2 else 3
            band_scores[b].append((f05(tp / k, tp / len(t)), tp, k))
        row = {}
        for b in (1, 2, 3):
            v = band_scores.get(b, [])
            if not v:
                continue
            row["band%d" % b] = dict(
                n=len(v),
                f05=round(float(np.mean([x[0] for x in v])), 4),
                pair_precision=round(
                    sum(x[1] for x in v) / max(sum(x[2] for x in v), 1), 4))
        # precision specifically in the [0.99, 0.995) band
        if T == 0.99:
            num = den = 0
            for s1_id, pairs in scored.items():
                t = gt.get(s1_id, [])
                if not t:
                    continue
                tset = set(t)
                for p, c in pairs:
                    if 0.99 <= p < 0.995:
                        den += 1
                        if c in tset:
                            num += 1
            row["near_miss_0.99_0.995"] = dict(
                n=den, precision=round(num / max(den, 1), 4))
        out["%.3f" % T] = row
        print(f"[thr] T={T}: {json.dumps(row)}", flush=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[thr] wrote {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
