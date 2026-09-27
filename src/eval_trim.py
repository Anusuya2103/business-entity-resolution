"""Train-side experiment: what to do with oversized predicted clusters?

Test pipeline predicts 32,225 entities with >=12 members (median 51) while
train GT caps at 11 - those clusters provably contain FPs, and under macro
F0.5 each one costs real score. This evaluates trim policies on TRAIN where
the truth is known, restricted to the affected population (S1 whose exact-key
blocking already yields >= --min-cands candidates).

Policies compared (macro F0.5 on that population):
  base        - model >= --thr, keep all
  thr<T>      - model >= T
  cap<K>      - keep top-K by model prob
  addr        - keep only address-agreeing pairs
  addr+cap<K> - keep address-agreeing, then top up to K by prob

Usage:
  python eval_trim.py --n-per-country 4000 --min-cands 12
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


def addr_agree(na1, na2):
    if not na1 or not na2:
        return False
    if na1 == na2:
        return True
    t1, t2 = set(na1.split()), set(na2.split())
    return len(t1 & t2) / max(len(t1 | t2), 1) >= 0.67


def score_entity(truth, pairs):
    """pairs: list of (prob, addr_agree, cid). Returns {policy: f05}."""
    out = {}
    tset = set(truth)

    def ev(sel):
        sel = list(sel)
        if not sel:
            return 0.0
        tp = sum(1 for x in sel if x[2] in tset)
        return f05(tp / len(sel), tp / len(truth))

    out["base"] = ev(pairs)
    for T in (0.999, 0.9995, 0.9999):
        out[f"thr{T}"] = ev([x for x in pairs if x[0] >= T])
    byp = sorted(pairs, key=lambda x: -x[0])
    for K in (1, 2, 3, 5, 8, 11):
        out[f"cap{K}"] = ev(byp[:K])
    ag = [x for x in pairs if x[1]]
    out["addr"] = ev(ag)
    for K in (3, 5, 11):
        keep = (ag + byp)[:K] if ag else byp[:K]
        out[f"addr+cap{K}"] = ev(keep)
    return out


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
    print(f"[trim] sample={len(keep)} s1_fields={len(s1)} gt={len(gt)} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return s1, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../../student_resource/dataset")
    ap.add_argument("--model",
                    default="../artifacts/models_pilot_us/lgb_model.txt")
    ap.add_argument("--n-per-country", type=int, default=4000)
    ap.add_argument("--max-entities", type=int, default=3000)
    ap.add_argument("--min-cands", type=int, default=12)
    ap.add_argument("--thr", type=float, default=0.995)
    ap.add_argument("--batch", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="../artifacts/eval_trim.json")
    args = ap.parse_args()

    t0 = time.time()
    s1, gt = load_sample(args)
    model = lgb.Booster(model_file=args.model)
    scores = defaultdict(list)
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
        print(f"[trim] {ctry}: base={len(base['ids'])} "
              f"({time.time()-t1:.0f}s)", flush=True)

        big = []
        for s1_id, (nm_s, ad_s, cc) in s1.items():
            if cc != ctry:
                continue
            x = norm_name(nm_s)
            inds = set(nm.get(x, ())) | set(cm.get(compact(x), ()))
            pin = extract_pin(ad_s)
            if pin:
                inds |= set(pm.get(pin, ()))
            if len(inds) >= args.min_cands:
                big.append((s1_id, x, ad_s, cc, inds))
        big.sort(key=lambda z: -len(z[4]))
        big = big[:args.max_entities]
        print(f"[trim] {ctry}: big_entities={len(big)} "
              f"({time.time()-t1:.0f}s)", flush=True)

        Xb, Kb = [], []

        def flush():
            if not Xb:
                return
            pr = model.predict(np.array(Xb, dtype=np.float32))
            for key, p in zip(Kb, pr):
                s1_id, na1, na2, cid = key
                scores[s1_id].append(
                    (float(p), addr_agree(na1, na2), cid))
            Xb.clear()
            Kb.clear()

        for s1_id, nn1, ad_s, cc1, inds in big:
            na1 = norm_address(ad_s)
            pin1 = extract_pin(ad_s)
            for i in inds:
                na2 = norm_address(base["addrs"][i])
                Xb.append(pair_features(nn1, na1, cc1, pin1,
                                        base["names"][i], na2,
                                        base["countries"][i], base["pins"][i],
                                        is2[i]))
                Kb.append((s1_id, na1, na2, base["ids"][i]))
                if len(Xb) >= args.batch:
                    flush()
        flush()
        print(f"[trim] {ctry}: cum_scored="
              f"{sum(len(v) for v in scores.values())} "
              f"({time.time()-t1:.0f}s)", flush=True)
        del base, nm, cm, pm

    agg = defaultdict(list)
    sizes, n_truth = [], 0
    for s1_id, pairs in scores.items():
        t = gt.get(s1_id, [])
        sizes.append(len(t))
        if not t:
            continue
        n_truth += 1
        sel = [x for x in pairs if x[0] >= args.thr]
        for k, v in score_entity(t, sel).items():
            agg[k].append(v)
    res = {k: float(np.mean(v)) for k, v in agg.items()}
    base_v = res.get("base", 0.0)
    print(f"[trim] big_entities={len(scores)} with_truth={n_truth} "
          f"avg_truth={np.mean(sizes):.2f} max_truth={max(sizes)}", flush=True)
    for k in sorted(res, key=lambda k: -res[k]):
        print(f"[trim] {k:14s} f05={res[k]:.4f} "
              f"delta={res[k]-base_v:+.4f}", flush=True)
    json.dump({"policies": res, "n_big": len(scores), "n_with_truth": n_truth,
               "avg_truth_size": float(np.mean(sizes)),
               "max_truth_size": int(max(sizes)),
               "args": {k: v for k, v in vars(args).items()}},
              open(args.out, "w"), indent=2)
    print(f"[trim] wrote {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
