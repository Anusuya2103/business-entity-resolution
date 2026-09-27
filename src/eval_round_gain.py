"""Train-side: are the token-blocked TOP-UP additions actually correct?

Rounds 4-7 add rarest-name/address-token candidates to under-matched S1 and
keep the ones the model approves at 0.995. The eval_thr_band run showed pair
precision is 0.87/0.85 for entities ending at 1/2 matches but only 0.61 for
3+. That mixes the original v1 pairs with the top-up adds, so this measures
the ADDED pairs only, bucketed by how many matches the entity had before.

Protocol (per country, on a train sample):
  1. base = exact/compact/PIN candidates scored at 0.995  (== v1 state)
  2. topup = rarest name/addr token candidates, EXCLUDING base matches and
     base candidates, scored at 0.995
  3. report precision of (2) per base-band (1, 2, 3-4, 5+)
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lightgbm as lgb
import numpy as np
import pandas as pd

from block_exact import compact, read_base
from features_a import FEAT_COLS, pair_features
from utils import (extract_pin, norm_address, norm_name, parse_id_list,
                   tokens)


def f05(p, r):
    if p + r == 0 or p == 0 or r == 0:
        return 0.0
    return 1.25 * p * r / (0.25 * p + r)


def build_tok_index(base_nn, base_na, max_df):
    df = Counter()
    for t in base_nn:
        for w in set(tokens(t)):
            df[w] += 1
    nidx = defaultdict(list)
    for i, t in enumerate(base_nn):
        for w in set(tokens(t)):
            if len(w) >= 3 and df[w] <= max_df:
                nidx[w].append(i)
    adf = Counter()
    for t in base_na:
        for w in set(tokens(t)):
            adf[w] += 1
    aidx = defaultdict(list)
    for i, t in enumerate(base_na):
        for w in set(tokens(t)):
            if len(w) > 2 and adf[w] <= max_df:
                aidx[w].append(i)
    return nidx, aidx, df, adf


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
    s1, gt = {}, {}
    for chunk in pd.read_csv(
            os.path.join(args.data, "train/train_source1.tsv"),
            sep="\t", dtype=str, keep_default_na=False, chunksize=200000):
        hit = chunk[chunk["entity_id"].isin(keep)]
        for r in hit.itertuples(index=False):
            s1[r.entity_id] = (r.business_name, r.business_address,
                               r.country or "<EMPTY>")
    for chunk in pd.read_csv(
            os.path.join(args.data, "train/train_ground_truth.tsv"),
            sep="\t", dtype=str, keep_default_na=False, chunksize=200000):
        hit = chunk[chunk["source1_entity_id"].isin(keep)]
        for r in hit.itertuples(index=False):
            gt[r.source1_entity_id] = parse_id_list(r.matched_entity_ids)
    print(f"[rg] sample={len(keep)} ({time.time()-t0:.0f}s)", flush=True)
    return s1, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../../student_resource/dataset")
    ap.add_argument("--model",
                    default="../artifacts/models_pilot_us/lgb_model.txt")
    ap.add_argument("--n-per-country", type=int, default=1200)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max-df", type=int, default=20000)
    ap.add_argument("--thr", type=float, default=0.995)
    ap.add_argument("--batch", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="../artifacts/eval_round_gain.json")
    args = ap.parse_args()

    t0 = time.time()
    s1, gt = load_sample(args)
    model = lgb.Booster(model_file=args.model)
    recs = {}
    for ctry in sorted({v[2] for v in s1.values()}):
        t1 = time.time()
        s2 = read_base(os.path.join(args.data, "train/train_source2.tsv"), ctry)
        s3 = read_base(os.path.join(args.data, "train/train_source3.tsv"), ctry)
        n2 = len(s2["ids"])
        base = {k: s2[k] + s3[k] for k in s2}
        is2 = [True] * n2 + [False] * len(s3["ids"])
        del s2, s3
        ids = base["ids"]
        nn = [norm_name(x) for x in base["names"]]
        na = [norm_address(x) for x in base["addrs"]]
        nm, cm, pm, am = (defaultdict(list), defaultdict(list),
                          defaultdict(list), defaultdict(list))
        for i in range(len(ids)):
            if nn[i]:
                nm[nn[i]].append(i)
            c = compact(nn[i])
            if c:
                cm[c].append(i)
            if na[i]:
                am[na[i]].append(i)
            if base["pins"][i]:
                pm[base["pins"][i]].append(i)
        nidx, aidx, df, adf = build_tok_index(nn, na, args.max_df)
        print(f"[rg] {ctry}: base={len(ids)} idx "
              f"({time.time()-t1:.0f}s)", flush=True)

        sample = [(k, v) for k, v in s1.items() if v[2] == ctry]
        Xb, Kb = [], []

        def flush():
            if not Xb:
                return
            pr = model.predict(np.array(Xb, dtype=np.float32))
            for (s1_id, cid, topup), p in zip(Kb, pr):
                recs.setdefault(s1_id, []).append((float(p), cid, topup))
            Xb.clear()
            Kb.clear()

        for s1_id, (nm_s, ad_s, cc) in sample:
            x = norm_name(nm_s)
            xa = norm_address(ad_s)
            pin = extract_pin(ad_s)
            exact = set(nm.get(x, ())) | set(cm.get(compact(x), ()))
            if pin:
                exact |= set(pm.get(pin, ()))
            if xa:
                exact |= set(am.get(xa, ()))
            ntk = sorted(set(tokens(x)), key=lambda w: df.get(w, 0))[:2]
            atk = sorted(set(tokens(xa)), key=lambda w: adf.get(w, 0))[:2]
            topup = []
            for w in ntk:
                topup.extend(nidx.get(w, ()))
            for w in atk:
                topup.extend(aidx.get(w, ()))
            k2 = k3 = 0
            for i in topup:
                if i in exact:
                    continue
                if is2[i]:
                    if k2 >= args.k:
                        continue
                    k2 += 1
                else:
                    if k3 >= args.k:
                        continue
                    k3 += 1
                Xb.append(pair_features(x, xa, cc, pin, base["names"][i],
                                        na[i], base["countries"][i],
                                        base["pins"][i], is2[i]))
                Kb.append((s1_id, ids[i], True))
                if len(Xb) >= args.batch:
                    flush()
            for i in exact:
                Xb.append(pair_features(x, xa, cc, pin, base["names"][i],
                                        na[i], base["countries"][i],
                                        base["pins"][i], is2[i]))
                Kb.append((s1_id, ids[i], False))
                if len(Xb) >= args.batch:
                    flush()
        flush()
        print(f"[rg] {ctry}: cum={sum(len(v) for v in recs.values())} "
              f"({time.time()-t1:.0f}s)", flush=True)
        del base, nm, cm, pm, am, nidx, aidx, nn, na

    out = defaultdict(lambda: dict(add_pairs=0, add_tp=0, ents=0,
                                  truth=0, base_pairs=0, base_tp=0,
                                  f05_base=0.0, f05_add=0.0))
    for s1_id, rows in recs.items():
        t = gt.get(s1_id, [])
        if not t:
            continue
        tset = set(t)
        basep = [(p, c) for p, c, top in rows if not top and p >= args.thr]
        addp = [(p, c) for p, c, top in rows if top and p >= args.thr]
        k = len(basep)
        band = k if k <= 2 else (3 if k <= 4 else 5)
        d = out[band]
        d["ents"] += 1
        d["truth"] += len(t)
        d["base_pairs"] += k
        btp = sum(1 for p, c in basep if c in tset)
        d["base_tp"] += btp
        d["add_pairs"] += len(addp)
        d["add_tp"] += sum(1 for p, c in addp if c in tset)
        d["f05_base"] += f05(btp / k, btp / len(t)) if k else 0.0
        k2 = k + len(addp)
        tp2 = btp + sum(1 for p, c in addp if c in tset)
        d["f05_add"] += f05(tp2 / k2, tp2 / len(t)) if k2 else 0.0

    res = {}
    for band, d in sorted(out.items()):
        res[str(band)] = dict(
            ents=d["ents"], avg_truth=round(d["truth"] / d["ents"], 2),
            base_pairs=d["base_pairs"],
            base_precision=round(d["base_tp"] / max(d["base_pairs"], 1), 4),
            add_pairs=d["add_pairs"],
            add_precision=round(d["add_tp"] / max(d["add_pairs"], 1), 4),
            f05_base=round(d["f05_base"] / d["ents"], 4),
            f05_after_adds=round(d["f05_add"] / d["ents"], 4),
            delta=round((d["f05_add"] - d["f05_base"]) / d["ents"], 4))
        print(f"[rg] band={band} {json.dumps(res[str(band)])}", flush=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[rg] wrote {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

