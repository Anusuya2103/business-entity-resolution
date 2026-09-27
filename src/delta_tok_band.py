"""Round-4 blocking: rarest-token top-up for under-matched S1 (1..CUR_MAX
current matches). Targets the recall gap: GT analysis says band-1 entities
miss 2-3 true members on average; their exact-key candidates are exhausted,
so score fresh name/address token candidates with the model at 0.995.
Score with: predict_fast.py --candidates output/round4_cand.tsv
--out output/round4_matches.tsv --threshold 0.995 --workers 8
"""
import argparse
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from block_exact import compact, read_base
from utils import extract_pin, norm_address, norm_name, tokens

SRC = {"test": ("test/test_source1.tsv", "test/test_source2.tsv",
                "test/test_source3.tsv")}


def build_tok_index(base_nn, base_na, max_df=5000):
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


def load_band_ids(match_path, cur_min, cur_max):
    """S1 whose current match count is in [cur_min, cur_max]."""
    tgt = set()
    cur = {}
    for ch in pd.read_csv(match_path, sep="\t", dtype=str,
                          keep_default_na=False, chunksize=200000):
        for s1, ids in zip(ch["source1_entity_id"],
                           ch["matched_entity_ids"]):
            n = ids.count(",") + 1 if ids else 0
            if cur_min <= n <= cur_max:
                tgt.add(s1)
                cur[s1] = ids
    return tgt, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="student_resource/dataset")
    ap.add_argument("--matching", default="output/matching_results.tsv")
    ap.add_argument("--out", default="output/delta2_cand.tsv")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max-df", type=int, default=20000)
    ap.add_argument("--countries", default="")
    ap.add_argument("--cur-min", type=int, default=1)
    ap.add_argument("--cur-max", type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    target_ids, cur_map = load_band_ids(args.matching, args.cur_min,
                                        args.cur_max)
    print(f"[d4] band_s1={len(target_ids)} cur=[{args.cur_min},{args.cur_max}]"
          f" ({time.time()-t0:.0f}s)", flush=True)
    if not target_ids:
        print("[d4] nothing to do", flush=True)
        return

    paths = [os.path.join(args.data, x) for x in SRC["test"]]
    if args.countries:
        countries = [x for x in args.countries.split(",") if x]
    else:
        countries = []
        for path in paths:
            for ch in pd.read_csv(path, sep="\t", dtype=str,
                                   keep_default_na=False, usecols=["country"],
                                   chunksize=200000):
                countries.extend(x if x else "<EMPTY>"
                                 for x in ch["country"].unique())
        countries = sorted(set(countries))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"[d2] countries={countries} k={args.k}", flush=True)
    total_s1 = total_new = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        fout.write("source1_entity_id\tcandidate_entity_ids\n")
        for country in countries:
            t1 = time.time()
            b2, b3 = read_base(paths[1], country), read_base(paths[2], country)
            base = {k: b2[k] + b3[k] for k in b2}
            ids = base["ids"]
            is2 = np.array([True] * len(b2["ids"]) +
                           [False] * len(b3["ids"]))
            nm, cm, pm, am, sm = (defaultdict(list) for _ in range(5))
            base_nn, base_na = [], []
            for i, name in enumerate(base["names"]):
                x = norm_name(name)
                a = norm_address(base["addrs"][i])
                base_nn.append(x)
                base_na.append(a)
                if x:
                    nm[x].append(i)
                    sm[" ".join(sorted(x.split()))].append(i)
                if compact(x):
                    cm[compact(x)].append(i)
                if a:
                    am[a].append(i)
                if base["pins"][i]:
                    pm[base["pins"][i]].append(i)
            nidx, aidx, df, adf = build_tok_index(base_nn, base_na,
                                                  args.max_df)
            print(f"[d2] {country}: base={len(ids)} idx ok "
                  f"({time.time()-t1:.0f}s)", flush=True)
            n = new = 0
            for ch in pd.read_csv(paths[0], sep="\t", dtype=str,
                                   keep_default_na=False, chunksize=20000):
                cc = ch["country"].where(ch["country"] != "", "<EMPTY>")
                g = ch[cc == country]
                if len(g) == 0:
                    continue
                g = g[g["entity_id"].isin(target_ids)]
                if len(g) == 0:
                    continue
                for r in g.itertuples(index=False):
                    x = norm_name(r.business_name)
                    xs = " ".join(sorted(x.split())) if x else ""
                    na = norm_address(r.business_address)
                    pin = extract_pin(r.business_address)
                    cur_set = set(cur_map[r.entity_id].split(","))
                    exact = set(nm.get(x, ())) | set(cm.get(compact(x), ()))
                    if pin:
                        exact |= set(pm.get(pin, ()))
                    if na:
                        exact |= set(am.get(na, ()))
                    if xs:
                        exact |= set(sm.get(xs, ()))
                    exact_ids = {ids[i] for i in exact}
                    ntk = sorted(set(tokens(x)),
                                 key=lambda w: df.get(w, 0))[:2]
                    nseq = [i for w in ntk for i in nidx.get(w, ())]
                    atk = sorted(set(tokens(na)),
                                 key=lambda w: adf.get(w, 0))[:2]
                    aseq = [i for w in atk for i in aidx.get(w, ())]
                    c2 = c3 = 0
                    picked = []
                    for i in nseq + aseq:
                        c = ids[i]
                        if c in exact_ids or c in cur_set:
                            continue
                        if is2[i]:
                            if c2 >= args.k:
                                continue
                            c2 += 1
                        else:
                            if c3 >= args.k:
                                continue
                            c3 += 1
                        picked.append(c)
                        if c2 >= args.k and c3 >= args.k:
                            break
                    n += 1
                    if picked:
                        fout.write(r.entity_id + "\t" +
                                   ",".join(dict.fromkeys(picked)) + "\n")
                        new += len(set(picked))
            total_s1 += n
            total_new += new
            print(f"[d2] {country}: empty_s1={n} "
                  f"new_pairs~{new} ({time.time()-t1:.0f}s)", flush=True)
            del b2, b3, base, nm, cm, pm, am, sm, nidx, aidx
    print(f"[d2] wrote {args.out} empty_s1={total_s1} "
          f"new_pairs~{total_new} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

