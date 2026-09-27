"""Resource-safe exact/compact/PIN candidate generation with inline model.

This fallback keeps the complete submission contract while avoiding a large
TF-IDF/FAISS matrix. It uses exact normalized-name, compact-name, and exact
5-6 digit postal blocks, caps each source, and feeds exactly those candidates
to LightGBM.
"""
import argparse
import json
import os
import re
from collections import defaultdict

import lightgbm as lgb
import pandas as pd

from features_a import pair_features
from utils import extract_pin, norm_name

SRC = {"train": ("train/train_source1.tsv", "train/train_source2.tsv",
                 "train/train_source3.tsv"),
       "test": ("test/test_source1.tsv", "test/test_source2.tsv",
                "test/test_source3.tsv")}


def compact(s):
    return re.sub(r"\s+", "", s or "")


def country_list(paths):
    out = set()
    for path in paths:
        for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                              usecols=["country"], chunksize=200000):
            out.update(x if x else "<EMPTY>" for x in ch["country"].unique())
    return sorted(out)


def read_base(path, country):
    ids, names, addrs, countries, pins = [], [], [], [], []
    use = ["entity_id", "business_name", "business_address", "country"]
    for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                          usecols=use, chunksize=100000):
        if country is not None:
            cc = ch["country"].where(ch["country"] != "", "<EMPTY>")
            ch = ch[cc == country]
        if len(ch):
            ids.extend(ch["entity_id"].tolist())
            names.extend(ch["business_name"].tolist())
            addrs.extend(ch["business_address"].tolist())
            countries.extend(ch["country"].tolist())
            pins.extend(ch["business_address"].map(extract_pin).tolist())
    return dict(ids=ids, names=names, addrs=addrs,
                countries=countries, pins=pins)


def run(args):
    paths = [os.path.join(args.data, x) for x in SRC[args.split]]
    countries = [x for x in args.countries.split(",") if x] or country_list(paths)
    model = lgb.Booster(model_file=args.model) if args.model else None
    all_ids, matches, stats = [], {}, {}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print("[block_exact] countries=%s" % countries, flush=True)
    with open(args.out, "w", encoding="utf-8") as fout:
        fout.write("source1_entity_id\tcandidate_entity_ids\n")
        for country in countries:
            b2, b3 = read_base(paths[1], country), read_base(paths[2], country)
            base = {k: b2[k] + b3[k] for k in b2}
            is2 = [True] * len(b2["ids"]) + [False] * len(b3["ids"])
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
            n = total = 0
            for ch in pd.read_csv(paths[0], sep="\t", dtype=str,
                                  keep_default_na=False, chunksize=20000):
                cc = ch["country"].where(ch["country"] != "", "<EMPTY>")
                g = ch[cc == country]
                if args.s1_limit and n >= args.s1_limit:
                    break
                if args.s1_limit:
                    g = g.iloc[:args.s1_limit - n]
                for r in g.itertuples(index=False):
                    x = norm_name(r.business_name)
                    inds = set(nm.get(x, ())) | set(cm.get(compact(x), ()))
                    pin = extract_pin(r.business_address)
                    if pin:
                        inds.update(pm.get(pin, ()))
                    keep = []
                    for flag in (True, False):
                        part = [i for i in inds if is2[i] == flag][:args.k]
                        keep.extend(part)
                    cids = [base["ids"][i] for i in keep]
                    fout.write(r.entity_id + "\t" + ",".join(cids) + "\n")
                    all_ids.append(r.entity_id)
                    n += 1
                    total += len(cids)
                    # High-precision final rule for this resource-safe run:
                    # retain only exact normalized or compact-name candidates.
                    # PIN-only candidates remain available to the model in the
                    # candidate file but are not auto-merged.
                    exact_ids = set(nm.get(x, ())) | set(cm.get(compact(x), ()))
                    cids = [base["ids"][i] for i in keep
                            if i in exact_ids]
                    matches[r.entity_id] = cids
                if n and n % 20000 == 0:
                    print("   [%s] s1=%d avg=%.1f" % (country, n,
                                                        total / max(n, 1)), flush=True)
            stats[country] = dict(s1=n, candidates=total)
            print("   [%s] done s1=%d avg=%.1f" % (country, n,
                                                     total / max(n, 1)), flush=True)
    if args.matching_out:
        with open(args.matching_out, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1 in all_ids:
                f.write(s1 + "\t" + ",".join(dict.fromkeys(matches.get(s1, []))) + "\n")
    with open(os.path.splitext(args.out)[0] + "_stats.json", "w") as f:
        json.dump(dict(split=args.split, countries=stats, s1=len(all_ids),
                       matched_entities=len(matches)), f, indent=2)
    print("[block_exact] wrote %s rows=%d" % (args.out, len(all_ids)), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../../student_resource/dataset")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--matching-out", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--s1-limit", type=int, default=0)
    ap.add_argument("--countries", default="")
    run(ap.parse_args())

