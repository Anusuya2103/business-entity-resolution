"""Union-merge model-approved add pairs into a base matching file.

Safe merge (the earlier pandas .update() REPLACED match lists - never use
it here): for each S1 present in `--add`, union its new ids onto the base
list, base first, then new. Entities absent from --add are untouched.
"""
import argparse
import os
import sys
import time

import pandas as pd


def read_map(path):
    mp = {}
    for ch in pd.read_csv(path, sep="\t", dtype=str,
                          keep_default_na=False, chunksize=200000):
        col = ("matched_entity_ids" if "matched_entity_ids" in ch.columns
               else "candidate_entity_ids")
        for s1, ids in zip(ch["source1_entity_id"], ch[col]):
            if ids:
                mp[s1] = ids.split(",")
    return mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--add", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    base = read_map(args.base)
    add = read_map(args.add)
    print(f"[merge] base={len(base)} add={len(add)} "
          f"({time.time()-t0:.0f}s)", flush=True)
    touched = added = 0
    for s1, ids in add.items():
        cur = base.setdefault(s1, [])
        seen = set(cur)
        new = [c for c in ids if c not in seen]
        if new:
            cur.extend(new)
            touched += 1
            added += len(new)
    print(f"[merge] touched={touched} pairs_added={added} "
          f"({time.time()-t0:.0f}s)", flush=True)

    n_non = n_pairs = n_rows = 0
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        src = pd.read_csv(args.base, sep="\t", dtype=str,
                          keep_default_na=False,
                          usecols=["source1_entity_id"], chunksize=200000)
        for ch in src:
            for s1 in ch["source1_entity_id"]:
                ids = base.get(s1)
                s = ",".join(ids) if ids else ""
                if s:
                    n_non += 1
                    n_pairs += len(ids)
                n_rows += 1
                f.write(s1 + "\t" + s + "\n")
    os.replace(tmp, args.out)
    print(f"[merge] wrote {args.out} rows={n_rows} nonempty={n_non} "
          f"pairs={n_pairs} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
