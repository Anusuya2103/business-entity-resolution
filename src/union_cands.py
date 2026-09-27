"""Union candidate lists from an add-file into candidate_pairs.tsv.

Memory-light: keeps each add-row's candidate list as ONE raw string
(399k rows x ~40 ids ~ 200MB) instead of splitting into per-id strings
(which would cost >1GB). Base is streamed in chunks, order preserved,
existing ids keep their position, new ids appended.
"""
import argparse
import os
import time

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="output/candidate_pairs.tsv")
    ap.add_argument("--add", nargs="+", required=True)
    ap.add_argument("--out", default="output/candidate_pairs.tsv")
    args = ap.parse_args()

    t0 = time.time()
    add = {}
    for path in args.add:
        for ch in pd.read_csv(path, sep="\t", dtype=str,
                              keep_default_na=False, chunksize=200000):
            col = ("candidate_entity_ids" if "candidate_entity_ids"
                   in ch.columns else "matched_entity_ids")
            for s1, ids in zip(ch["source1_entity_id"], ch[col]):
                if not ids:
                    continue
                cur = add.get(s1)
                if cur is None:
                    add[s1] = ids
                else:
                    seen = set(cur.split(","))
                    extra = [c for c in ids.split(",") if c not in seen]
                    if extra:
                        add[s1] = cur + "," + ",".join(extra)
    print(f"[ucand] add_s1={len(add)} ({time.time()-t0:.0f}s)", flush=True)

    tmp = args.out + ".tmp"
    n_rows = n_cand = n_touched = 0
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for ch in pd.read_csv(args.base, sep="\t", dtype=str,
                              keep_default_na=False, chunksize=50000):
            for s1, ids in zip(ch["source1_entity_id"],
                               ch["candidate_entity_ids"]):
                extra = add.pop(s1, None)
                if extra:
                    n_touched += 1
                    seen = set(ids.split(",")) if ids else set()
                    ex = [c for c in extra.split(",") if c not in seen]
                    if ex:
                        ids = (ids + "," + ",".join(ex)) if ids \
                            else ",".join(ex)
                n_rows += 1
                if ids:
                    n_cand += ids.count(",") + 1
                f.write(s1 + "\t" + ids + "\n")
    os.replace(tmp, args.out)
    print(f"[ucand] wrote {args.out} rows={n_rows} cand_pairs={n_cand} "
          f"touched={n_touched} leftover_add_rows={len(add)} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
