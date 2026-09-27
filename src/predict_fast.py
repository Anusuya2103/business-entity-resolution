"""Fast parallel inference: score candidate_pairs.tsv -> matching_results.tsv.

Why this exists (vs predict.py):
    predict.py scores all ~51M candidate pairs in a single process and
    re-normalizes both entities on every pair (~57 us/pair -> ~50+ min).
    This variant:
      1. splits candidate_pairs.tsv into byte-range shards (line-aligned),
      2. scores each shard in its own process (multiprocessing spawn pool),
      3. normalizes each entity exactly ONCE per shard and hoists S1-side
         token sets out of the pair loop,
      4. writes passing pairs (score >= threshold) to per-shard part files,
      5. merges parts in test_source1.tsv order into the final submission TSV.

Usage (from src/):
    python predict_fast.py --workers 5 --threshold 0.995
    python predict_fast.py --self-test 2000   # parity check vs pair_features
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lightgbm as lgb
import numpy as np
import pandas as pd

from features_a import FEAT_COLS, pair_features, s_metrics
from utils import extract_pin, norm_address, norm_name

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
DEF_DATA = os.path.join(ROOT, "student_resource", "dataset")
DEF_CAND = os.path.join(ROOT, "output", "candidate_pairs.tsv")
DEF_MODEL = os.path.abspath(os.path.join(
    SRC_DIR, "..", "artifacts", "models_pilot_us", "lgb_model.txt"))
DEF_OUT = os.path.join(ROOT, "output", "matching_results.tsv")
PARTS_DIR = os.path.join(ROOT, "output", "_score_parts")


def shard_offsets(path, n):
    """Line-aligned byte offsets splitting `path` into n shards."""
    size = os.path.getsize(path)
    offs = [0]
    with open(path, "rb") as f:
        for i in range(1, n):
            f.seek(size * i // n)
            f.readline()  # advance to start of next line
            pos = f.tell()
            if offs[-1] < pos < size:
                offs.append(pos)
    offs.append(size)
    return offs


def load_norm(path, need=None):
    """entity_id -> (norm_name, norm_address, country, pin), filtered by need."""
    look = {}
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                             chunksize=200000):
        if need is not None:
            chunk = chunk[chunk["entity_id"].isin(need)]
        for r in chunk.itertuples(index=False):
            look[r.entity_id] = (norm_name(r.business_name),
                                 norm_address(r.business_address),
                                 r.country,
                                 extract_pin(r.business_address))
    return look



def run_shard(task):
    """Score one byte-range shard; write passing pairs to its part file."""
    (wid, start, end, cand_path, data_dir, model_path,
     thr, part_path, batch) = task
    if start >= end:
        with open(part_path, "w", encoding="utf-8"):
            pass
        return wid, 0, 0, 0.0
    t0 = time.time()
    # ---- pass 1: ids needed by this shard (file is line-aligned) ----
    need_s1, need_c = set(), set()
    with open(cand_path, "rb") as f:
        f.seek(start)
        while True:
            pos = f.tell()
            if pos >= end:
                break
            line = f.readline()
            if not line:
                break
            if pos == 0:
                continue  # header row
            j = line.find(b"\t")
            need_s1.add(line[:j].decode("utf-8"))
            cs = line[j + 1:].rstrip(b"\r\n")
            if cs:
                need_c.update(cs.decode("utf-8").split(","))
    # ---- pass 2: lookups, normalized exactly once per entity ----
    tsub = os.path.join(data_dir, "test")
    l1 = load_norm(os.path.join(tsub, "test_source1.tsv"), need_s1)
    l2 = load_norm(os.path.join(tsub, "test_source2.tsv"),
                   {x for x in need_c if x[:3] == "S2-"})
    l3 = load_norm(os.path.join(tsub, "test_source3.tsv"),
                   {x for x in need_c if x[:3] == "S3-"})
    look = {**l2, **l3}
    del l2, l3, need_c
    model = lgb.Booster(model_file=model_path)
    print(f"[w{wid}] loaded s1={len(l1)} cand={len(look)} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- pass 3: score ----
    n_pairs = n_pass = 0
    next_rep = 2_000_000
    Xb, Kb = [], []

    def flush():
        nonlocal n_pass
        if not Xb:
            return
        p = model.predict(np.asarray(Xb, dtype=np.float32), num_threads=2)
        for (s, cid2), sc in zip(Kb, p):
            if sc >= thr:
                out.write(f"{s}\t{cid2}\t{sc:.6f}\n")
                n_pass += 1
        Xb.clear()
        Kb.clear()

    with open(cand_path, "rb") as f, \
            open(part_path, "w", encoding="utf-8") as out:
        f.seek(start)
        while True:
            pos = f.tell()
            if pos >= end:
                break
            line = f.readline()
            if not line:
                break
            if pos == 0:
                continue
            j = line.find(b"\t")
            s1 = line[:j].decode("utf-8")
            cs = line[j + 1:].rstrip(b"\r\n")
            rec = l1.get(s1)
            if rec is None or not cs:
                continue
            nn1, na1, cc1, pin1 = rec
            # S1-side invariants hoisted out of the pair loop
            tn1 = set(nn1.split())
            ta1 = set(na1.split())
            lnn1, lna1 = len(nn1), len(na1)
            for cid in cs.decode("utf-8").split(","):
                c = look.get(cid)
                if c is None:
                    continue
                nn2, na2, cc2, pin2 = c
                n = s_metrics(nn1, nn2)
                a = s_metrics(na1, na2)
                pin_both = bool(pin1 and pin2)
                pin_match = bool(pin_both and pin1 == pin2)
                tn2 = set(nn2.split())
                ta2 = set(na2.split())
                Xb.append([
                    n["ratio"], n["sort"], n["sett"], n["jw"], n["jac"],
                    a["ratio"], a["sort"], a["sett"], a["jw"], a["jac"],
                    abs(lnn1 - len(nn2)), abs(lna1 - len(na2)),
                    float(bool(nn1) and nn1 == nn2),
                    float(bool(na1) and na1 == na2),
                    float(pin_match), float(pin_both),
                    float((cc1 or "") == (cc2 or "")),
                    float(cid[:3] == "S2-"),
                    (n["sort"] / 100.0) * (a["sort"] / 100.0),
                    len(tn1 & tn2) / max(len(tn1 | tn2), 1),
                    len(ta1 & ta2) / max(len(ta1 | ta2), 1),
                    float((not na1) or (not na2)),
                ])
                Kb.append((s1, cid))
                n_pairs += 1
                if len(Xb) >= batch:
                    flush()
            if n_pairs >= next_rep:
                print(f"[w{wid}] pairs={n_pairs} pass={n_pass} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                next_rep += 2_000_000
        flush()
    print(f"[w{wid}] DONE pairs={n_pairs} pass={n_pass} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return wid, n_pairs, n_pass, time.time() - t0



def merge_and_write(parts, thr, data_dir, out_path):
    t0 = time.time()
    matches = {}
    n = 0
    for p in parts:
        with open(p, encoding="utf-8") as f:
            for line in f:
                s1, cid, sc = line.rstrip("\r\n").split("\t")
                sc = float(sc)
                if sc >= thr:
                    matches.setdefault(s1, []).append((sc, cid))
                    n += 1
    print(f"[merge] loaded {n} passing pairs over {len(matches)} S1 "
          f"({time.time() - t0:.0f}s)", flush=True)
    order = []
    p1 = os.path.join(data_dir, "test", "test_source1.tsv")
    for chunk in pd.read_csv(p1, sep="\t", dtype=str, keep_default_na=False,
                             usecols=["entity_id"], chunksize=200000):
        order.extend(chunk["entity_id"].tolist())
    print(f"[merge] s1 order rows={len(order)} ({time.time() - t0:.0f}s)",
          flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    nonempty = 0
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in order:
            ms = matches.get(s1)
            if not ms:
                f.write(s1 + "\t\n")
                continue
            ms.sort(key=lambda x: -x[0])
            seen, ded = set(), []
            for _, cid in ms:
                if cid not in seen:
                    seen.add(cid)
                    ded.append(cid)
            f.write(s1 + "\t" + ",".join(ded) + "\n")
            nonempty += 1
    size = os.path.getsize(out_path)
    print(f"[merge] wrote {out_path} rows={len(order)} nonempty={nonempty} "
          f"size={size / 1e6:.1f} MB ({time.time() - t0:.0f}s)", flush=True)
    if size > 512 * 1024 * 1024:
        raise SystemExit(f"ERROR: {out_path} is {size} bytes > 512 MiB limit")
    return size


def self_test(n):
    """Assert inlined features are bit-identical to features_a.pair_features."""
    from build_matrix import load_lookup
    need_s1 = set()
    pairs = []
    with open(DEF_CAND, "rb") as f:
        f.readline()
        while len(pairs) < n:
            line = f.readline()
            if not line:
                break
            j = line.find(b"\t")
            s1 = line[:j].decode("utf-8")
            cs = line[j + 1:].rstrip(b"\r\n").decode("utf-8")
            if not cs:
                continue
            for cid in cs.split(",")[:3]:
                pairs.append((s1, cid))
                need_s1.add(s1)
                if len(pairs) >= n:
                    break
    tsub = os.path.join(DEF_DATA, "test")
    raw1 = load_lookup(os.path.join(tsub, "test_source1.tsv"), need_s1)
    need_c = {c for _, c in pairs}
    raw2 = load_lookup(os.path.join(tsub, "test_source2.tsv"),
                       {x for x in need_c if x[:3] == "S2-"})
    raw3 = load_lookup(os.path.join(tsub, "test_source3.tsv"),
                       {x for x in need_c if x[:3] == "S3-"})
    raw = {**raw2, **raw3}
    max_diff = 0.0
    for s1, cid in pairs:
        r1, r2 = raw1[s1], raw[cid]
        nn1 = norm_name(r1[0]); na1 = norm_address(r1[1])
        pin1 = r1[3]; cc1 = r1[2]
        nn2 = norm_name(r2[0]); na2 = norm_address(r2[1])
        pin2 = r2[3]; cc2 = r2[2]
        tn1 = set(nn1.split()); ta1 = set(na1.split())
        n = s_metrics(nn1, nn2); a = s_metrics(na1, na2)
        pin_both = bool(pin1 and pin2)
        pin_match = bool(pin_both and pin1 == pin2)
        tn2 = set(nn2.split()); ta2 = set(na2.split())
        fast = np.array([
            n["ratio"], n["sort"], n["sett"], n["jw"], n["jac"],
            a["ratio"], a["sort"], a["sett"], a["jw"], a["jac"],
            abs(len(nn1) - len(nn2)), abs(len(na1) - len(na2)),
            float(bool(nn1) and nn1 == nn2),
            float(bool(na1) and na1 == na2),
            float(pin_match), float(pin_both),
            float((cc1 or "") == (cc2 or "")),
            float(cid[:3] == "S2-"),
            (n["sort"] / 100.0) * (a["sort"] / 100.0),
            len(tn1 & tn2) / max(len(tn1 | tn2), 1),
            len(ta1 & ta2) / max(len(ta1 | ta2), 1),
            float((not na1) or (not na2)),
        ], dtype=np.float64)
        ref = np.array(pair_features(
            r1[0], r1[1], r1[2], r1[3], r2[0], r2[1], r2[2], r2[3],
            cid[:3] == "S2-"), dtype=np.float64)
        max_diff = max(max_diff, float(np.abs(fast - ref).max()))
    print(f"self-test pairs={len(pairs)} feat_dim={len(FEAT_COLS)} "
          f"max_abs_diff={max_diff}")
    assert max_diff == 0.0, "feature parity check FAILED"
    print("self-test PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEF_DATA)
    ap.add_argument("--candidates", default=DEF_CAND)
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--threshold", type=float, default=0.995)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--batch", type=int, default=200000)
    ap.add_argument("--shards", type=int, default=0,
                    help="byte shards (default = workers)")
    ap.add_argument("--self-test", type=int, default=0,
                    help="check feature parity on N pairs and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test(args.self_test)
        return

    t0 = time.time()
    n_shards = args.shards or args.workers
    offs = shard_offsets(args.candidates, n_shards)
    print(f"[fast] shards={len(offs) - 1} offsets={offs}", flush=True)
    os.makedirs(PARTS_DIR, exist_ok=True)
    tasks = []
    for i in range(len(offs) - 1):
        part = os.path.join(PARTS_DIR, f"part_{i:02d}.tsv")
        tasks.append((i, offs[i], offs[i + 1], args.candidates, args.data,
                      args.model, args.threshold, part, args.batch))

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        results = pool.map(run_shard, tasks)

    total_pairs = total_pass = 0
    for wid, np_, npr, secs in sorted(results):
        total_pairs += np_
        total_pass += npr
        print(f"[fast] shard {wid}: pairs={np_} pass={npr} {secs:.0f}s",
              flush=True)
    print(f"[fast] TOTAL pairs={total_pairs} pass={total_pass} "
          f"scoring={time.time() - t0:.0f}s", flush=True)

    parts = [os.path.join(PARTS_DIR, f"part_{i:02d}.tsv")
             for i in range(len(offs) - 1)]
    merge_and_write(parts, args.threshold, args.data, args.out)
    for p in parts:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(PARTS_DIR)
    except OSError:
        pass
    print(f"[fast] finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

