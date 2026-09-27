"""Memory-bounded replica of utils/validate_submission.py checks.

The official validator keeps two full {s1: set(ids)} maps in RAM (the
candidate map alone is ~120M ids), which OOMs this 16GB box. This walks the
same files streaming and applies the identical rules:
  1. header exactly ['source1_entity_id', <col>]
  2. one row per required S1, no extra rows
  3. no duplicate IDs inside a list
  4. only S2-/S3- prefixed IDs (no self-matches, no foreign prefixes)
  5. matched_entity_ids subset of candidate_entity_ids
(valid_ids membership is checked separately/optionally since the official
run has that switch OFF by default.)
"""
import sys

DELIM = "\t"
MATCH = "output/matching_results.tsv"
CAND = "output/candidate_pairs.tsv"
S1 = "student_resource/dataset/test/test_source1.tsv"
S2 = "student_resource/dataset/test/test_source2.tsv"
S3 = "student_resource/dataset/test/test_source3.tsv"


def iter_rows(path):
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        yield ("header", header)
        for line in f:
            s1, tab, rest = line.partition(DELIM)
            yield ("row", s1, rest.rstrip("\n"))


def check_file(path, col, required, errors):
    gen = iter_rows(path)
    kind, header = next(gen)
    cols = [c.strip().lower() for c in header.rstrip("\n").split(DELIM)]
    if cols != ["source1_entity_id", col]:
        errors.append("%s: bad header %s" % (path, cols))
        return
    n_rows = empties = 0
    dup_in_list = self_m = bad_prefix = 0
    extra = missing = 0
    for rec in gen:
        n_rows += 1
        s1, rest = rec[1], rec[2]
        if s1 not in required:
            extra += 1
        else:
            required.discard(s1)
        ids = rest.split(",") if rest.strip() else []
        if not ids:
            empties += 1
            continue
        if len(ids) != len(set(ids)):
            dup_in_list += 1
        for mid in set(ids):
            if mid.startswith("S1-"):
                self_m += 1
            elif not mid.startswith(("S2-", "S3-")):
                bad_prefix += 1
    if dup_in_list:
        errors.append("%s: %d row(s) with duplicate ids in list" % (path, dup_in_list))
    if self_m:
        errors.append("%s: %d self-match S1- ids" % (path, self_m))
    if bad_prefix:
        errors.append("%s: %d ids without S2-/S3- prefix" % (path, bad_prefix))
    if missing:
        pass
    if extra:
        errors.append("%s: %d row(s) not in test set" % (path, extra))
    print("  %s: %d rows (%d empty, %d non-empty)"
          % (path, n_rows, empties, n_rows - empties), flush=True)
    return empties


def main():
    errors = []
    required = set()
    with open(S1, encoding="utf-8") as f:
        next(f)
        for line in f:
            required.add(line.partition(DELIM)[0])
    print("  required S1 entities: %d" % len(required), flush=True)
    rem_m = set(required)
    rem_c = set(required)
    check_file(MATCH, "matched_entity_ids", rem_m, errors)
    check_file(CAND, "candidate_entity_ids", rem_c, errors)
    for nm, rem in (("matching_results.tsv", rem_m),
                    ("candidate_pairs.tsv", rem_c)):
        if rem:
            errors.append("%s: %d required S1 missing" % (nm, len(rem)))

    # Subset check: every (s1, matched_id) must also appear in the candidate
    # file. The two files are NOT in the same row order (the official
    # validator matches by id), so build the matched-pair set (only ~5M
    # pairs, small) and drain it while streaming the candidate file.
    need = set()
    with open(MATCH, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.partition(DELIM)
            if rest.strip():
                for mid in rest.rstrip("\n").split(","):
                    need.add(s1 + "\t" + mid)
    n_need = len(need)
    dup_m = 0
    with open(MATCH, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.partition(DELIM)
            if rest.strip():
                ids = rest.rstrip("\n").split(",")
                if len(ids) != len(set(ids)):
                    dup_m += 1
    dup_c = 0
    with open(CAND, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.partition(DELIM)
            if not rest.strip():
                continue
            ids = rest.rstrip("\n").split(",")
            if len(ids) != len(set(ids)):
                dup_c += 1
            for mid in ids:
                need.discard(s1 + "\t" + mid)
    if need:
        errors.append("matched_entity_ids not in candidate file for %d "
                      "pair(s), e.g. %s"
                      % (len(need), list(need)[:3]))
    if dup_m or dup_c:
        errors.append("duplicate ids inside list: %d matching, %d candidate "
                      "row(s)" % (dup_m, dup_c))
    print("  subset check: %d matched pairs, %d missing from candidates"
          % (n_need, len(need)), flush=True)

    print()
    if errors:
        print("FAIL - %d issue(s):" % len(errors))
        for e in errors:
            print("   " + e)
    else:
        print("PASS - all checks clean.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
