"""Step 3a: pair feature engineering (chunked, low-memory).

Features per (S1, candidate) pair — all country-agnostic (open string set):
 name: rapidfuzz ratio/token_sort/token_set, Jaro-Winkler (jellyfish),
       Jaccard, TF-IDF-cosine recomputed cheaply via token overlap fallback,
       len-diff, abbrev-normalized exact, prefix overlap
 addr: same string metrics on normalized addresses + token overlap +
       PIN exact bool + PIN-present-both flag
 cross: country exact bool, candidate source (S2/S3), name x addr interaction
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from utils import norm_name, norm_address, extract_pin, tokens


def s_metrics(a, b):
    if not a and not b:
        return dict(ratio=100.0, sort=100.0, sett=100.0, jw=1.0, jac=1.0)
    if not a or not b:
        return dict(ratio=0.0, sort=0.0, sett=0.0, jw=0.0, jac=0.0)
    try:
        jw = float(JaroWinkler.normalized_similarity(a, b))
    except Exception:
        jw = 0.0
    sa, sb = set(a.split()), set(b.split())
    jac = len(sa & sb) / max(len(sa | sb), 1)
    return dict(ratio=float(fuzz.ratio(a, b)), sort=float(fuzz.token_sort_ratio(a, b)),
                sett=float(fuzz.token_set_ratio(a, b)), jw=jw, jac=float(jac))


FEAT_COLS = [
    "n_ratio", "n_sort", "n_set", "n_jw", "n_jac",
    "a_ratio", "a_sort", "a_set", "a_jw", "a_jac",
    "n_len_diff", "a_len_diff", "n_exact", "a_exact",
    "pin_match", "pin_both", "country_match", "is_s2",
    "n_x_a", "n_tok_overlap", "a_tok_overlap", "addr_empty_either",
]


def pair_features(s1_name, s1_addr, s1_country, s1_pin,
                  c_name, c_addr, c_country, c_pin, is_s2):
    nn1, nn2 = norm_name(s1_name), norm_name(c_name)
    na1, na2 = norm_address(s1_addr), norm_address(c_addr)
    n, a = s_metrics(nn1, nn2), s_metrics(na1, na2)
    tn1, tn2 = set(tokens(nn1)), set(tokens(nn2))
    ta1, ta2 = set(tokens(na1)), set(tokens(na2))
    pin_both = bool(s1_pin and c_pin)
    pin_match = bool(pin_both and s1_pin == c_pin)
    nx = (n["sort"] / 100.0) * (a["sort"] / 100.0)
    return [
        n["ratio"], n["sort"], n["sett"], n["jw"], n["jac"],
        a["ratio"], a["sort"], a["sett"], a["jw"], a["jac"],
        abs(len(nn1) - len(nn2)), abs(len(na1) - len(na2)),
        float(bool(nn1) and nn1 == nn2), float(bool(na1) and na1 == na2),
        float(pin_match), float(pin_both),
        float((s1_country or "") == (c_country or "")),
        float(is_s2), nx,
        len(tn1 & tn2) / max(len(tn1 | tn2), 1),
        len(ta1 & ta2) / max(len(ta1 | ta2), 1),
        float((not na1) or (not na2)),
    ]
