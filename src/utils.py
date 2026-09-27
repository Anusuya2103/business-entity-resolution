"""Shared normalization / IO utilities (stdlib + pandas only)."""
import re
import unicodedata

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llc", "ltd", "limited", "plc", "pllc",
    "pvt", "private", "pvt ltd", "private limited",
    "llp", "lp", "llp.", "gmbh", "sarl", "sas", "sasu", "sci",
    "pty", "holdings", "group", "enterprises", "enterprise",
    "services", "solutions", "associates", "assoc", "intl", "international",
    "dba",
}

ADDR_ABBR = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "pl": "place", "pkwy": "parkway", "hwy": "highway", "ste": "suite",
    "apt": "apartment", "fl": "floor", "bldg": "building", "no": "",
    "opp": "opposite", "nr": "near", "n": "north", "s": "south",
    "e": "east", "w": "west",
}

NAME_ABBR = {
    "&": "and", "@": "at",
    "corp": "corporation", "inc": "incorporated", "co": "company",
    "intl": "international", "assoc": "associates", "dept": "department",
    "pvt": "private", "ltd": "limited", "mfg": "manufacturing",
    "pharma": "pharmaceuticals", "tech": "technologies",
    "info": "information", "mgmt": "management", "svc": "services",
    "ctr": "center", "hosp": "hospital", "univ": "university",
}

_PIN_RE = re.compile(r"\b(\d{5,6})\b")
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


def ascii_fold(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    return s.encode("ascii", "ignore").decode("ascii")


def clean_text(s) -> str:
    if s is None:
        return ""
    s = str(s)
    s = ascii_fold(s).lower()
    s = s.replace("&", " and ").replace("@", " at ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def norm_name(s, strip_legal=True, expand_abbr=True, sort_tokens=False) -> str:
    """Normalized business name used for blocking keys + features."""
    t = clean_text(s)
    if not t:
        return ""
    toks = t.split()
    if expand_abbr:
        toks = [NAME_ABBR.get(w, w) for w in toks]
    if strip_legal:
        toks = [w for w in toks if w not in LEGAL_SUFFIXES]
    if sort_tokens:
        toks = sorted(toks)
    return " ".join(toks)


def norm_address(s, expand_abbr=True) -> str:
    t = clean_text(s)
    if not t:
        return ""
    toks = t.split()
    if expand_abbr:
        out = []
        for w in toks:
            rep = ADDR_ABBR.get(w, w)
            if rep:
                out.append(rep)
        toks = out
    return " ".join(toks)


def extract_pin(s) -> str:
    """Generic numeric postal-code extractor (4-6 digit token), country-agnostic."""
    if s is None:
        return ""
    m = _PIN_RE.findall(ascii_fold(str(s)))
    return m[-1] if m else ""


def tokens(s: str):
    return s.split() if s else []


def read_tsv(path, usecols=None, chunksize=None, **kw):
    import pandas as pd
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       usecols=usecols, chunksize=chunksize, **kw)


def parse_id_list(s) -> list:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]
