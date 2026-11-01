"""Three-way verdict rules (same / variant / different), shared by src/variant.py, src/service.py
and src/live.py.

Kept free of faiss and torch imports so the live app can use it.
"""
import re

import numpy as np

COMBO = re.compile(r"\bcombo\b|\bkit\b|\bset of \d+|\bpack of \d+|\+|\b\d+\s*in\s*1\b", re.IGNORECASE)

# Words that name a different model, formulation or condition of the same product line
# (Airdopes 141 vs 141 ANC / Neo / Pro; cream vs Daily UV SPF cream; new vs refurbished phone).
VARIANT_WORDS = {"pro", "max", "plus", "lite", "mini", "ultra", "neo", "elite", "prime", "anc", "se", "fe",
                 "nano", "edge", "fold", "flip", "xl", "slim", "turbo", "gen", "nxt", "neckband", "uv", "spf",
                 "digital", "refurbished", "renewed", "unboxed", "used", "preowned"}
# A listing for an accessory or spare part is never the product itself.
ACCESSORY_WORDS = {"cover", "case", "replacement", "compatible", "refill", "spare", "protector", "skin",
                   "strap", "pouch", "sticker", "stopper", "cap"}
UNIT = r"(?:mah|ah|gb|tb|mb|kw|w|watts?|v|hrs?|hours?|db|ml|ltrs?|litres?|liters?|l|g|gms?|grams?|kg|mm|cm|m|ft|inch|in|hz|mp|pcs|pc|x)"
# Number + unit tokens are sizes or specs, not model codes (42hr, 32db, 5000mah, 128gb, 2nd).
SPEC = re.compile(rf"^\d+(\.\d+)?({UNIT}|k|st|nd|rd|th|s|min|mins|yrs|yr|d)$")
# "<name> <number>" model names: Airdopes 141, PIC 20, PIC 3.1 (not "200 ml", not "pack of 2").
NAMED_NUMBER = re.compile(rf"\b([a-z]{{2,}})\s?(\d+(?:\.\d+)?)\b(?![a-z])(?!\s?{UNIT}\b)")
STOP = {"of", "for", "with", "and", "the", "upto", "up", "to", "in", "by", "set", "pack", "x", "only", "size"}
# Sizes the trained number-token feature misses (watt, liters, gm, jars); compared only inside the verdict rules.
SIZE = re.compile(r"(\d+(?:\.\d+)?)\s?(kw|watts?|w|ml|ltrs?|litres?|liters?|l|kg|gms?|grams?|g|jars?)\b")
SIZE_UNIT = {"kw": ("power", 1000), "watt": ("power", 1), "watts": ("power", 1), "w": ("power", 1),
             "ml": ("volume", 1), "l": ("volume", 1000), "ltr": ("volume", 1000), "ltrs": ("volume", 1000),
             "litre": ("volume", 1000), "litres": ("volume", 1000), "liter": ("volume", 1000), "liters": ("volume", 1000),
             "kg": ("mass", 1000), "g": ("mass", 1), "gm": ("mass", 1), "gms": ("mass", 1), "gram": ("mass", 1),
             "grams": ("mass", 1), "jar": ("jars", 1), "jars": ("jars", 1)}


def combo_mismatch(title_a, title_b):
    """True when exactly one title describes a combo, kit, set/pack or N-in-1 product."""
    return bool(COMBO.search(title_a)) != bool(COMBO.search(title_b))


def _model_parts(title):
    t = title.lower()
    words = re.findall(r"[a-z0-9]+", t)
    variant = {w for w in words if w in VARIANT_WORDS}
    accessory = bool({w for w in words if w in ACCESSORY_WORDS})
    codes = {w for w in words if re.search(r"[a-z]", w) and re.search(r"\d", w) and len(w) >= 2 and not SPEC.match(w)}
    named = {(w, float(n)) for w, n in NAMED_NUMBER.findall(t) if w not in STOP}
    # a glued code reads like a spaced one: "DX2" = "DX 2", so "DX 6" vs "DX2" is a conflict
    named |= {(m.group(1), float(m.group(2))) for c in codes for m in [re.fullmatch(r"([a-z]{2,})(\d+)", c)] if m}
    return variant, accessory, codes, named, set(words)


def model_conflict(title_a, title_b):
    """True when the titles name different models or one is only an accessory:
    - different model/formulation/condition words (Pro, Neo, ANC, Gen, UV, refurbished)
    - exactly one title is an accessory or spare part (cover, replacement, compatible)
    - only one title names a model at all (JBL Tune 510BT vs "Headphone")
    - model codes that share nothing (BHD308 vs BHD356, V2 vs V3)
    - the same name with different numbers (PIC 20 vs PIC 3.1)
    - the same number after different names that the other title never mentions (Airdopes 141 vs Immortal 141)
    """
    va, xa, ca, na, wa = _model_parts(title_a)
    vb, xb, cb, nb, wb = _model_parts(title_b)
    if va != vb or xa != xb:
        return True
    if bool(ca or na) != bool(cb or nb):  # only one title names a model: cannot confirm it is the same one
        return True
    if ca and cb and not ca & cb:
        return True
    nums_a, nums_b = {n for _, n in na}, {n for _, n in nb}
    for word in {w for w, _ in na} & {w for w, _ in nb}:
        if not {n for w, n in na if w == word} & {n for w, n in nb if w == word} and not nums_a & nums_b:
            return True
    for num in nums_a & nums_b:
        names_a, names_b = {w for w, n in na if n == num}, {w for w, n in nb if n == num}
        if not names_a & names_b and not names_a & wb and not names_b & wa:
            return True
    return False


FILLER = {"with", "and", "for", "the", "buy", "new", "free", "online", "best", "offer", "pack", "set", "each", "only",
          "from", "upto", "get", "more", "size", "combo", "pcs", "piece", "pieces", "india", "indian", "genuine"}


def name_conflicts(titles, max_share=0.3, min_offers=8):
    """Pairs whose titles use different NAME words (one search's offers at a time).

    A name word is one that fewer than max_share of the offers in this search use, so it names a
    model, line, shade or scent (Matchless vs Xeno, Plum Pick vs Red Coat), while words most offers
    share (the brand, "mixer", "grinder") are ignored. Two titles that both carry name words but share
    none of them describe different items, and so do a vague title and a named one. Needs enough
    offers to know which words are rare.
    Returns a function (i, j) -> bool.
    """
    docs = [{w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in FILLER} for t in titles]
    n = len(titles)
    if n < min_offers:
        return lambda i, j: False
    df = {}
    for d in docs:
        for w in d:
            df[w] = df.get(w, 0) + 1
    names = [{w for w in d if df[w] / n < max_share} for d in docs]
    # also when only one title names its item: a vague title cannot confirm it is the same one
    return lambda i, j: bool(names[i]) != bool(names[j]) or (bool(names[i]) and not names[i] & names[j])


def size_conflict(title_a, title_b):
    """True when both titles give a power, volume or mass and no value matches (1600 W vs 1200 Watt)."""
    def sizes(t):
        out = {}
        for value, unit in SIZE.findall(t.lower()):
            kind, factor = SIZE_UNIT[unit]
            out.setdefault(kind, set()).add(round(float(value) * factor, 3))
        return out
    sa, sb = sizes(title_a), sizes(title_b)
    return any(not sa[k] & sb[k] for k in sa.keys() & sb.keys())


def classify(prob, conflict, colour_d, combo, cfg, other_model=None):
    """Vectorised verdict: 'same', 'variant' or 'different' for each pair.
    other_model (optional): pairs whose titles name different models are 'different'."""
    same_model = prob >= cfg["india_same_model_threshold"]
    if other_model is not None:
        same_model &= ~other_model.astype(bool)
    differs = conflict.astype(bool) | combo.astype(bool) | (colour_d > cfg["colour_cut"])
    return np.where(same_model & ~differs, "same", np.where(same_model, "variant", "different"))
