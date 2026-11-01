"""Three-way verdict rules (same / variant / different), shared by src/variant.py, src/service.py
and src/live.py.

Kept free of faiss and torch imports so the live app can use it.
"""
import re

import numpy as np

COMBO = re.compile(r"\bcombo\b|\bkit\b|\bset of \d+|\bpack of \d+|\+|\b\d+\s*in\s*1\b", re.IGNORECASE)

# Words that name a different model of the same product line (Airdopes 141 vs 141 ANC / Neo / Pro).
VARIANT_WORDS = {"pro", "max", "plus", "lite", "mini", "ultra", "neo", "elite", "prime", "anc", "se", "fe",
                 "nano", "edge", "fold", "flip", "xl", "slim", "turbo", "gen", "nxt", "neckband"}
# Number + unit tokens are sizes or specs, not model codes (42hr, 32db, 5000mah, 128gb, 2nd).
SPEC = re.compile(r"^\d+(\.\d+)?(mah|ah|gb|tb|mb|w|kw|v|hr|hrs|h|db|ml|l|ltr|g|gm|kg|mm|cm|m|ft|inch|in|hz|khz|"
                  r"mp|k|x|pcs|pc|st|nd|rd|th|s|min|mins|yrs|yr|d)$")
NAMED_NUMBER = re.compile(r"\b([a-z]{3,})\s?(\d{2,4})\b(?![a-z])")


def combo_mismatch(title_a, title_b):
    """True when exactly one title describes a combo, kit, set/pack or N-in-1 product."""
    return bool(COMBO.search(title_a)) != bool(COMBO.search(title_b))


def _model_parts(title):
    t = title.lower()
    words = re.findall(r"[a-z0-9]+", t)
    variant = {w for w in words if w in VARIANT_WORDS}
    codes = {w for w in words if re.search(r"[a-z]", w) and re.search(r"\d", w) and len(w) >= 3 and not SPEC.match(w)}
    named = set(NAMED_NUMBER.findall(t))
    return variant, codes, named


def model_conflict(title_a, title_b):
    """True when the titles name different models: different model words (Pro, Neo, ANC, Gen),
    model codes that share nothing (BHD308 vs BHD356), or the same number after different
    names (Airdopes 141 vs Immortal 141)."""
    va, ca, na = _model_parts(title_a)
    vb, cb, nb = _model_parts(title_b)
    if va != vb:
        return True
    if ca and cb and not ca & cb:
        return True
    shared_numbers = {n for _, n in na} & {n for _, n in nb}
    return any(not ({w for w, x in na if x == n} & {w for w, x in nb if x == n}) for n in shared_numbers)


def classify(prob, conflict, colour_d, combo, cfg, other_model=None):
    """Vectorised verdict: 'same', 'variant' or 'different' for each pair.
    other_model (optional): pairs whose titles name different models are 'different'."""
    same_model = prob >= cfg["india_same_model_threshold"]
    if other_model is not None:
        same_model &= ~other_model.astype(bool)
    differs = conflict.astype(bool) | combo.astype(bool) | (colour_d > cfg["colour_cut"])
    return np.where(same_model & ~differs, "same", np.where(same_model, "variant", "different"))
