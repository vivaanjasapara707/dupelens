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
ACCESSORY_WORDS = {"cover", "covers", "case", "cases", "replacement", "compatible", "refill", "spare", "protector",
                   "skins", "decal", "decals", "strap", "straps", "pouch", "sticker", "stickers", "stopper", "cap"}
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


GENERIC = {"price", "buy", "online", "new", "latest", "best", "india", "in", "with", "and", "for", "the",
           "of", "phone", "smartphone", "mobile", "cheap", "deal", "offer", "lowest", "apple", "by"}


def _tokens(text):
    """Lower-case word tokens; '128 GB' -> '128gb' so sizes are one token."""
    t = re.sub(r"['’]", "", text.lower())                    # Levi's -> levis
    t = re.sub(r"\bplay\s?station\s?(\d)\b", r"ps\1", t)     # PlayStation 5 -> ps5
    t = re.sub(r"\bslim[\s-]fit\b", "slimfit", t)             # a jeans cut, not the model word "Slim"
    t = re.sub(r"\banalog(?:ue)?[\s-]digital\b", "anadigi", t)  # a watch type, not the model word "Digital"
    t = re.sub(r"(\d)\s+(gb|tb|ml|litres?|liters?|ltrs?|l|gms?|g|kg|watts?|w|mah|mm|cm|inch)\b", r"\1\2", t)
    t = re.sub(r"\b([a-z]+)-(\d[a-z0-9]*)\b", r"\1\2", t)   # a hyphenated model code is one token: WH-1000XM5
    t = re.sub(r"/\d+\b", "", t)                             # region suffix: BT3221/15 -> BT3221
    return re.findall(r"\d+(?:\.\d+)?[a-z]*|[a-z]+[a-z0-9]*", t)   # keeps 1.5litre and 18e together


# "2nd Gen" / "2. Gen" / "Gen 2" (a bare "141 Gen 2" is Gen 2, not Gen 141)
GEN = re.compile(r"\b(\d+)(?:st|nd|rd|th|\.)\s*gen(?:eration)?\b|\bgen(?:eration)?[.\s]*(\d+)\b")


def query_parts(query):
    """The words that identify the product in a search: brand / model words and model numbers. Sizes such as
    128GB are preferences (pre-selected in the variant filter), not identity."""
    # "3s" / "6s" are model numbers (MX Master 3S, iPhone 6s), not sizes
    toks = [w for w in _tokens(query) if w not in GENERIC and (not SPEC.match(w) or re.fullmatch(r"\d+s", w))
            and w not in ("5g", "4g")]
    return [w for w in toks if any(c.isdigit() for c in w)], [w for w in toks if w.isalpha()]


def query_match(query, title):
    """How a listing title relates to what the user searched for:
      'exact'     names the searched product: every model number of the search, (nearly) every model word,
                  and no extra model word the search did not ask for (Pro, Max, Plus, Mini, Ultra, Lite, Neo,
                  FE, ...) and no letter-suffixed number (18e when the search says 18);
      'accessory' an accessory or spare part for it (case, cover, "... for iPhone 18");
      'related'   another model of the same brand or line (iPhone 18 Pro for "iPhone 18");
      'other'     anything else.
    """
    nums, words = query_parts(query)
    tt = _tokens(title)
    ts = set(tt)
    if not nums and not words:
        return "other"
    gens = {a or b for a, b in GEN.findall(title.lower())}
    if gens:
        if not gens & set(nums) and "gen" not in words:
            return "related"
        ts |= gens
        ts.discard("gen")
    compact = "".join(tt)
    # moisturising ~ moisturizing; "beard" inside "Beardtrimmer"
    word_hit = lambda w: w in ts or (len(w) >= 4 and w in compact) or any(len(w) >= 5 and t[:6] == w[:6] for t in ts)
    have_nums = all(n in ts for n in nums)                     # model NUMBERS / codes must all match
    hits = sum(word_hit(w) for w in words)
    if nums:   # a model code identifies the product; the other words only need to half-match
        need = max(1, int(np.ceil(0.5 * len(words))))
    else:      # no model number: the words are the identity (Dettol Original vs Dettol Skincare)
        need = len(words) if len(words) <= 4 else int(np.ceil(0.8 * len(words)))
    related = (bool(words) and word_hit(words[0])) or (bool(nums) and any(n in ts for n in nums))
    # accessory: accessory words, or the searched model named only after "for" ("Flipkart SmartBuy for iPhone 16")
    before_for = tt[:tt.index("for")] if "for" in tt else tt
    if is_accessory(title) or ("for" in tt and nums and not any(n in before_for for n in nums)):
        return "accessory" if related else "other"
    # model words the user asked for (Pro, Max, Plus, ...) must be there: "iPhone 16" is not "iPhone 16 Pro"
    asked_variant = set(words) & VARIANT_WORDS
    # the name in front of each model number must be there too: "Airdopes 141" is not "Immortal 141"
    qt = [w for w in _tokens(query) if w not in GENERIC]
    anchors = {qt[i - 1] for i, w in enumerate(qt) if w in nums and i > 0 and qt[i - 1].isalpha()}
    # the first word of the search (usually the brand) must be there: "Dssb Buds 3 Pro" is not "Samsung Galaxy Buds 3 Pro"
    lead = qt[0] if qt and qt[0] in words else None
    if not (have_nums and hits >= need and asked_variant <= ts and all(word_hit(a) for a in anchors)
            and (lead is None or word_hit(lead))):
        return "related" if related else "other"
    extra_variant = (ts & VARIANT_WORDS) - set(words)
    suffixed = any(re.fullmatch(r"(\d+)[a-z]{1,2}", t) and re.fullmatch(r"(\d+)[a-z]{1,2}", t).group(1) in nums
                   and t not in nums and not SPEC.match(t) and t not in ("5g", "4g") for t in ts)
    if extra_variant or suffixed:
        return "related"
    return "exact"


def is_accessory(title):
    """True when the listing is an accessory or spare part (case, cover, replacement ...), not the product."""
    # words after "with" describe what comes in the box: "AirPods Pro with MagSafe Case" is the earbuds
    head = re.split(r"\bwith\b", title.lower())[0]
    return bool(set(re.findall(r"[a-z]+", head)) & ACCESSORY_WORDS) or bool(ACCESSORY_PHRASES.search(head))


# phone skins and spare screens written as phrases, and the same words in Hindi titles (case, cover, replacement, strap)
ACCESSORY_PHRASES = re.compile(r"\b(?:mobile|phone|back|vinyl|laptop)\s+skins?\b|\blcd\s+(?:screen|display)\b|"
                               r"\bdisplay\s+combo\b|\btempered\s+glass\b|\bscreen\s+(?:guard|protector)\b|"
                               r"केस|कवर|रिप्लेसमेंट|स्ट्रैप")


def lead_word(query):
    """The first word of a search when it is a name word (usually the brand): 'samsung' in 'samsung galaxy buds 3'."""
    qt = [w for w in _tokens(query) if w not in GENERIC]
    nums, words = query_parts(query)
    return qt[0] if qt and qt[0] in words else None


def _model_parts(title):
    t = title.lower()
    words = re.findall(r"[a-z0-9]+", t)
    variant = {w for w in words if w in VARIANT_WORDS}
    accessory = bool({w for w in words if w in ACCESSORY_WORDS})
    codes = {w for w in words if re.search(r"[a-z]", w) and re.search(r"\d", w) and len(w) >= 2 and not SPEC.match(w)}
    # a hyphenated code is one code: "WH-1000XM5" -> wh1000xm5 (its pieces wh / 1000xm5 alone would match WF-1000XM5)
    for joined in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", t):
        flat = joined.replace("-", "")
        if re.search(r"[a-z]", flat) and re.search(r"\d", flat) and not SPEC.match(flat):
            codes -= set(joined.split("-"))
            codes.add(flat)
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
    # a model number with a letter suffix is another model: "iPhone 16e" vs "iPhone 16", "Redmi 13C" vs "Redmi 13"
    suffixed = lambda codes: {re.match(r"\d+", c).group(0) for c in codes if re.fullmatch(r"\d+[a-z]{1,2}", c)}
    plain = lambda words: {w for w in words if w.isdigit()}
    if suffixed(ca) & plain(wb) - suffixed(cb) or suffixed(cb) & plain(wa) - suffixed(ca):
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


def name_levels(titles, max_share=0.3, min_offers=8):
    """How different two titles' NAME words are (one search's offers at a time): 0 none, 1 variant, 2 other model.

    A name word is one that fewer than max_share of the offers in this search use, so it names a
    model, line, shade or scent (Matchless vs Xeno, Plum Pick vs Red Coat), while words most offers
    share (the brand, "mixer", "grinder") are ignored.
      2 (other model): both titles carry name words, share none, and those words are not only colours.
      1 (variant):     only colour words differ (Cosmic Orange vs Deep Blue), or one title is vague and the
                       other names its item: the same model, but not confirmed as the identical item.
    Needs enough offers to know which words are rare. Returns a function (i, j) -> 0, 1 or 2.
    """
    docs = [{w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in FILLER} for t in titles]
    n = len(titles)
    if n < min_offers:
        return lambda i, j: 0
    df = {}
    for d in docs:
        for w in d:
            df[w] = df.get(w, 0) + 1
    names = [{w for w in d if df[w] / n < max_share} for d in docs]

    def level(i, j):
        a, b = names[i], names[j]
        if not a and not b or a & b:
            return 0
        if bool(a) != bool(b) or (a | b) <= COLOURS:
            return 1
        return 2
    return level


def storage_unconfirmed(title_a, title_b):
    """True when only one title states a storage size (256 GB): the identical item cannot be confirmed."""
    has = lambda t: bool(re.search(r"\d+\s?(?:gb|tb)\b(?!\s*ram)", t.lower()))
    return has(title_a) != has(title_b)


COLOURS = {"black", "white", "silver", "grey", "gray", "gold", "golden", "rose", "pink", "red", "blue", "navy", "green", "olive",
           "yellow", "orange", "purple", "violet", "lavender", "brown", "beige", "cream", "peach", "teal", "turquoise",
           "maroon", "titanium", "graphite", "midnight", "starlight", "champagne", "copper", "bronze", "mint", "coral",
           "multicolor", "multicolour", "transparent"}
BUNDLE = re.compile(r"\bbundle\b|\bwith free\b|\bfree gift\b|\bcombo\b|\b(?:pack|set) of \d+|\bbuy \d+ get \d+", re.IGNORECASE)


def colours(title):
    return set(re.findall(r"[a-z]+", title.lower())) & COLOURS


def colour_conflict(title_a, title_b):
    """True when both titles name a colour and they share none (Black vs Silver)."""
    ca, cb = colours(title_a), colours(title_b)
    return bool(ca and cb and not ca & cb)


def network_free(text):
    """'4G' / '5G' are phone networks, not grams: drop them before reading sizes."""
    return re.sub(r"\b[2-5]g\b", " ", text.lower())


def _sizes(title):
    out = {}
    for value, unit in SIZE.findall(network_free(title)):
        kind, factor = SIZE_UNIT[unit]
        out.setdefault(kind, set()).add(round(float(value) * factor, 3))
    return out


def pack_mismatch(title_a, title_b):
    """True when one title is a bundle/combo/multi-pack and the other is not, including combos written as
    several sizes in one title ("Popular 2 L 3 L" vs "Popular 3 L")."""
    def is_pack(t):
        return bool(BUNDLE.search(t)) or any(len(v) >= 2 for k, v in _sizes(t).items() if k != "power")
    return is_pack(title_a) != is_pack(title_b)


def size_conflict(title_a, title_b):
    """True when both titles give a power, volume or mass and no value matches (1600 W vs 1200 Watt)."""
    sa, sb = _sizes(title_a), _sizes(title_b)
    return any(not sa[k] & sb[k] for k in sa.keys() & sb.keys())


def classify(prob, conflict, colour_d, combo, cfg, other_model=None):
    """Vectorised verdict: 'same', 'variant' or 'different' for each pair.
    other_model (optional): pairs whose titles name different models are 'different'."""
    same_model = prob >= cfg["india_same_model_threshold"]
    if other_model is not None:
        same_model &= ~other_model.astype(bool)
    differs = conflict.astype(bool) | combo.astype(bool) | (colour_d > cfg["colour_cut"])
    return np.where(same_model & ~differs, "same", np.where(same_model, "variant", "different"))
