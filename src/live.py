"""Live search: give DupeLens a product name, a product link or a photo; it finds every online
store selling that exact item and where it is best to buy.

1. Understand the input: a name is searched as is; a store link gives the name from the link;
   a photo goes to Google Lens, which names the product.
2. Ask Serper (Google Shopping results for India) for up to 40 offers.
3. Download each offer's photo and compute the signals the model was trained on
   (CLIP, MPNet, phash, DINOv2, colour), exactly as src/service.py does for one listing.
4. Score every pair of offers with the trained M1_v2 model and the same/variant/different
   rules (+ the model-word rule in src/verdict.py).
5. Group offers that are the exact same item; the page then ranks the sellers.
Price is only used AFTER matching, never as a model feature.

Needs a free Serper key (serper.dev) in the environment variable SERPER_API_KEY or in a
file .env in the repo root (line: SERPER_API_KEY=...). Every Serper answer is saved in
cache/live/, so repeating a search costs no credit and works offline.
Run once from the command line: python -m src.live "boat airdopes 141"
"""
import functools
import hashlib
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus, unquote, urlparse

import numpy as np
import pandas as pd
import requests

from src import config
from src.extra import colour_dist
from src.numtokens import compare, extract
from src.verdict import (classify, colour_conflict, colours, combo_mismatch, is_accessory, model_conflict,
                         lead_word, name_levels, pack_mismatch, query_match, size_conflict, storage_unconfirmed)

SERPER = "https://google.serper.dev/"
LIVE_CACHE = config.CACHE_DIR / "live"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}
NOT_SHOPS = {"youtube", "tiktok", "instagram", "pinterest", "facebook", "reddit", "x", "twitter", "quora", "wikipedia"}


def api_key():
    """SERPER_API_KEY from the environment, else from the .env file in the repo root."""
    key = config.env("SERPER_API_KEY")
    if not key:
        raise RuntimeError("No Serper key: put SERPER_API_KEY=... in .env (free at serper.dev)")
    return key


def serper(endpoint, payload, tag=None):
    """One Serper call (shopping, lens or search); answers cached on disk by request.
    tag: a separate cache slot, e.g. the date for the daily price tracker (a fresh answer once per day)."""
    LIVE_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{endpoint}|{json.dumps(payload, sort_keys=True).lower()}{'|' + tag if tag else ''}".encode()).hexdigest()[:16]
    path = LIVE_CACHE / f"{endpoint}_{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    r = requests.post(SERPER + endpoint, headers={"X-API-KEY": api_key(), "Content-Type": "application/json"},
                      json=payload, timeout=40)
    r.raise_for_status()
    data = r.json()
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


# Well-known Indian stores: big retailers and marketplaces, quick-commerce apps, pharmacies.
# Matched at the start of a word in the store name ("Amazon.in", "JioMart Grocery", "Apollo247").
WELL_KNOWN = re.compile(r"\b(amazon|flipkart|shopsy|croma|reliance ?digital|jiomart|tata ?cliq|vijay ?sales|myntra|ajio|"
                        r"nykaa|purplle|tira\b|bigbasket|blinkit|zepto|instamart|swiggy|dmart|meesho|snapdeal|"
                        r"pharmeasy|apollo|1mg|netmeds|firstcry|pepperfry|decathlon|lenskart|ikea|samsung|apple|"
                        r"oneplus|mi\.com|xiaomi)", re.IGNORECASE)
NOT_BRAND = {"buy", "new", "original", "the", "combo", "set", "pack"}
# Retailers that sell their own stock (not marketplaces where outside sellers list): a low price there is a real
# deal, not a fake, so they are never called "suspiciously cheap". Amazon, Flipkart, Meesho, Snapdeal are marketplaces.
FIRST_PARTY = re.compile(r"croma|reliance|jiomart|vijay ?sales|bigbasket|blinkit|zepto|instamart|swiggy|dmart|"
                         r"tata ?cliq|nykaa|purplle|apollo|1mg|netmeds|pharmeasy|decathlon|ikea", re.IGNORECASE)


def well_known(store, title):
    """True for a well-known store, or the brand's own store ("boAt" or "Wildcraft Official" selling that brand)."""
    store = store or ""
    if WELL_KNOWN.search(store):
        return True
    words = [w for w in re.findall(r"[a-z0-9]+", (title or "").lower()) if w not in NOT_BRAND]
    brand = words[0] if words else ""
    if len(brand) > 2 and brand in re.sub(r"[^a-z0-9]", "", store.lower()):
        return True
    first = store.lower().split()[0] if store.split() else ""  # "Wildcraft Official" selling "Cargo Pack ... Wildcraft"
    return "official" in store.lower() and len(first) > 2 and first in words


def attributes(title):
    """Variant attributes written in a title: storage, RAM, colour, size and pack count (shown as pills, and used to match the details typed in a search)."""
    from src.verdict import SIZE, SIZE_UNIT, network_free
    t, out = title.lower(), {}
    storage = []
    for value, unit, ram in re.findall(r"(\d+(?:\.\d+)?)\s?(gb|tb)\b(\s*(?:ram|ddr\w*))?", t):
        gb = float(value) * (1024 if unit == "tb" else 1)
        if ram:
            out["ram"] = f"{value} GB RAM"
        else:
            storage.append(gb)
    if storage:
        gb = max(storage)
        out["storage"] = f"{gb / 1024:g} TB" if gb >= 1024 else f"{gb:g} GB"
    c = colours(title)
    if c:
        out["colour"] = " / ".join(sorted(w.capitalize() for w in c))
    for value, unit in SIZE.findall(network_free(t)):
        kind, factor = SIZE_UNIT[unit]
        if kind == "volume":
            ml = float(value) * factor
            out["size"] = f"{ml / 1000:g} L" if ml >= 1000 else f"{ml:g} ml"
        elif kind == "mass":
            g = float(value) * factor
            out["size"] = f"{g / 1000:g} kg" if g >= 1000 else f"{g:g} g"
        if "size" in out:
            break
    m = re.search(r"\b(?:pack|set) of (\d+)|\b(\d+)\s?(?:pcs|pieces)\b", t)
    if m:
        out["pack"] = f"Pack of {m.group(1) or m.group(2)}"
    return out


def match_search(query, title, store=None):
    """query_match (src/verdict.py), plus the details the user typed: "iphone 17 pro 256gb orange" makes a
    512 GB or Silver listing "related" (another variant), not "exact". A listing that does not state the detail
    stays exact (titles often leave out colour or storage)."""
    m = query_match(query, title)
    lead = lead_word(query)
    if m == "related" and lead and lead in (store or "").lower():   # the brand's own store leaves out the brand
        m = query_match(query, f"{lead} {title}")
    if m != "exact":
        return m
    want, have = attributes(query), attributes(title)
    for key in ("storage", "ram", "size", "pack"):
        if want.get(key) and have.get(key) and want[key] != have[key]:
            return "related"
    if have.get("pack") and not want.get("pack") and have["pack"] != "Pack of 1":   # a 2-pack is not one item
        return "related"
    if colours(query) and colours(title) and not colours(query) & colours(title):
        return "related"
    return m


# Authorised resellers (store names that say so, and well-known Apple Premium Resellers in India).
AUTHORISED = re.compile(r"authori[sz]ed|premium reseller|\bofficial\b|\bunicorn\b|\bimagine\b|icrescent|maple store|"
                        r"aptronix|iplanet|\bample\b", re.IGNORECASE)
# Refurbished / used: words in the title, or sellers whose business is refurbished or used goods.
REFURB_WORDS = re.compile(r"\b(refurbished|renewed|pre-?owned|pre-?loved|second[- ]hand|used|open[- ]box|unboxed|"
                          r"(?:super|good|fair|excellent|superb|very good|a|b|c) grade|grade [abc]|excellent condition|"
                          r"like new|certified refurbished|wholesale)\b|/\s*(?:excellent|good|fair|superb)\b", re.IGNORECASE)
REFURB_STORES = re.compile(r"cashify|ovantica|budli|grest|gudfast|control ?z|rebuysell|yaantra|togofogo|easyphones|"
                           r"tetro|refit|befach|eazypc|fonezone", re.IGNORECASE)
# International import stores: priced for importing (foreign price + shipping + duty), not the Indian market price.
IMPORT_STORES = re.compile(r"ubuy|desertcart|microless|etsy|stockx|ebay|etoren|cremashop|halfprice|garmade|biblio|"
                           r"\.ae\b|\.eu\b|\.uk\b|amazon\.com\b", re.IGNORECASE)


def sells_own_stock(store, title):
    """A retailer selling its own stock (FIRST_PARTY), an authorised reseller, or the brand's own store named like
    one ("boAt", "JBL India", "Wildcraft Official"; not "AirPods Guru")."""
    store = store or ""
    tier = store_tier(store, title)
    words = store.lower().split()
    official = tier == "brand" and (len(words) == 1 or words[1] in ("india", "official", "store", "online", "in"))
    return bool(FIRST_PARTY.search(store)) or tier == "authorised" or official


def store_tier(store, title):
    """'big' (retailer, marketplace, quick commerce, pharmacy), 'authorised' (authorised / official reseller),
    'brand' (the brand's own store), or 'other' (a less-known shop)."""
    store = store or ""
    if WELL_KNOWN.search(store):
        return "big"
    if AUTHORISED.search(store):
        return "authorised"
    return "brand" if well_known(store, title) else "other"


COPY_WORDS = re.compile(r"\b(?:first|1st|master|mirror|7a|aaa)\s*copy\b|\breplica\b|\bclone\b|\bdupe\b|"
                        r"\bcopy\s+(?:smart|product|earbuds?|headphones?|watch|shoes?|sneakers?)\b", re.IGNORECASE)


def condition(title, store):
    """'refurbished' when the title says refurbished / used / renewed / graded, or the seller deals in them;
    'bulk' for wholesale lots ("1000 Pieces iPhone 16 Pro", "MOQ 10"); 'copy' when the title admits it is a copy
    ("first copy", "replica", "clone"); else 'new'."""
    if COPY_WORDS.search(title or ""):
        return "copy"
    if REFURB_WORDS.search(title or "") or REFURB_STORES.search(store or ""):
        return "refurbished"
    if re.search(r"\b(?:\d{2,}\s?(?:pieces|pcs|units)|moq|bulk|lot of \d+)\b", (title or "").lower()):
        return "bulk"
    return "new"


def parse_price(text):
    """'₹1,299.00' -> 1299.0; None when there is no number."""
    m = re.search(r"\d[\d,]*(\.\d+)?", str(text or ""))
    return float(m.group(0).replace(",", "")) if m else None


def download(url):
    """Photo bytes, or None if it cannot be downloaded (that offer is then dropped, not guessed)."""
    try:
        if url.startswith("data:image"):  # some results embed the photo directly
            import base64
            return base64.b64decode(url.split(",", 1)[1])
        r = requests.get(url, timeout=15, headers=HEADERS)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


@functools.lru_cache(maxsize=1024)
def cached_download(url):
    """download() with a memory cache, so the Photo X-ray does not fetch photos the search already fetched."""
    return download(url)


# ---------- 1. understand the input ----------

def clean_title(t):
    """Shop page titles -> product name: drop 'Buy', ': Amazon.in: Beauty', '| Flipkart', 'Online at ...'."""
    t = re.split(r"\s+[:|]\s+|\s+-\s+Buy\b|\s+Online at\b|\s+at Best Price\b", t)[0]
    t = re.sub(r"^\s*Buy\s+", "", t, flags=re.IGNORECASE)
    return " ".join(t.split()[:12]).strip(" ,-|")


def name_from_link(url):
    """Product name from a store link. Amazon/Flipkart put the name in the address itself;
    short links (amzn.in, fkrt.it) are followed first. Other shops: the page's own title."""
    host = urlparse(url).netloc.lower()
    if any(s in host for s in ("amzn.", "fkrt.", "dl.flipkart")):
        try:
            url = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True, stream=True).url
        except Exception:
            pass
    parts = [unquote(p) for p in urlparse(url).path.split("/") if p]
    for marker in ("dp", "gp", "p"):  # amazon.in/<name>/dp/<id>, flipkart.com/<name>/p/<id>
        if marker in parts and parts.index(marker) > 0:
            return clean_title(parts[parts.index(marker) - 1].replace("-", " "))
    try:
        html = requests.get(url, headers=HEADERS, timeout=15).text
        m = re.search(r'<meta[^>]+property="og:title"[^>]+content="([^"]+)"', html) or re.search(r"<title>([^<]+)", html)
        if m:
            return clean_title(m.group(1))
    except Exception:
        pass
    words = max(parts, key=len, default="").replace("-", " ").replace("_", " ")
    return clean_title(words) if len(words) > 5 else None


def name_from_photo(image_url):
    """Google Lens (through Serper) on a public photo address -> product name + pages that show it."""
    hits = serper("lens", {"url": image_url}).get("organic", [])  # gl/hl make Lens return nothing (tested)
    shops = [h for h in hits if h.get("title") and h.get("source", "").lower().split(".")[0] not in NOT_SHOPS]
    name = next((clean_title(h["title"]) for h in shops if len(clean_title(h["title"]).split()) >= 2), None)
    seen = [dict(title=h["title"], store=h.get("source"), link=h.get("link"), image=h.get("imageUrl")) for h in shops[:8]]
    return name, seen


def is_image_url(url):
    return bool(re.search(r"\.(jpe?g|png|webp|gif|avif)(\?|$)", url.lower())) or "images-amazon" in url \
        or "media-amazon" in url or "encrypted-tbn" in url or "rukminim" in url


def understand(text=None, image_bytes=None, image_url=None):
    """Whatever the user gave -> dict(mode, query, ref_photo, seen_on). ref_photo is the user's own
    photo (bytes): it is compared with every offer by the model, like any other listing."""
    text = (text or "").strip()
    if text.startswith(("http://", "https://")) and not image_url:
        if is_image_url(text):
            image_url, text = text, ""
        else:
            name = name_from_link(text)
            if not name:
                raise ValueError("Could not read a product name from that link. Paste the product name instead.")
            return dict(mode="link", query=name, ref_photo=None, seen_on=[], link=text)
    if image_bytes is not None or image_url:
        if image_url is None:  # an uploaded file needs a public address before Lens can look at it
            image_url = public_photo_url(image_bytes)
        photo = image_bytes or download(image_url)
        name, seen = name_from_photo(image_url)
        if text:  # the user typed a name as well: trust it
            name = text
        if not name:
            raise ValueError("Google Lens could not recognise that photo. Try a clearer product photo, or type the name.")
        return dict(mode="photo", query=name, ref_photo=photo, seen_on=seen)
    if len(text) < 2:
        raise ValueError("Type a product name, paste a link, or add a photo.")
    return dict(mode="name", query=text, ref_photo=None, seen_on=[])


def public_photo_url(image_bytes):
    """Save an uploaded photo where Google Lens can fetch it: our own server's /uploads/ folder.
    Only works when the app runs online (PUBLIC_URL set, e.g. on Hugging Face Spaces)."""
    base = os.environ.get("PUBLIC_URL")
    if not base:
        raise ValueError("Uploaded photos work on the online version. On this laptop, paste the photo's "
                         "address instead (right-click the photo -> Copy image address), or type the name.")
    folder = config.CACHE_DIR / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha1(image_bytes).hexdigest()[:16] + ".jpg"
    (folder / name).write_bytes(image_bytes)
    return f"{base.rstrip('/')}/uploads/{name}"


# ---------- 2-4. signals and pair scores ----------

def encode_all(matcher, titles, photos):
    """Signals for many offers at once (batched version of Matcher.encode)."""
    from src.embed import encode, load_image
    from src.extra import colour_hist
    clip, mpnet, dino = matcher.encoders()
    ims, hashes = zip(*[load_image(io.BytesIO(b)) for b in photos])
    return dict(img=encode(clip, list(ims), batch_size=32), txt=encode(mpnet, list(titles), batch_size=32),
                phash=np.array(hashes, dtype=np.uint64), dino=dino(list(ims)),
                colour=np.vstack([colour_hist(im) for im in ims]))


def score_pairs(matcher, sig, titles):
    """Model probability, verdict and the main pair features for every pair (a < b) of offers.

    Features are built like src/features.py: the offers are the gallery, and
    rank_ab is where b sits in a's list of nearest offers (k + 1 if beyond k).
    """
    n = len(titles)
    img, txt = matcher._project(sig["img"], "img"), matcher._project(sig["txt"], "txt")  # PCA: 512 / 768 -> 32
    ret = matcher._retrieval(img, txt)
    sims = ret @ ret.T
    np.fill_diagonal(sims, -np.inf)
    k = min(matcher.k, n - 1)
    rank = np.full((n, n), k + 1)
    for i in range(n):
        for pos, j in enumerate(np.argsort(-sims[i])[:k], start=1):
            rank[i, j] = pos
    a, b = np.triu_indices(n, 1)
    tf = matcher.tfidf.transform(titles)
    tokens = [extract(t) for t in titles]
    num = np.array([compare(tokens[i], tokens[j]) for i, j in zip(a, b)], dtype=float).reshape(-1, 2)
    lengths = np.array([len(t) for t in titles], dtype=float)
    X = pd.DataFrame({
        "img_cos": (img[a] * img[b]).sum(1), "txt_cos": (txt[a] * txt[b]).sum(1),
        "tfidf_cos": np.asarray(tf[a].multiply(tf[b]).sum(1)).ravel(),
        "phash_dist": np.bitwise_count(sig["phash"][a] ^ sig["phash"][b]).astype(float),
        "num_match": num[:, 0], "num_conflict": num[:, 1],
        "mutual": ((rank[a, b] <= k) & (rank[b, a] <= k)).astype(float),
        "rank_min": np.minimum(rank[a, b], rank[b, a]).astype(float),
        "rank_max": np.maximum(rank[a, b], rank[b, a]).astype(float),
        "len_ratio": np.minimum(lengths[a], lengths[b]) / np.maximum(lengths[a], lengths[b]),
        "dino_cos": (sig["dino"][a] * sig["dino"][b]).sum(1),
        "colour_dist": colour_dist(sig["colour"][a], sig["colour"][b]),
    })[matcher.cfg["features_v2"]]
    prob = matcher.model.predict_proba(X)[:, 1]
    combo = np.array([combo_mismatch(titles[i], titles[j]) for i, j in zip(a, b)])
    names = name_levels(titles)  # rare words in this search = model / line / shade names: 2 other model, 1 variant
    level = np.array([names(i, j) for i, j in zip(a, b)])
    other_model = np.array([model_conflict(titles[i], titles[j]) for i, j in zip(a, b)]) | (level == 2)
    # same model but another size, colour or pack -> "variant" (rules in src/verdict.py; not model features)
    sizes = np.array([size_conflict(titles[i], titles[j]) or colour_conflict(titles[i], titles[j])
                      or pack_mismatch(titles[i], titles[j]) or storage_unconfirmed(titles[i], titles[j])
                      for i, j in zip(a, b)]) | (level == 1)
    verdict = classify(prob, num[:, 1].astype(bool) | sizes, X["colour_dist"].to_numpy(), combo, matcher.cfg, other_model)
    verdict_lr = classify(prob, num[:, 1], X["colour_dist"].to_numpy(), combo, matcher.cfg)  # ablation: no live rules
    P, V, V_lr = np.eye(n), np.full((n, n), "same", dtype=object), np.full((n, n), "same", dtype=object)
    P[a, b] = P[b, a] = prob
    V[a, b] = V[b, a] = verdict
    V_lr[a, b] = V_lr[b, a] = verdict_lr
    F = {f: np.zeros((n, n)) for f in ("tfidf_cos", "img_cos", "dino_cos", "colour_dist")}
    for f in F:
        F[f][a, b] = F[f][b, a] = X[f].to_numpy()
    return P, V, F, ret, V_lr


CALIBRATOR = config.CACHE_DIR / "calibrator_v2.joblib"  # written by src/calibrate.py (Shopee validation)


def calibrate(prob):
    """Calibrated probability (Platt or isotonic, fitted on Shopee validation); None if not fitted yet."""
    if not CALIBRATOR.exists():
        return None
    import joblib
    cal = joblib.load(CALIBRATOR)
    if cal["kind"] == "platt":
        p = np.clip(prob, 1e-6, 1 - 1e-6)
        return cal["model"].predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1].reshape(prob.shape)
    return cal["model"].predict(prob.ravel()).reshape(prob.shape)


def pca_map(ret):
    """2-D map of the offers: PCA of their 64-number retrieval vectors (centred, top 2 directions)."""
    Z = ret - ret.mean(0)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    xy = U[:, :2] * S[:2]
    share = (S[:2] ** 2 / (S ** 2).sum()).round(3).tolist()
    return np.round(xy, 4).tolist(), share


def group_offers(V, order):
    """Greedy groups: take the top-ranked unassigned offer, add every offer the model calls 'same' as it.

    Comparing each offer to the group's first offer (not chaining a-b-c) avoids the
    long chains that connected components made on the Indian catalog.
    """
    left, groups = list(order), []
    while left:
        seed = left[0]
        members = [j for j in left if j == seed or V[seed, j] == "same"]
        groups.append(dict(seed=seed, members=members))
        left = [j for j in left if j not in members]
    return groups


# Store websites whose product pages may be added from Google web search (exact product links with prices).
STORE_DOMAINS = {"amazon.in": ("Amazon", "big"), "flipkart.com": ("Flipkart", "big"), "croma.com": ("Croma", "big"),
                 "reliancedigital.in": ("Reliance Digital", "big"), "jiomart.com": ("JioMart", "big"),
                 "tatacliq.com": ("Tata CLiQ", "big"), "vijaysales.com": ("Vijay Sales", "big"), "myntra.com": ("Myntra", "big"),
                 "nykaa.com": ("Nykaa", "big"), "bigbasket.com": ("BigBasket", "big"), "blinkit.com": ("Blinkit", "big"),
                 "ajio.com": ("AJIO", "big"), "snapdeal.com": ("Snapdeal", "big"), "pharmeasy.in": ("PharmEasy", "big"),
                 "1mg.com": ("Tata 1mg", "big"), "decathlon.in": ("Decathlon", "big"), "apple.com": ("Apple Store", "brand"),
                 "samsung.com": ("Samsung", "brand"), "mi.com": ("Xiaomi", "brand"), "oneplus.in": ("OnePlus", "brand"),
                 "boat-lifestyle.com": ("boAt", "brand"), "iplanet.one": ("iPlanet (Apple Premium Reseller)", "authorised"),
                 "imagineonline.store": ("Imagine (Apple Premium Reseller)", "authorised"),
                 "unicornstore.in": ("Unicorn (Apple Premium Reseller)", "authorised")}
SNIPPET_PRICE = re.compile(r"(?:₹|rs\.?|inr)\s?([\d,]+(?:\.\d+)?)(\s*(?:/|per)\s*(?:mo|month)\w*)?", re.IGNORECASE)
NOT_A_PRICE = re.compile(r"emi|/mo|month|off\b|discount|cashback|exchange|bonus|save|saving|coupon|upto|up to", re.IGNORECASE)
MRP = re.compile(r"\bm\.?r\.?p\b|regular price|list price|maximum retail", re.IGNORECASE)
NOT_A_PRODUCT_PAGE = re.compile(r"blog|news|article|resource-center|unboxed|review|compare|specification|/specs|"
                                r"price-in-india|/c/|/stories/|/guide", re.IGNORECASE)


def snippet_price(snippet):
    """First rupee amount in a Google snippet that is not an EMI, discount, cashback or exchange amount, and not the
    MRP (the printed maximum price, often 2-4x what the product sells for in India)."""
    for m in SNIPPET_PRICE.finditer(snippet or ""):
        before, after = snippet[max(0, m.start() - 20):m.start()], snippet[m.end():m.end() + 12]
        if m.group(2) or NOT_A_PRICE.search(before) or MRP.search(re.split(r"₹|rs\.?|inr", snippet[max(0, m.start() - 30):m.start()], flags=re.IGNORECASE)[-1]) or re.match(r"\s*(?:off|cashback|discount|instant|bank|emi)", after, re.IGNORECASE):
            continue
        value = parse_price(m.group(1))
        if value and value >= 10:
            return value
    return None


def typical_price(offers, pages):
    """The fair price of the searched product, and where it comes from.
    International import stores never count (their prices include shipping and duty).
    Normally: the median price of the trusted stores' exact, new, non-accessory shopping listings when 3+ have a
    price; with fewer, of all of them plus the big stores' own product pages. But when the big stores' pages
    (2+ prices) say the product costs more than 4 times what the shopping listings say (more than any normal
    discount), most shopping listings are cheap copies (AirPods Pro 2 on Google Shopping India): then the median of
    the big stores' page prices is the fair price. Page prices far (0.5x-2x) from the fair price are dropped:
    snippets also show EMI, exchange and accessory amounts."""
    base = [o for o in offers if o.get("match") == "exact" and o.get("price_value") and o.get("condition") == "new"
            and not o.get("accessory") and not o.get("imported")]
    trusted = [o["price_value"] for o in base if o.get("known")]
    page_prices = [p["price_value"] for p in pages if p.get("price_value")]
    shop = trusted if len(trusted) >= 3 else [o["price_value"] for o in base]
    shop_typical = float(np.median(shop)) if len(shop) >= 3 else None
    if len(page_prices) >= 2 and shop_typical is not None and np.median(page_prices) > 4 * shop_typical:
        typical, source = float(np.median(page_prices)), "store pages"
    elif len(trusted) >= 3:
        typical, source = shop_typical, "listings"
    else:   # few trusted shopping prices: all exact listings and the big stores' page prices (some are unlabelled MRPs)
        both = [o["price_value"] for o in base] + page_prices
        typical, source = (float(np.median(both)) if len(both) >= 3 else shop_typical), "listings"
    for p in pages:
        if p.get("price_value") and typical and not 0.5 * typical <= p["price_value"] <= 2 * typical:
            p["price"] = p["price_value"] = None
    return typical, source


def store_pages(query):
    """Exact product pages on big / official store websites from Google web search (1 Serper credit): the link goes
    straight to the product, the price is read from the page snippet Google shows (confirm on the page; prices far
    from the fair price are dropped by typical_price)."""
    try:
        hits = serper("search", {"q": f"{query} price", "gl": "in", "hl": "en", "num": 20}).get("organic", [])
    except Exception:
        return []
    out, seen = [], set()
    for h in hits:
        host = urlparse(h.get("link", "")).netloc.lower().removeprefix("www.")
        site = next((v for d, v in STORE_DOMAINS.items() if host == d or host.endswith("." + d)), None)
        title = clean_title(h.get("title", "")) or h.get("title", "")
        if not site or NOT_A_PRODUCT_PAGE.search(h.get("link", "")) or match_search(query, title, site[0]) != "exact" \
                or condition(title, site[0]) != "new":
            continue
        price = snippet_price(h.get("snippet", ""))
        key = (site[0], title.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(title=title, store=site[0], price=f"₹{price:,.0f}" if price else None, price_value=price,
                        link=h["link"], image=None, rating=h.get("rating"), rating_count=None, delivery=None,
                        known=True, tier=site[1], condition="new", first_party=True, accessory=False, attrs=attributes(title),
                        match="exact", source="store page",
                        price_note="from" if re.search(r"\bfrom\b|\bstart(?:s|ing)?\b", h.get("snippet", ""), re.IGNORECASE) else None))
    return out


def search(matcher, text=None, image_bytes=None, image_url=None, num=40, vectors=False, tag=None):
    """Full live search -> JSON-ready dict for the website."""
    t = {"start": time.time()}
    inp = understand(text, image_bytes, image_url)
    # One result page (40 offers). A second page was tried: it doubled cost and time and mostly added
    # accessories (cases, covers), not more sellers of the product.
    raw = serper("shopping", {"q": inp["query"], "gl": "in", "hl": "en", "num": num}, tag=tag).get("shopping", [])
    raw = [o for o in raw if o.get("title") and o.get("imageUrl")]
    # Smart fallback: if no trusted store sells the item new in these results, ask Google Shopping again with the
    # big marketplaces' names (2 credits each, only then). Discontinued products may still return only accessories.
    trusted_new = [o for o in raw if store_tier(o.get("source"), o["title"]) != "other"
                   and condition(o["title"], o.get("source")) == "new" and not is_accessory(o["title"])]
    extra = []
    if len(trusted_new) < 2 and tag is None:
        with ThreadPoolExecutor(2) as pool:
            more = list(pool.map(lambda st: serper("shopping", {"q": f"{inp['query']} {st}", "gl": "in", "hl": "en", "num": num}).get("shopping", []),
                                 ["amazon", "flipkart"]))
        seen = {(o["title"].lower(), o.get("source", "").lower(), o.get("price")) for o in raw}
        for o in more[0] + more[1]:
            key = (o.get("title", "").lower(), o.get("source", "").lower(), o.get("price"))
            if o.get("title") and o.get("imageUrl") and key not in seen and len(raw) < 2 * num:
                seen.add(key)
                raw.append(o)
        extra = ["amazon", "flipkart"]
    t["offers"] = time.time()
    with ThreadPoolExecutor(16) as pool:
        photos = list(pool.map(lambda o: download(o["imageUrl"]), raw))
    keep = [i for i, p in enumerate(photos) if p]
    offers = [dict(title=raw[i]["title"], store=raw[i].get("source", ""), price=raw[i].get("price"),
                   price_value=parse_price(raw[i].get("price")), link=raw[i].get("link"),
                   image=raw[i]["imageUrl"], rating=raw[i].get("rating"), rating_count=raw[i].get("ratingCount"),
                   delivery=raw[i].get("delivery"), known=store_tier(raw[i].get("source"), raw[i]["title"]) != "other",
                   tier=store_tier(raw[i].get("source"), raw[i]["title"]),
                   imported=bool(IMPORT_STORES.search(raw[i].get("source") or "")),
                   first_party=sells_own_stock(raw[i].get("source"), raw[i]["title"]),
                   condition=condition(raw[i]["title"], raw[i].get("source")),
                   accessory=is_accessory(raw[i]["title"]), attrs=attributes(raw[i]["title"]),
                   # what the user typed is the reference for name / link searches (not for photos: Lens only guesses)
                   match=match_search(inp["query"], raw[i]["title"], raw[i].get("source")) if inp["mode"] != "photo" else None)
              for i in keep]
    photos = [photos[i] for i in keep]
    if inp["ref_photo"]:  # the user's own photo is item 0, compared with every offer like a listing
        offers.insert(0, dict(title=inp["query"], store="Your photo", price=None, price_value=None, link=None,
                              image=image_url, is_ref=True))
        photos.insert(0, inp["ref_photo"])
    t["photos"] = time.time()
    out = dict(query=inp["query"], mode=inp["mode"], seen_on=inp["seen_on"], found=len(raw), extra_search=extra,
               dropped_no_photo=len(raw) - len(keep), offers=offers, groups=[])
    if len(offers) >= 2:
        titles = [o["title"] for o in offers]
        sig = encode_all(matcher, titles, photos)
        t["encode"] = time.time()
        P, V, F, ret, V_lr = score_pairs(matcher, sig, titles)
        # Start each group from the offer most other offers confirm as the same item (the user's photo first).
        links = (V == "same").sum(1)
        order = sorted(range(len(offers)), key=lambda j: (not offers[j].get("is_ref"), -links[j], j))
        out["groups"] = group_offers(V, order)
        out["prob"] = np.round(P, 4).tolist()          # lets the page re-group around any offer the user clicks
        cal = calibrate(P)
        out["prob_cal"] = np.round(cal, 4).tolist() if cal is not None else None  # used for the sort order
        out["verdict"] = [[v[0].upper() for v in row] for row in V]  # S / V / D
        out["verdict_lr"] = [[v[0].upper() for v in row] for row in V_lr]  # model + v2 rules only (ablation)
        out["features"] = {f: np.round(m, 3).tolist() for f, m in F.items()}
        out["pca_xy"], out["pca_share"] = pca_map(ret)
        # Bait-listing detector (PCA): among listings whose TITLE names the searched product, a photo that the other
        # listings' main PCA directions cannot rebuild (DINOv2 features, leave-one-out, robust z > 2.5) shows
        # something else. Only less-known shops are flagged (a big store, authorised reseller or brand store with an
        # unusual photo is a photo choice, not a scam); flagged listings are never the best price.
        # Injection test: results/json/photo_xray_test*.json.
        from src.photo_xray import loo_residuals, robust_z
        exact = [i for i, o in enumerate(offers) if o.get("match") == "exact"]
        if len(exact) >= 5:
            D = np.asarray(sig["dino"], dtype=np.float64)[exact]
            z = robust_z(np.sqrt((loo_residuals(D, min(3, len(exact) - 2)) ** 2).sum(1)))
            for i, zi in zip(exact, z):
                offers[i]["photo_z"] = round(float(zi), 2)
                offers[i]["photo_mismatch"] = bool(zi > 2.5 and offers[i]["tier"] == "other")
        if vectors:  # 64-number retrieval vectors (PCA 32 + 32), for the cross-search PCA map in src/insights.py
            out["vectors"] = np.round(ret, 5).tolist()
        t["score"] = time.time()
    # Big-store product pages for name / link searches: added AFTER the model scored the shopping offers (they have
    # no photo); matched to the search by name only, so their "probability" is -1 (shown as "matched by name").
    if inp["mode"] != "photo" and tag is None and out.get("verdict"):
        pages = store_pages(inp["query"])
        out["typical"], out["typical_source"] = typical_price(offers, pages)
        n = len(offers)
        for k, pg in enumerate(pages):
            offers.append(pg)
            for i, row in enumerate(out["verdict"]):
                row.append("V" if offers[i].get("match") == "exact" else "D")
                out["prob"][i].append(-1)
                if out.get("prob_cal"):
                    out["prob_cal"][i].append(-1)
        m = len(pages)
        for k in range(m):
            out["verdict"].append([out["verdict"][i][n + k] for i in range(n)] + ["S" if a == k else "V" for a in range(m)])
            out["prob"].append([-1] * (n + m))
            if out.get("prob_cal"):
                out["prob_cal"].append([-1] * (n + m))
        out["store_pages"] = m
        t["pages"] = time.time()
    out["query_attrs"] = attributes(inp["query"]) if inp["mode"] != "photo" else {}
    names = {"offers": "Find offers", "photos": "Download photos", "encode": "Neural networks read photos + titles",
             "score": "Model scores every pair", "pages": "Big-store product pages"}
    keys = list(t)
    out["timing"] = [dict(step=names[k], seconds=round(t[k] - t[keys[i]], 2)) for i, k in enumerate(keys[1:])]
    out["seconds"] = round(time.time() - t["start"], 2)
    return out


def store_link(title, store):
    """The real product page at that store: one Serper web search (1 credit), first hit on the store's site."""
    key = re.sub(r"[^a-z0-9]", "", store.lower().replace(".com", "").replace(".in", ""))
    hits = serper("search", {"q": f"{title} {store}", "gl": "in", "hl": "en", "num": 10}).get("organic", [])
    for h in hits:
        host = re.sub(r"[^a-z0-9]", "", urlparse(h.get("link", "")).netloc.lower())
        # Amazon.in / Flipkart are Indian sites: never send a buyer to amazon.com
        if key and key[:6] in host and not (key.startswith("amazon") and "amazonin" not in host):
            return h["link"]
    return f"https://www.google.com/search?q={quote_plus(title + ' ' + store)}"


def save_examples(matcher, queries):
    """Run searches and save them as dashboard/data/searches/*.json for the static site (no server)."""
    folder = config.ROOT / "dashboard" / "data" / "searches"
    folder.mkdir(parents=True, exist_ok=True)
    index = []
    for q in queries:
        res = search(matcher, q)
        res["cached_at"] = time.strftime("%d %b %Y")
        name = re.sub(r"[^a-z0-9]+", "-", q.lower()).strip("-") + ".json"
        (folder / name).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        index.append(dict(q=q, file=name))
        print(f"saved {name}: {len(res['offers'])} offers, top group {len(res['groups'][0]['members']) if res['groups'] else 0}")
    (folder / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    from src.service import Matcher
    if sys.argv[1:2] == ["--save"]:  # python -m src.live --save "query one" "query two" ...
        save_examples(Matcher(), sys.argv[2:])
        sys.exit()
    res = search(Matcher(), " ".join(sys.argv[1:]) or "boat airdopes 141")
    print(f"[{res['mode']}] {res['query']} | {res['found']} offers, {res['dropped_no_photo']} without photo, {res['seconds']} s")
    print(" | ".join(f"{s['step']} {s['seconds']} s" for s in res["timing"]))
    o = res["offers"]
    for g in res["groups"][:6]:
        print(f"\n[{len(g['members'])}] {o[g['seed']]['title'][:80]}")
        for j in g["members"]:
            print(f"   - {o[j]['store']:<22} {str(o[j]['price']):<12} {o[j]['title'][:60]}")
