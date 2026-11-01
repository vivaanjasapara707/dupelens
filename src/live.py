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
from src.verdict import classify, combo_mismatch, model_conflict, name_conflicts, size_conflict

SERPER = "https://google.serper.dev/"
LIVE_CACHE = config.CACHE_DIR / "live"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}
NOT_SHOPS = {"youtube", "tiktok", "instagram", "pinterest", "facebook", "reddit", "x", "twitter", "quora", "wikipedia"}


def api_key():
    """SERPER_API_KEY from the environment, else from the .env file in the repo root."""
    key = os.environ.get("SERPER_API_KEY")
    env = config.ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("SERPER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        raise RuntimeError("No Serper key: put SERPER_API_KEY=... in .env (free at serper.dev)")
    return key


def serper(endpoint, payload):
    """One Serper call (shopping, lens or search); answers cached on disk by request."""
    LIVE_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{endpoint}|{json.dumps(payload, sort_keys=True).lower()}".encode()).hexdigest()[:16]
    path = LIVE_CACHE / f"{endpoint}_{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    r = requests.post(SERPER + endpoint, headers={"X-API-KEY": api_key(), "Content-Type": "application/json"},
                      json=payload, timeout=40)
    r.raise_for_status()
    data = r.json()
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


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
    names = name_conflicts(titles)  # rare words in this search = model / line / shade names
    other_model = np.array([model_conflict(titles[i], titles[j]) or names(i, j) for i, j in zip(a, b)])
    sizes = np.array([size_conflict(titles[i], titles[j]) for i, j in zip(a, b)])  # watt/litre/gm: verdict only
    verdict = classify(prob, num[:, 1].astype(bool) | sizes, X["colour_dist"].to_numpy(), combo, matcher.cfg, other_model)
    P, V = np.eye(n), np.full((n, n), "same", dtype=object)
    P[a, b] = P[b, a] = prob
    V[a, b] = V[b, a] = verdict
    F = {f: np.zeros((n, n)) for f in ("tfidf_cos", "img_cos", "dino_cos", "colour_dist")}
    for f in F:
        F[f][a, b] = F[f][b, a] = X[f].to_numpy()
    return P, V, F, ret


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


def search(matcher, text=None, image_bytes=None, image_url=None, num=40):
    """Full live search -> JSON-ready dict for the website."""
    t = {"start": time.time()}
    inp = understand(text, image_bytes, image_url)
    raw = serper("shopping", {"q": inp["query"], "gl": "in", "hl": "en", "num": num}).get("shopping", [])
    raw = [o for o in raw if o.get("title") and o.get("imageUrl")]
    t["offers"] = time.time()
    with ThreadPoolExecutor(16) as pool:
        photos = list(pool.map(lambda o: download(o["imageUrl"]), raw))
    keep = [i for i, p in enumerate(photos) if p]
    offers = [dict(title=raw[i]["title"], store=raw[i].get("source", ""), price=raw[i].get("price"),
                   price_value=parse_price(raw[i].get("price")), link=raw[i].get("link"),
                   image=raw[i]["imageUrl"], rating=raw[i].get("rating"), rating_count=raw[i].get("ratingCount"),
                   delivery=raw[i].get("delivery"))
              for i in keep]
    photos = [photos[i] for i in keep]
    if inp["ref_photo"]:  # the user's own photo is item 0, compared with every offer like a listing
        offers.insert(0, dict(title=inp["query"], store="Your photo", price=None, price_value=None, link=None,
                              image=image_url, is_ref=True))
        photos.insert(0, inp["ref_photo"])
    t["photos"] = time.time()
    out = dict(query=inp["query"], mode=inp["mode"], seen_on=inp["seen_on"], found=len(raw),
               dropped_no_photo=len(raw) - len(keep), offers=offers, groups=[])
    if len(offers) >= 2:
        titles = [o["title"] for o in offers]
        sig = encode_all(matcher, titles, photos)
        t["encode"] = time.time()
        P, V, F, ret = score_pairs(matcher, sig, titles)
        # Start each group from the offer most other offers confirm as the same item (the user's photo first).
        links = (V == "same").sum(1)
        order = sorted(range(len(offers)), key=lambda j: (not offers[j].get("is_ref"), -links[j], j))
        out["groups"] = group_offers(V, order)
        out["prob"] = np.round(P, 4).tolist()          # lets the page re-group around any offer the user clicks
        out["verdict"] = [[v[0].upper() for v in row] for row in V]  # S / V / D
        out["features"] = {f: np.round(m, 3).tolist() for f, m in F.items()}
        out["pca_xy"], out["pca_share"] = pca_map(ret)
        t["score"] = time.time()
    names = {"offers": "Find offers", "photos": "Download photos", "encode": "Neural networks read photos + titles",
             "score": "Model scores every pair"}
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
        if key and key[:6] in host:
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
