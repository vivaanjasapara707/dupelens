"""Live search: type any product, see which online stores sell the SAME item, and where it is cheapest.

1. Ask Serper (Google Shopping results for India) for offers matching the search words.
2. Download each offer's photo and compute the signals the model was trained on
   (CLIP, MPNet, phash, DINOv2, colour), exactly as src/service.py does for one listing.
3. Score every pair of offers with the trained M1_v2 model and the same/variant/different rules.
4. Group offers that are the exact same item; per group: cheapest store, typical price, buy links.
Price is only used AFTER matching, never as a model feature.

Needs a free Serper key (serper.dev) in the environment variable SERPER_API_KEY or in a
file .env in the repo root (line: SERPER_API_KEY=...). Each search response is saved in
cache/live/, so repeating a search costs no credit and works offline.
Run once from the command line: python -m src.live "boat airdopes 141"
"""
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from src import config
from src.extra import colour_dist
from src.numtokens import compare, extract
from src.verdict import classify, combo_mismatch

SERPER_URL = "https://google.serper.dev/shopping"
LIVE_CACHE = config.CACHE_DIR / "live"
HEADERS = {"User-Agent": "Mozilla/5.0"}


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


def parse_price(text):
    """'₹1,299.00' -> 1299.0; None when there is no number."""
    m = re.search(r"\d[\d,]*(\.\d+)?", str(text or ""))
    return float(m.group(0).replace(",", "")) if m else None


def fetch_offers(query, num=40):
    """Google Shopping offers for India from Serper; responses cached on disk by query."""
    LIVE_CACHE.mkdir(parents=True, exist_ok=True)
    path = LIVE_CACHE / (hashlib.sha1(f"{query.lower().strip()}|{num}".encode()).hexdigest()[:16] + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["shopping"]
    r = requests.post(SERPER_URL, headers={"X-API-KEY": api_key(), "Content-Type": "application/json"},
                      json={"q": query, "gl": "in", "hl": "en", "num": num}, timeout=30)
    r.raise_for_status()
    data = r.json()
    data["query"] = query
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data.get("shopping", [])


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


def encode_all(matcher, titles, photos):
    """Signals for many offers at once (batched version of Matcher.encode)."""
    import io
    from src.embed import encode, load_image
    from src.extra import colour_hist
    clip, mpnet, dino = matcher.encoders()
    ims, hashes = zip(*[load_image(io.BytesIO(b)) for b in photos])
    return dict(img=encode(clip, list(ims), batch_size=32), txt=encode(mpnet, list(titles), batch_size=32),
                phash=np.array(hashes, dtype=np.uint64), dino=dino(list(ims)),
                colour=np.vstack([colour_hist(im) for im in ims]))


def score_pairs(matcher, sig, titles):
    """Model probability and verdict for every pair (a < b) of offers.

    Features are built like src/features.py: the offers are the gallery, and
    rank_ab is where b sits in a's list of nearest offers (k + 1 if beyond k).
    """
    n = len(titles)
    img, txt = matcher._project(sig["img"], "img"), matcher._project(sig["txt"], "txt")
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
    verdict = classify(prob, num[:, 1], X["colour_dist"].to_numpy(), combo, matcher.cfg)
    P, V = np.eye(n), np.full((n, n), "same", dtype=object)
    P[a, b] = P[b, a] = prob
    V[a, b] = V[b, a] = verdict
    return P, V


def group_offers(V, order):
    """Greedy groups: take the top-ranked unassigned offer, add every offer the model calls 'same' as it.

    Comparing each offer to the group's first offer (not chaining a-b-c) avoids the
    long chains that connected components made on the Indian catalog.
    """
    left, groups = list(order), []
    while left:
        seed = left[0]
        members = [j for j in left if j == seed or V[seed, j] == "same"]
        variants = [j for j in left if V[seed, j] == "variant"]
        groups.append(dict(seed=seed, members=members, variants=variants))
        left = [j for j in left if j not in members]
    return groups


def price_summary(offers, members):
    """Cheapest store and typical (median) price of one group; typical needs >= 3 priced offers."""
    priced = [j for j in members if offers[j]["price_value"] is not None]
    if not priced:
        return dict(best=None, typical=None, min_price=None, max_price=None)
    prices = np.array([offers[j]["price_value"] for j in priced])
    best = priced[int(np.argmin(prices))]
    typical = float(np.median(prices)) if len(prices) >= 3 else None
    return dict(best=best, typical=typical, min_price=float(prices.min()), max_price=float(prices.max()))


def search(matcher, query, num=40):
    """Full live search -> JSON-ready dict for the website."""
    t0 = time.time()
    raw = [o for o in fetch_offers(query, num) if o.get("title") and o.get("imageUrl")]
    with ThreadPoolExecutor(16) as pool:
        photos = list(pool.map(lambda o: download(o["imageUrl"]), raw))
    keep = [i for i, p in enumerate(photos) if p]
    offers = [dict(title=raw[i]["title"], store=raw[i].get("source", ""), price=raw[i].get("price"),
                   price_value=parse_price(raw[i].get("price")), link=raw[i].get("link"),
                   image=raw[i]["imageUrl"], rating=raw[i].get("rating"), rating_count=raw[i].get("ratingCount"),
                   delivery=raw[i].get("delivery"))
              for i in keep]
    out = dict(query=query, found=len(raw), dropped_no_photo=len(raw) - len(keep), offers=offers, groups=[])
    if len(offers) < 2:
        out["seconds"] = round(time.time() - t0, 2)
        return out
    titles = [o["title"] for o in offers]
    sig = encode_all(matcher, titles, [photos[i] for i in keep])
    P, V = score_pairs(matcher, sig, titles)
    for g in group_offers(V, range(len(offers))):
        g.update(price_summary(offers, g["members"]))
        out["groups"].append(g)
    out["prob"] = np.round(P, 4).tolist()          # lets the page re-group around any offer the user clicks
    out["verdict"] = [[v[0].upper() for v in row] for row in V]  # S / V / D
    out["seconds"] = round(time.time() - t0, 2)
    return out


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
    print(f"{res['found']} offers, {res['dropped_no_photo']} without photo, {res['seconds']} s")
    for g in res["groups"][:5]:
        o = res["offers"]
        best = o[g["best"]] if g["best"] is not None else None
        print(f"\n[{len(g['members'])} stores] {o[g['seed']]['title'][:80]}")
        if best:
            print(f"   best: {best['price']} at {best['store']} | typical {g['typical']} | range {g['min_price']}-{g['max_price']}")
        for j in g["members"]:
            print(f"   - {o[j]['store']:<22} {o[j]['price']:<12} {o[j]['title'][:60]}")
