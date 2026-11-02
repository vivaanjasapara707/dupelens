"""Store insights across many live searches: which store is cheaper, how far prices spread for the SAME item,
and a PCA map of every offer.

--collect : run the searches in QUERIES (each costs 2 Serper credits the first time; answers are cached),
            keep each result with its 64-number retrieval vectors in cache/insights/<slug>.json
--analyse : dashboard/data/insights.json + results/json/insights.json

Price spread uses the middle half of the prices (75th / 25th percentile, needs 4+ prices): the plain max / min
spread was dominated by matching mistakes (bulk packs, accessories), so it is kept per search only for reference.
Per search, the "main item" is the largest group of offers the model calls the same item (accessories such as
cases never count). Store prices are compared only inside that group, so Amazon and Flipkart are compared on
the identical item. CIs resample whole searches (1,000 times).
PCA: the offers' retrieval vectors (CLIP photo and MPNet title, each already compressed by PCA to 32 numbers)
from ALL searches go through one more PCA to 2 numbers, which places every offer on one map; products of
the same category land near each other. The share of variance each direction explains is reported.
"""
import json
import re
import sys

import numpy as np

from src import config

FOLDER = config.CACHE_DIR / "insights"
QUERIES = {
    "Audio & wearables": ["boAt Rockerz 255 Pro Plus", "JBL Go 3 speaker", "Noise ColorFit Pro 4 smartwatch", "Apple AirPods Pro 2",
                          "OnePlus Nord Buds 2r", "boAt Airdopes 141", "Sony WH-1000XM5 headphones", "boAt Rockerz 450 headphones"],
    "Phones & accessories": ["Samsung Galaxy M15 5G 128GB", "iPhone 15 128GB", "Realme Narzo 70 5G", "Anker 20W USB C charger",
                             "SanDisk Ultra 128GB microSD", "Mi Power Bank 3i 20000 mAh", "Redmi 13C 5G 128GB"],
    "Computer": ["Logitech MK270 keyboard mouse combo", "HP 680 ink cartridge black", "TP-Link Archer C6 router",
                 "Samsung T7 portable SSD 1TB", "Logitech M235 wireless mouse", "Casio FX-991ES Plus calculator"],
    "Kitchen": ["Pigeon Favourite electric kettle 1.5 litre", "Prestige Iris mixer grinder 750 W", "Milton Thermosteel Duo DLX 1000 ml",
                "Hawkins Contura pressure cooker 3 litre", "Philips HD9252 air fryer", "Prestige Popular pressure cooker 3 litre",
                "Butterfly mixer grinder 750 W", "Prestige PIC 20 induction cooktop"],
    "Personal care & beauty": ["Himalaya Neem face wash 150 ml", "Dove shampoo 650 ml", "Maybelline Fit Me foundation 128",
                               "Gillette Mach3 razor", "Colgate MaxFresh toothpaste 150 g", "Nivea Soft light moisturising cream 200 ml",
                               "Philips BT3221 beard trimmer", "Boroplus antiseptic cream 80 ml", "Lakme 9 to 5 primer matte lipstick",
                               "Vaseline intensive care deep restore body lotion 400 ml", "Dettol original liquid handwash 200 ml"],
    "Grocery": ["Tata Salt 1 kg", "Fortune sunflower oil 1 litre", "Nescafe Classic coffee 100 g", "Maggi noodles 12 pack",
                "Cadbury Dairy Milk Silk 150 g", "Red Label tea 1 kg"],
    "Home & electrical": ["Havells Stealth Air ceiling fan 1200 mm", "Syska 9W LED bulb pack of 4", "Wipro 9W LED bulb",
                          "Usha table fan 400 mm", "Bajaj DX 6 dry iron"],
    "Stationery, books & toys": ["Classmate notebook 172 pages pack of 6", "Parker Vector pen", "Faber-Castell colour pencils 24",
                                 "Rubik's cube 3x3", "Atomic Habits book paperback"],
    "Fashion & bags": ["Fastrack analog watch men", "Wildcraft 35 L backpack", "Puma unisex cap", "Ray-Ban RB2132 sunglasses"],
}
CATEGORY = {q: c for c, qs in QUERIES.items() for q in qs}
PLATFORMS = [("amazon", "Amazon"), ("shopsy", "Shopsy"), ("flipkart", "Flipkart"), ("jiomart", "JioMart"), ("croma", "Croma"),
             ("reliance", "Reliance Digital"), ("tata", "Tata CLiQ"), ("vijay", "Vijay Sales"), ("myntra", "Myntra"),
             ("nykaa", "Nykaa"), ("blinkit", "Blinkit"), ("zepto", "Zepto"), ("bigbasket", "BigBasket"), ("meesho", "Meesho"),
             ("dmart", "DMart"), ("apollo", "Apollo"), ("pharmeasy", "PharmEasy"), ("1mg", "1mg")]


def slug(q):
    return re.sub(r"[^a-z0-9]+", "-", q.lower()).strip("-")


def platform(store):
    s = (store or "").lower()
    return next((name for key, name in PLATFORMS if key in s), None)


def collect():
    from src.live import search
    from src.service import Matcher
    FOLDER.mkdir(parents=True, exist_ok=True)
    m = Matcher()
    for q in CATEGORY:
        path = FOLDER / f"{slug(q)}.json"
        if path.exists():
            continue
        res = search(m, q, vectors=True)
        res.pop("pair_features", None)   # a table, only for the PCA notebook export
        path.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        print(f"{q}: {len(res['offers'])} offers", flush=True)


def main_group(res):
    """Largest group of offers the model calls the same item, not an accessory (ties: earlier in results)."""
    O, groups = res["offers"], [g for g in res.get("groups", []) if not res["offers"][g["seed"]].get("accessory")]
    return max(groups, key=lambda g: len(g["members"]), default=None) if groups else None


def search_rows():
    rows = []
    for q in CATEGORY:
        path = FOLDER / f"{slug(q)}.json"
        if not path.exists():
            continue
        res = json.loads(path.read_text(encoding="utf-8"))
        O, g = res["offers"], main_group(res)
        if g is None:
            continue
        mem = [O[j] for j in g["members"] if O[j]["price_value"]]
        prices = [o["price_value"] for o in mem]
        by_platform = {}
        for o in mem:
            name = platform(o["store"])
            if name:
                by_platform[name] = min(by_platform.get(name, 1e12), o["price_value"])
        rows.append(dict(query=q, category=CATEGORY[q], offers=len(O), well_known_offers=sum(o["known"] for o in O),
                         stores=len({o["store"] for o in O}), item=O[g["seed"]]["title"], sellers=len(mem),
                         min=min(prices) if prices else None, max=max(prices) if prices else None,
                         median=float(np.median(prices)) if prices else None,
                         # middle-half spread (75th / 25th percentile): one wrong match cannot dominate it, unlike max / min
                         spread_pct=round(100 * (np.percentile(prices, 75) / np.percentile(prices, 25) - 1), 1) if len(prices) >= 4 else None,
                         max_min_pct=round(100 * (max(prices) / min(prices) - 1), 1) if len(prices) >= 2 else None,
                         by_platform=by_platform,
                         cheapest_platform=min(by_platform, key=by_platform.get) if by_platform else None))
    return rows


def boot(rows, stat, rng, n=config.N_BOOT):
    vals = [stat([rows[k] for k in rng.integers(0, len(rows), len(rows))]) for _ in range(n)]
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return [round(float(np.percentile(vals, 2.5)), 1), round(float(np.percentile(vals, 97.5)), 1)] if vals else None


def analyse():
    rng = np.random.default_rng(config.SEED)
    rows = search_rows()
    both = [r for r in rows if "Amazon" in r["by_platform"] and "Flipkart" in r["by_platform"]]
    diff = lambda rs: float(np.median([100 * (r["by_platform"]["Amazon"] / r["by_platform"]["Flipkart"] - 1) for r in rs])) if rs else None
    amazon_cheaper = lambda rs: 100 * np.mean([r["by_platform"]["Amazon"] < r["by_platform"]["Flipkart"] for r in rs]) if rs else None
    spread = [r for r in rows if r["spread_pct"] is not None]
    med_spread = lambda rs: float(np.median([r["spread_pct"] for r in rs])) if rs else None
    over20 = lambda rs: 100 * np.mean([r["spread_pct"] > 20 for r in rs]) if rs else None
    cheapest = {}
    for r in rows:
        if r["cheapest_platform"] and len(r["by_platform"]) >= 2:
            cheapest[r["cheapest_platform"]] = cheapest.get(r["cheapest_platform"], 0) + 1
    by_cat = {}
    for c in QUERIES:
        rs = [r for r in spread if r["category"] == c]
        if rs:
            by_cat[c] = dict(searches=len(rs), median_spread_pct=round(med_spread(rs), 1),
                             median_sellers=float(np.median([r["sellers"] for r in rs])))

    # PCA map over every offer of every search
    pts, vecs = [], []
    for q in CATEGORY:
        path = FOLDER / f"{slug(q)}.json"
        if not path.exists():
            continue
        res = json.loads(path.read_text(encoding="utf-8"))
        for o, v in zip(res["offers"], res.get("vectors", [])):
            vecs.append(v)
            pts.append(dict(category=CATEGORY[q], query=q, title=o["title"][:90], store=o["store"], price=o["price"]))
    V = np.array(vecs, dtype=float)
    Z = V - V.mean(0)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    share = (S ** 2 / (S ** 2).sum())
    xy = U[:, :2] * S[:2]
    for p, (x, y) in zip(pts, xy):
        p["x"], p["y"] = round(float(x), 4), round(float(y), 4)
    # Does the 2-D map keep categories together? Share of each offer's 10 nearest map neighbours in its category,
    # against chance (sum of squared category shares). PCA never saw the categories.
    cat = np.array([p["category"] for p in pts])
    dist = ((xy[:, None] - xy[None]) ** 2).sum(-1)
    np.fill_diagonal(dist, np.inf)
    same_cat = float((cat[np.argsort(dist, 1)[:, :10]] == cat[:, None]).mean())
    chance = float(sum(((cat == c).mean()) ** 2 for c in set(cat)))

    out = dict(
        searches=len(rows), offers=int(sum(r["offers"] for r in rows)), stores=None,
        well_known_share_pct=round(100 * sum(r["well_known_offers"] for r in rows) / max(1, sum(r["offers"] for r in rows)), 1),
        amazon_vs_flipkart=dict(searches_with_both=len(both),
                                median_amazon_vs_flipkart_pct=round(diff(both), 1) if both else None,
                                ci=boot(both, diff, rng) if both else None,
                                amazon_cheaper_pct=round(amazon_cheaper(both), 1) if both else None,
                                amazon_cheaper_ci=boot(both, amazon_cheaper, rng) if both else None,
                                note="negative = Amazon cheaper; same item only (model's main group per search)"),
        price_spread=dict(searches=len(spread), median_spread_pct=round(med_spread(spread), 1) if spread else None,
                          ci=boot(spread, med_spread, rng), share_over_20pct=round(over20(spread), 1) if spread else None,
                          share_over_20pct_ci=boot(spread, over20, rng)),
        cheapest_platform_counts=dict(sorted(cheapest.items(), key=lambda kv: -kv[1])),
        by_category=by_cat,
        pca=dict(points=len(pts), searches=len({p["query"] for p in pts}), explained_variance_pct=[round(100 * float(s), 1) for s in share[:10]],
                 first_two_pct=round(100 * float(share[:2].sum()), 1),
                 neighbours_same_category_pct=round(100 * same_cat, 1), chance_pct=round(100 * chance, 1), input="64 numbers per offer = PCA(CLIP photo, 32) + PCA(MPNet title, 32)"),
        per_search=rows,
        method="search-level bootstrap (1,000 resamples) for CIs; prices as shown by Google Shopping India on the search date",
    )
    (config.RESULTS_DIR / "json" / "insights.json").write_text(json.dumps({k: v for k, v in out.items()}, indent=1, ensure_ascii=False), encoding="utf-8")
    out["map"] = pts
    (config.ROOT / "dashboard" / "data" / "insights.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    summary = {k: out[k] for k in ("searches", "offers", "well_known_share_pct", "amazon_vs_flipkart", "price_spread", "cheapest_platform_counts")}
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("PCA:", out["pca"])


if __name__ == "__main__":
    collect() if "--collect" in sys.argv else analyse()
