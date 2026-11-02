"""PCA Genuine Look score: does a listing's photo look like the REAL product?

1. Reference = the photos of every listing whose title names the searched product: most sellers show the real
   product, so their consensus is its genuine look. (Tested alternative: only the photos of stores selling
   genuine stock as the reference. On 13 searches it caught fewer injected wrong photos, 84.3% vs 98.4%, and
   flagged more small-shop listings, 27.3% vs 7.8%: results/json/genuine_look_eval.json. So the consensus is used.)
   Limit: a copy built to look exactly like the real product passes any photo check; price gives it away, which
   is why the Scam Radar on the website combines this score with the price.
2. PCA on the reference photos' DINOv2 features (384 numbers each) finds the few directions in which genuine
   photos of this product vary (angle, lighting, colour). That is the product's "genuine look".
3. Every listing's photo is rebuilt from those directions. A photo of the real product is rebuilt well; a copy,
   another item or a logo leaves a large rebuild error. Reference photos are judged leave-one-out (rebuilt from
   the OTHER reference photos), so none is judged against itself.
4. The error is compared with the reference photos' own errors (robust z-score); z > 2.5 = "doesn't look like
   the real product" (the same line as the tested bait check). Genuine look % = 100 / (1 + e^(z - 2.5)):
   about 92% for a typical genuine photo, 50% exactly at the line, low beyond it.
Only less-known shops are ever warned about (src/live.py); trusted stores are the reference, not suspects.
Run: python -m src.genuine --eval   -> results/json/genuine_look_eval.json
"""
import json
import sys

import numpy as np

from src import config

LINE = 2.5   # robust z above this = does not look like the real product


def _residual(x, mu, comp):
    z = x - mu
    return float(np.linalg.norm(z - (z @ comp.T) @ comp))


def genuine_look(D, reference=None, k=3):
    """D: (n, d) DINOv2 vectors of one product's listings; reference: n booleans (default: all listings).
    -> dict(z=[...], score=[...], reference="all listings" | "chosen photos", n_ref)."""
    D = np.asarray(D, dtype=np.float64)
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    n = len(D)
    ref = [i for i in range(n) if reference is None or reference[i]]
    name = "all listings" if reference is None else "chosen photos"
    kk = max(1, min(k, len(ref) - 2))
    R = D[ref]
    res = np.zeros(n)
    for i in range(n):
        rest = R if i not in ref else np.delete(R, ref.index(i), axis=0)   # leave-one-out for reference photos
        mu = rest.mean(0)
        _, _, vt = np.linalg.svd(rest - mu, full_matrices=False)
        res[i] = _residual(D[i], mu, vt[:min(kk, len(vt))])
    base = res[ref]
    med = np.median(base)
    spread = max(np.median(np.abs(base - med)) * 1.4826, 0.1 * med, 1e-6)   # floor: few reference photos
    z = (res - med) / spread
    score = 100 / (1 + np.exp(z - LINE))
    return dict(z=[round(float(v), 2) for v in z], score=[int(round(float(v))) for v in score],
                reference=name, n_ref=len(ref))


# ---------------- evaluation: slip other products' photos in as fake listings ----------------

def evaluate(queries):
    """For every search with 3+ trusted reference photos: each OTHER search's exact listing photos are slipped in as
    if a less-known shop sold them under this product's name. Caught = z > 2.5. Also reported: how often this
    product's own less-known listings are flagged (an upper bound on false alarms, since some of them really are
    copies). Self-labelled by construction (an injected photo is known to be another product); CIs resample searches."""
    from src.live import search
    from src.live_eval import search_bootstrap  # noqa: F401  (same CI method as the other evaluations)
    from src.service import Matcher
    import pandas as pd
    m, rng = Matcher(), np.random.default_rng(config.SEED)
    data = {}
    for q in queries:
        r = search(m, q, vectors=True)
        ex = [i for i, o in enumerate(r["offers"]) if o.get("match") == "exact" and r.get("dino") and i < len(r["dino"])]
        if len(ex) >= 4:
            data[q] = dict(D=np.asarray([r["dino"][i] for i in ex]), own=[r["offers"][i].get("first_party", False) for i in ex],
                           known=[bool(r["offers"][i].get("known")) for i in ex],
                           store=[r["offers"][i]["store"] for i in ex], title=[r["offers"][i]["title"] for i in ex])
    rows = []
    for q, d in data.items():
        if sum(d["own"]) < 3:   # the alternative needs 3+ photos from stores selling genuine stock
            continue
        g, g_own = genuine_look(d["D"]), genuine_look(d["D"], d["own"])
        for i, o in enumerate(d["known"]):
            if not o:
                rows.append(dict(query=q, kind="small-shop listing flagged (used: all listings as reference)", flagged=g["z"][i] > LINE))
                rows.append(dict(query=q, kind="small-shop listing flagged (tried: own-stock stores as reference)", flagged=g_own["z"][i] > LINE))
        for q2, d2 in data.items():
            if q2 == q:
                continue
            for x in d2["D"][:3]:   # up to 3 photos of each other product
                D = np.vstack([d["D"], x])
                rows.append(dict(query=q, kind="injected wrong photo caught (used: all listings as reference)",
                                 flagged=genuine_look(D)["z"][-1] > LINE))
                rows.append(dict(query=q, kind="injected wrong photo caught (tried: own-stock stores as reference)",
                                 flagged=genuine_look(D, d["own"] + [False])["z"][-1] > LINE))
    df = pd.DataFrame(rows)
    out = {"searches_with_reference": int(df["query"].nunique()) if len(df) else 0}
    for kind, part in df.groupby("kind"):
        per = part.groupby("query")["flagged"].mean()
        boots = [per.sample(len(per), replace=True, random_state=int(s)).mean() for s in rng.integers(0, 1e9, 1000)]
        out[kind] = dict(n=int(len(part)), flagged_pct=round(100 * float(part["flagged"].mean()), 1),
                         ci95=[round(100 * float(np.percentile(boots, 2.5)), 1), round(100 * float(np.percentile(boots, 97.5)), 1)])
    out["note"] = ("Injected photos are other products' real listing photos, so the label is known by construction "
                   "(no hand labels). Own less-known listings include real copies, so their flag rate is an upper "
                   "bound on false alarms.")
    (config.RESULTS_DIR / "json" / "genuine_look_eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if "--eval" in sys.argv:
        evaluate(["iphone 16", "iphone 17 pro", "samsung galaxy s24 fe", "redmi note 14 pro", "macbook air m3",
                  "hp victus 15", "sony wh-1000xm5", "boat airdopes 141", "jbl flip 6", "apple airpods pro 2",
                  "noise colorfit pro 5", "apple watch series 10", "samsung 43 inch crystal 4k tv",
                  "dyson v12 detect slim", "nike air force 1", "levis 511 jeans", "maybelline fit me foundation",
                  "cetaphil gentle skin cleanser", "nivea soft 200ml", "tata salt 1kg", "nescafe classic 200g",
                  "amul butter 500g", "logitech mx master 3s", "kindle paperwhite", "wildcraft backpack",
                  "milton thermosteel 1 litre", "dettol original soap", "boat rockerz 450", "iphone 15",
                  "samsung galaxy buds 3 pro"])
