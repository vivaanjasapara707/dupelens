"""PCA Colour Decoder: which colour is the product in each listing photo, from the photo alone.

The most common wrong "same item" in the final evaluation round was a colour or finish that differs only in the
photos (6 of 17 errors, results/error_analysis_final.csv). Titles often do not name the colour, so:
1. Each photo's product is cut out of its white background (src/photo_xray.cutout) and its pixels are described by a
   68-number colour profile in CIELAB (the colour space built to match human colour difference): a 8x8 histogram of
   the colour plane (a*, b*) plus a 4-bin lightness histogram.
2. PCA over the search's photos finds the directions in which their colours differ; in that space photos of the same
   colour sit together.
3. k-means in the PCA space groups the photos into colours; k (1 to 5) is chosen by silhouette score, and if no
   split is clear (best silhouette < 0.35) all photos are one colour. Every group is named by the nearest of 16
   reference colours to its median product colour.
4. Two listings whose photos land in different colour groups are not called the same item (src/live.py).
Fixed settings above were chosen before any evaluation.
Run: python -m src.colour_pca --eval   -> results/json/colour_pca_eval.json
"""
import io
import json
import sys

import numpy as np
from PIL import Image

from src import config

NAMES = {"Black": "#151515", "White": "#f4f4f4", "Grey": "#8a8a8a", "Silver": "#c4c7cc", "Gold": "#d4af37",
         "Beige": "#d9c4a3", "Brown": "#7b4a2e", "Red": "#c62828", "Pink": "#ec6f9c", "Orange": "#f57c00",
         "Yellow": "#f6c42b", "Green": "#2e7d32", "Teal": "#00897b", "Blue": "#1e66d0", "Navy": "#1d2a6b",
         "Purple": "#7b1fa2"}
MIN_SILHOUETTE, MAX_K = 0.35, 5


def rgb_to_lab(rgb):
    """sRGB in [0, 1] (..., 3) -> CIELAB (D65)."""
    c = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


NAME_LAB = {n: rgb_to_lab(np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])) for n, h in NAMES.items()}


def product_pixels(image):
    """Lab values of the product's pixels (background and pure white removed)."""
    from src.photo_xray import cutout
    a = cutout(image).reshape(-1, 3)
    keep = (a < 0.92).any(1)
    px = a[keep] if keep.sum() >= 30 else a
    return rgb_to_lab(px)


def profile(lab):
    """68-number colour profile: 8x8 histogram of (a*, b*) + 4-bin L* histogram, each part summing to 1."""
    ab, _, _ = np.histogram2d(lab[:, 1], lab[:, 2], bins=8, range=[[-60, 60], [-60, 60]])
    li, _ = np.histogram(lab[:, 0], bins=4, range=[0, 100])
    return np.concatenate([ab.ravel() / max(ab.sum(), 1), li / max(li.sum(), 1)])


def name_of(lab_colour):
    return min(NAME_LAB, key=lambda n: float(np.sum((NAME_LAB[n] - lab_colour) ** 2)))


def decode(images):
    """images: PIL images -> colour group per photo, group names / swatches, and the 2-D PCA colour map."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    labs = [product_pixels(im) for im in images]
    X = np.stack([profile(l) for l in labs])
    n = len(X)
    mu = X.mean(0)
    U, S, vt = np.linalg.svd(X - mu, full_matrices=False)
    share = S ** 2 / max((S ** 2).sum(), 1e-12)
    Z = U[:, :3] * S[:3]                                   # PCA colour space (3 directions)
    best_k, best_s, labels = 1, -1.0, np.zeros(n, dtype=int)
    for k in range(2, min(MAX_K, n - 1) + 1):
        lab_k = KMeans(k, n_init=10, random_state=config.SEED).fit_predict(Z)
        if len(set(lab_k)) < 2:
            continue
        s = silhouette_score(Z, lab_k)
        if s > best_s:
            best_k, best_s, labels = k, s, lab_k
    if best_s < MIN_SILHOUETTE:
        best_k, labels = 1, np.zeros(n, dtype=int)
    groups = []
    for c in range(best_k):
        med = np.median(np.concatenate([labs[i] for i in range(n) if labels[i] == c]), axis=0)
        nm = name_of(med)
        groups.append(dict(name=nm, hex=NAMES[nm], count=int((labels == c).sum())))
    # two groups with the same name (e.g. two shades of blue) keep separate numbers
    seen = {}
    for g in groups:
        seen[g["name"]] = seen.get(g["name"], 0) + 1
        if seen[g["name"]] > 1:
            g["name"] = f'{g["name"]} {seen[g["name"]]}'
    own = [NAMES[name_of(np.median(l, axis=0))] for l in labs]   # each photo's own dominant colour, for the map
    return dict(group=[int(c) for c in labels], groups=groups, k=best_k, silhouette=round(float(best_s), 3),
                xy=np.round(Z[:, :2], 4).tolist(), share=[round(float(x), 3) for x in share[:2]], swatch=own)


def decode_offers(photo_bytes):
    """Like decode(), from raw photo bytes; photos that are missing or unreadable get group -1 ("colour unknown")."""
    ims, idx = [], []
    for i, data in enumerate(photo_bytes):
        try:
            ims.append(Image.open(io.BytesIO(data)).convert("RGB"))
            idx.append(i)
        except Exception:
            pass
    if len(ims) < 2:
        return None
    d = decode(ims)
    group, xy, swatch = [-1] * len(photo_bytes), [None] * len(photo_bytes), [None] * len(photo_bytes)
    for k, i in enumerate(idx):
        group[i], xy[i], swatch[i] = d["group"][k], d["xy"][k], d["swatch"][k]
    return {**d, "group": group, "xy": xy, "swatch": swatch}


# ---------------- evaluation on the labelled live rounds ----------------

def evaluate():
    """For pairs the system called "same" (stratum S of the final round and round 4): would the colour rule (photos in
    different colour groups -> not the same item) have removed the wrong ones? Colour groups are computed over all
    offers of each search with the fixed settings above; CIs resample searches."""
    from src.live import download, search
    from src.live_eval import search_bootstrap
    from src.service import Matcher
    import pandas as pd
    m, rng, out = Matcher(), np.random.default_rng(config.SEED), {}
    for name in ("final", "round4"):
        lab = pd.read_csv(config.RESULTS_DIR / f"live_{name}_sheet.csv").merge(
            pd.read_csv(config.CACHE_DIR / f"live_{name}_key.csv"), on="pair_id")
        s = lab[lab.stratum == "S"].copy()
        split = []
        for q, rows in s.groupby("query"):
            res = search(m, q)
            O = res["offers"]
            col = decode_offers([download(o["image"]) if o.get("image") else None for o in O])
            idx = {o["title"]: i for i, o in reversed(list(enumerate(O)))}
            for r in rows.itertuples():
                ga, gb = col["group"][idx[r.a_title]], col["group"][idx[r.b_title]]
                split.append((r.pair_id, ga >= 0 and gb >= 0 and ga != gb))
        s = s.merge(pd.DataFrame(split, columns=["pair_id", "colour_split"]), on="pair_id")
        kept = s[~s.colour_split]
        same = lambda g: (g["label"] == "same").to_numpy(dtype=float)
        out[name] = dict(pairs=len(s), removed_by_colour=int(s.colour_split.sum()),
                         removed_that_were_not_identical=int(((s.label != "same") & s.colour_split).sum()),
                         removed_that_were_identical=int(((s.label == "same") & s.colour_split).sum()),
                         before=search_bootstrap(s, same, rng), after=search_bootstrap(kept, same, rng))
        print(name, json.dumps(out[name]))
    out["labels"] = "Claude, an AI assistant, not humans"
    (config.RESULTS_DIR / "json" / "colour_pca_eval.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if "--eval" in sys.argv:
        evaluate()
