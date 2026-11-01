"""Photo X-ray: PCA on the product photos of one model's listings ("eigen-products", like eigenfaces).

1. Each photo: the product is cut out of its white background (box around the non-white pixels), padded to a
   square and resized to 48x48 RGB = 6,912 numbers.
2. PCA over the listings' photos (centred): the MEAN photo and the top EIGEN-IMAGES, i.e. the directions in which
   the listing photos differ most (colour, angle, packaging), with the share of variation each explains.
3. Each photo is rebuilt from the top k directions of a PCA fitted on the OTHER photos (leave-one-out, so an odd
   photo cannot hide in a direction of its own). What the rebuild misses is the photo's residual:
     - a heatmap of where the photo differs from what the other listings look like;
     - an odd-one-out score: the residual in DINOv2 space (what the object is, robust to lighting), as a robust
       z-score (median / MAD) within the group. z > 2.5 -> "photo unlike the others" (possibly a different item,
       a used or fake product, an accessory or a stock photo).
Run the injection test: python -m src.photo_xray --test   -> results/json/photo_xray_test.json
"""
import base64
import io
import json
import sys

import numpy as np
from PIL import Image

from src import config

SIDE = 48


def cutout(im):
    """Product cut out of a near-white background, padded to a square, SIDE x SIDE RGB in [0, 1]."""
    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    mask = (a < 0.92).any(axis=2)
    if mask.sum() > 50:
        ys, xs = np.where(mask)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape[:2]
    side = max(h, w)
    sq = np.ones((side, side, 3), dtype=np.float32)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = a
    return np.asarray(Image.fromarray((sq * 255).astype(np.uint8)).resize((SIDE, SIDE), Image.BILINEAR),
                      dtype=np.float32) / 255.0


def png(arr):
    """(SIDE, SIDE, 3) or (SIDE, SIDE) array in [0, 1] -> PNG data URI, scaled up for display."""
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((SIDE * 2, SIDE * 2), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def heat(err):
    """Residual map (SIDE, SIDE) -> dark-to-amber colour image (0 = matches the others)."""
    e = np.clip(err / (err.max() + 1e-6), 0, 1)[..., None]
    return (1 - e) * np.array([0.03, 0.08, 0.15]) + e * np.array([0.98, 0.70, 0.10])


def loo_residuals(X, k):
    """Residual of each row when rebuilt from the top-k PCA directions of all OTHER rows."""
    out = np.zeros_like(X)
    for i in range(len(X)):
        rest = np.delete(X, i, axis=0)
        mu = rest.mean(0)
        _, _, vt = np.linalg.svd(rest - mu, full_matrices=False)
        comp = vt[:min(k, len(vt))]
        z = X[i] - mu
        out[i] = z - (z @ comp.T) @ comp
    return out


def robust_z(v):
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-9
    return (v - med) / mad


def xray(images, dino=None, k=3):
    """images: list of PIL images (one model's listings). dino: optional (n, 384) DINOv2 vectors."""
    n = len(images)
    P = np.stack([cutout(im) for im in images])            # (n, S, S, 3)
    X = P.reshape(n, -1)
    mu = X.mean(0)
    _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    share = s ** 2 / (s ** 2).sum()
    eigen = []
    for c in range(min(3, len(vt))):
        v = vt[c].reshape(SIDE, SIDE, 3)
        eigen.append(dict(image=png(0.5 + 0.5 * v / (np.abs(v).max() + 1e-9)), share=round(float(share[c]), 3)))
    kk = min(k, n - 2) if n > 3 else 1
    R = loo_residuals(X, kk)                                 # pixel residuals, leave-one-out
    rebuilt = X - R
    pix_err = np.sqrt((R ** 2).mean(1))
    if dino is not None:
        D = np.asarray(dino, dtype=np.float64)
        sem_err = np.sqrt((loo_residuals(D, kk) ** 2).sum(1))
    else:
        sem_err = pix_err
    z = robust_z(sem_err)
    items = []
    for i in range(n):
        items.append(dict(photo=png(P[i]), rebuilt=png(rebuilt[i].reshape(SIDE, SIDE, 3)),
                          heat=png(heat(np.abs(R[i].reshape(SIDE, SIDE, 3)).mean(2))),
                          pixel_residual=round(float(pix_err[i]), 4), residual=round(float(sem_err[i]), 4),
                          z=round(float(z[i]), 2), odd=bool(n >= 5 and z[i] > 2.5)))
    return dict(n=n, k=kk, mean=png(mu.reshape(SIDE, SIDE, 3)), eigen=eigen,
                first_three_share=round(float(share[:3].sum()), 3), items=items)


# ---------------- injection test: does the X-ray find a photo of another product? ----------------

def injection_test(max_searches=60, seed=7, hard=False):
    """For each saved insights search: the model's main listing (>= 5 photos) plus ONE intruder photo: an offer from
    a different search (easy), or with hard=True a look-alike from the SAME search that the model kept out of the
    listing (another model of the same brand or line). Success = the intruder gets the highest odd-one-out score. Compares DINOv2 and pixel
    residuals; chance = 1 / (listing size + 1)."""
    from src.insights import CATEGORY, FOLDER, main_group, slug
    from src.live import download
    from src.service import Matcher
    rng = np.random.default_rng(seed)
    m = Matcher()
    dino_model = m.encoders()[2]
    searches = []
    for q in CATEGORY:
        p = FOLDER / f"{slug(q)}.json"
        if p.exists():
            searches.append((q, json.loads(p.read_text(encoding="utf-8"))))
    rows = []
    for qi, (q, res) in enumerate(searches[:max_searches]):
        g = main_group(res)
        if g is None:
            continue
        O, V = res["offers"], res["verdict"]
        model = [j for j in range(len(O)) if (j == g["seed"] or V[g["seed"]][j] in ("S", "V")) and not O[j].get("accessory")]
        if len(model) < 5:
            continue
        if hard:
            outside = [j for j in range(len(O)) if j not in model and not O[j].get("accessory")]
            if not outside:
                continue
            other, intruder = (q, res), O[outside[rng.integers(0, len(outside))]]
        else:
            other = searches[(qi + 1 + rng.integers(0, len(searches) - 1)) % len(searches)]
            intruder = other[1]["offers"][rng.integers(0, len(other[1]["offers"]))]
        urls = [O[j]["image"] for j in model[:20]] + [intruder["image"]]
        datas = [download(u) for u in urls]
        if any(d is None for d in datas):
            continue
        ims = [Image.open(io.BytesIO(d)).convert("RGB") for d in datas]
        out = xray(ims, dino=dino_model(ims))
        sem = np.array([it["residual"] for it in out["items"]])
        pix = np.array([it["pixel_residual"] for it in out["items"]])
        rows.append(dict(query=q, intruder_from=other[0], n=len(ims), dino_top1=bool(sem.argmax() == len(ims) - 1),
                         pixel_top1=bool(pix.argmax() == len(ims) - 1), dino_flagged=bool(out["items"][-1]["odd"]),
                         chance=round(1 / len(ims), 3)))
        print(q, "->", rows[-1]["dino_top1"], rows[-1]["pixel_top1"], flush=True)
    bs = lambda key: [round(float(np.mean([r[key] for r in [rows[i] for i in rng.integers(0, len(rows), len(rows))]])), 3)
                      for _ in range(config.N_BOOT)]
    ci = lambda key: [float(np.percentile(bs(key), 2.5)), float(np.percentile(bs(key), 97.5))]
    out = dict(searches=len(rows), dino_top1=round(float(np.mean([r["dino_top1"] for r in rows])), 3), dino_top1_ci=ci("dino_top1"),
               pixel_top1=round(float(np.mean([r["pixel_top1"] for r in rows])), 3), pixel_top1_ci=ci("pixel_top1"),
               dino_flagged_z_over_2_5=round(float(np.mean([r["dino_flagged"] for r in rows])), 3),
               chance_mean=round(float(np.mean([r["chance"] for r in rows])), 3),
               method=("one look-alike from the same search (kept out of the listing by the model)" if hard else "one photo of an offer from a different search") + " added to each main listing; success = it gets the "
                      "highest leave-one-out PCA residual; CIs resample searches", rows=rows)
    out["test"] = "hard (look-alike from the same search)" if hard else "easy (offer from a different search)"
    (config.RESULTS_DIR / "json" / f"photo_xray_test{'_hard' if hard else ''}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    if "--test" in sys.argv:
        injection_test(hard="--hard" in sys.argv)
