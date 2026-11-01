"""Calibrate M1_v2's probability on the Shopee VALIDATION split (Platt scaling and isotonic regression).

"Calibrated" means: of all pairs the model gives 0.8, about 80% are truly the same product.
1. Embed only the validation listings (5,045 photos + titles; cached as cache/shopee_val_*.npy), because
   the full Shopee cache is not on this machine. Needs the Kaggle data in data/ (see HANDOFF.md section 0).
2. Rebuild the validation candidate pairs and features exactly as src/train_v2.py does; score with the
   saved M1_v2 (no retraining, test split untouched).
3. Cross-fitted calibration: 5 folds grouped by product group, so each pair is calibrated by a map fitted
   on other products. Report Brier score, log loss and expected calibration error (ECE, 10 bins) for raw,
   Platt and isotonic; save a reliability plot. Fit the chosen map on all validation pairs and save it.
Outputs: results/json/calibration_v2.json, results/plots/reliability_m1_v2.png, cache/calibrator_v2.joblib
Run: python -m src.calibrate
"""
import faiss  # noqa: F401  import first (OpenMP clash with torch, see config.py)
import json

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

from src import config, features
from src.data import load_split
from src.train import prepare
from src.train_v2 import add_image_features

OUT_MAP = config.CACHE_DIR / "calibrator_v2.joblib"


def embed_val(val):
    """CLIP, MPNet, phash, DINOv2 and colour for the validation listings only (cached)."""
    names = ["img", "txt", "phash", "dino", "colour"]
    paths = {n: config.CACHE_DIR / f"shopee_val_{n}.npy" for n in names}
    if all(p.exists() for p in paths.values()):
        return {n: np.load(p) for n, p in paths.items()}
    from concurrent.futures import ThreadPoolExecutor
    from sentence_transformers import SentenceTransformer
    from src.embed import IMAGE_MODEL, TEXT_MODEL, embed_images, encode, get_device, load_image
    from src.extra import Dino, colour_hist
    device = get_device()
    files = [str(config.TRAIN_IMAGES / f) for f in val["image"]]
    img, phash = embed_images(files, SentenceTransformer(IMAGE_MODEL, device=device))
    txt = encode(SentenceTransformer(TEXT_MODEL, device=device), val["title"].tolist(), batch_size=64)
    dino_model, dino, colour = Dino(device), [], []
    with ThreadPoolExecutor(8) as pool:
        for start in range(0, len(files), 128):
            ims = [im for im, _ in pool.map(load_image, files[start:start + 128])]
            dino.append(dino_model(ims))
            colour.extend(pool.map(colour_hist, ims))
    out = dict(img=img, txt=txt.astype(np.float32), phash=phash, dino=np.vstack(dino), colour=np.vstack(colour))
    for n, p in paths.items():
        np.save(p, out[n])
    return out


def ece(y, p, bins=10):
    """Expected calibration error: average |observed rate - mean probability| over equal-width bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == k].mean() - p[idx == k].mean()) * (idx == k).mean() for k in range(bins) if (idx == k).any()))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def main():
    cfg = json.loads((config.CACHE_DIR / "final_config_v2.json").read_text())
    s = (cfg["pca_variant"], cfg["dim"])
    df = load_split()
    val = df[df["split"] == "val"].reset_index(drop=True)
    emb = embed_val(val)
    pcas = joblib.load(config.CACHE_DIR / "pca.joblib")
    d = add_image_features(prepare(val, emb, pcas, features.load_tfidf(), "val", s, s), emb["dino"], emb["colour"])
    model = joblib.load(config.CACHE_DIR / "models_v2.joblib")["m1_v2"]
    p = model.predict_proba(d["X"][cfg["features_v2"]])[:, 1]
    y = d["y"].astype(int)
    groups = d["labels"][d["a"]]

    platt, iso = np.zeros_like(p), np.zeros_like(p)
    for tr, te in GroupKFold(5).split(p, y, groups):
        platt[te] = LogisticRegression().fit(logit(p[tr]), y[tr]).predict_proba(logit(p[te]))[:, 1]
        iso[te] = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p[tr], y[tr]).predict(p[te])
    rows = {name: dict(brier=round(brier_score_loss(y, q), 4), log_loss=round(log_loss(y, np.clip(q, 1e-6, 1 - 1e-6)), 4),
                       ece=round(ece(y, q), 4)) for name, q in [("raw", p), ("platt", platt), ("isotonic", iso)]}
    best = min(("platt", "isotonic"), key=lambda k: rows[k]["ece"])
    final = (LogisticRegression().fit(logit(p), y) if best == "platt"
             else IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y))
    joblib.dump(dict(kind=best, model=final), OUT_MAP)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for name, q in [("raw", p), ("Platt", platt), ("isotonic", iso)]:
        edges = np.linspace(0, 1, 11)
        idx = np.minimum((q * 10).astype(int), 9)
        xs = [q[idx == k].mean() for k in range(10) if (idx == k).sum() >= 20]
        ys = [y[idx == k].mean() for k in range(10) if (idx == k).sum() >= 20]
        ax.plot(xs, ys, marker="o", label=f"{name} (ECE {rows[name.lower()]['ece']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.set(xlabel="predicted probability (same product)", ylabel="observed share that is the same product",
           title=f"M1_v2 reliability, Shopee validation ({len(p):,} pairs)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "plots" / "reliability_m1_v2.png", dpi=150)

    out = dict(split="Shopee validation (cross-fitted, 5 folds grouped by product group)", pairs=len(p),
               positives=int(y.sum()), metrics=rows, chosen=best, note="the matching threshold still uses the raw probability")
    (config.RESULTS_DIR / "json" / "calibration_v2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
