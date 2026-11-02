"""Export the data for notebooks/pca_dupelens.ipynb (run once; the notebook then needs no models or internet).

1. notebooks/data/pair_features.csv: the 12 named pair features the matcher's logistic regression uses, for every
   labelled pair of the live evaluation rounds (labelled by Claude, an AI assistant, not humans).
2. notebooks/data/genuine_demo.csv: DINOv2 photo vectors (384 numbers) of the "iphone 16" listings plus one photo of
   another product (Sony WH-1000XM5 headphones) slipped in, for the Genuine Look demo.
Run: python -m src.pca_export
"""
import numpy as np
import pandas as pd

from src import config
from src.live import search
from src.service import Matcher

OUT = config.ROOT / "notebooks" / "data"
FEATURES = ["img_cos", "txt_cos", "tfidf_cos", "phash_dist", "num_match", "num_conflict", "mutual", "rank_min",
            "rank_max", "len_ratio", "dino_cos", "colour_dist"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m, rows, cache = Matcher(), [], {}
    for name in ("eval", "test", "final", "round4"):
        lab = pd.read_csv(config.RESULTS_DIR / f"live_{name}_sheet.csv").merge(
            pd.read_csv(config.CACHE_DIR / f"live_{name}_key.csv"), on="pair_id")
        for q, part in lab.groupby("query"):
            if q not in cache:
                cache[q] = search(m, q, vectors=True)
            r = cache[q]
            X = r["pair_features"]
            idx = {o["title"]: i for i, o in reversed(list(enumerate(r["offers"])))}
            for p in part.itertuples():
                i, j = idx.get(p.a_title), idx.get(p.b_title)
                if i is None or j is None or i == j:
                    continue
                a, b = min(i, j), max(i, j)
                hit = X[(X["a"] == a) & (X["b"] == b)]
                if len(hit):
                    rows.append({"round": name, "query": q, "a_title": p.a_title, "b_title": p.b_title,
                                 **{f: round(float(hit.iloc[0][f]), 5) for f in FEATURES}, "label": p.label})
    df = pd.DataFrame(rows).drop_duplicates(subset=["query", "a_title", "b_title"])
    df.to_csv(OUT / "pair_features.csv", index=False)
    print("pairs:", len(df), df["label"].value_counts().to_dict())

    r = search(m, "iphone 16", vectors=True)
    keep = [i for i, o in enumerate(r["offers"]) if o.get("match") == "exact" and i < len(r["dino"])]
    D = pd.DataFrame([r["dino"][i] for i in keep], columns=[f"d{k}" for k in range(384)])
    D.insert(0, "store", [r["offers"][i]["store"] for i in keep])
    D.insert(1, "title", [r["offers"][i]["title"] for i in keep])
    s = search(m, "sony wh-1000xm5", vectors=True)
    j = next(i for i, o in enumerate(s["offers"]) if o.get("match") == "exact" and i < len(s["dino"]))
    fake = pd.DataFrame([s["dino"][j]], columns=[f"d{k}" for k in range(384)])
    fake.insert(0, "store", "INJECTED (photo of another product)")
    fake.insert(1, "title", "Apple iPhone 16 (photo is really: " + s["offers"][j]["title"][:50] + ")")
    pd.concat([D, fake], ignore_index=True).to_csv(OUT / "genuine_demo.csv", index=False)
    print("genuine demo rows:", len(D) + 1)


if __name__ == "__main__":
    main()
