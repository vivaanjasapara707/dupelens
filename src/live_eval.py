"""Accuracy of live search on FRESH searches (not the Airdopes search the model-word rule was designed on).

--sheet [--test] : run the searches, sample offer pairs in three strata, write a blind sheet (titles only, no scores,
          shuffled) to results/live_eval_sheet.csv + photo grids results/labelling/live_*.jpg; the stratum and
          model verdict go to a hidden key cache/live_eval_key.csv.
          S = pairs the model calls "same"        -> precision of "same"
          R = random pairs from the same search    -> how often a naive "all results are the product" site is right
          H = rejected pairs with similar titles   -> how many true matches the model misses among look-alikes
--stats : after the `label` column is filled (same / variant / different), report each stratum with a
          bootstrap 95% CI. Labels are made by Claude, an AI assistant, not humans, blind to the key.
"""
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src import config
from src.live import download, search

# "tune": the first 8 searches; their labels were used to design the model-word rules, so their score is
# optimistic. "test": 8 fresh searches run only after the rules were frozen -> the number we report.
SETS = {
    "tune": ["Philips BHD308 hair dryer", "Milton Thermosteel Flip Lid 1000 ml", "Samsung Galaxy M35 5G 128GB",
             "Nivea Soft light moisturising cream 200 ml", "Logitech M235 wireless mouse", "Pigeon electric kettle 1.5 litre",
             "Prestige PIC 20 induction cooktop", "JBL Tune 510BT headphones"],
    "test": ["Bajaj DX 6 dry iron", "boAt Rockerz 450 headphones", "Dettol original liquid handwash 200 ml",
             "Redmi 13C 5G 128GB", "Lakme 9 to 5 primer matte lipstick", "Wildcraft 35 L backpack",
             "Butterfly mixer grinder 750 W", "Usha table fan 400 mm"],
    # "final": run once after the name-word rule was frozen (the "test" set was used to design it).
    # "round4": run once after the model-level view and rules v3 (hyphen codes, colours, packs, storage, name levels)
    # were frozen; adds stratum V (pairs called "other variant", shown in the listing) to measure same-MODEL precision.
    "round4": ["OnePlus Nord CE4 Lite 5G 128GB", "Samsung Galaxy S24 FE 128GB", "boAt Stone 350 speaker",
               "Philips HL7756 mixer grinder 750 W", "Maybelline Colossal kajal", "Head & Shoulders shampoo 340 ml",
               "Cello Opalware dinner set", "Skybags backpack 30 L"],
    "final": ["Prestige Popular pressure cooker 3 litre", "Philips BT3221 beard trimmer", "Mi Power Bank 3i 20000 mAh",
              "Boroplus antiseptic cream 80 ml", "Casio FX-991ES Plus calculator", "Sony WH-1000XM5 headphones",
              "Wipro 9W LED bulb", "Vaseline intensive care deep restore body lotion 400 ml"],
}
SET = next((k for k in ("round4", "final", "test") if f"--{k}" in sys.argv), "tune")
QUERIES = SETS[SET]
N = {"S": 40, "V": 30, "R": 30, "H": 30}
SHEET = config.RESULTS_DIR / ("live_eval_sheet.csv" if SET == "tune" else f"live_{SET}_sheet.csv")
KEY = config.CACHE_DIR / ("live_eval_key.csv" if SET == "tune" else f"live_{SET}_key.csv")


def make_sheet():
    from src.service import Matcher
    m, rng, rows = Matcher(), np.random.default_rng(config.SEED), []
    for q in QUERIES:
        res = search(m, q)
        O, V = res["offers"], np.array(res["verdict"])
        tf = np.array(res["features"]["tfidf_cos"])
        a, b = np.triu_indices(len(O), 1)
        for i, j in zip(a, b):
            rows.append(dict(query=q, a_title=O[i]["title"], b_title=O[j]["title"], a_store=O[i]["store"], b_store=O[j]["store"],
                             a_image=O[i]["image"], b_image=O[j]["image"], verdict=V[i, j], tfidf=tf[i, j]))
        print(f"{q}: {len(O)} offers, {int((V[a, b] == 'S').sum())} same pairs")
    pool = pd.DataFrame(rows)
    picks = [pool[pool.verdict == "S"].sample(min(N["S"], (pool.verdict == "S").sum()), random_state=1).assign(stratum="S")]
    rest = pool.drop(picks[0].index)
    if SET == "round4":  # pairs shown as "other variant" of the same model
        picks.append(rest[rest.verdict == "V"].sample(min(N["V"], (rest.verdict == "V").sum()), random_state=5).assign(stratum="V"))
        rest = rest.drop(picks[-1].index)
        pool.groupby("query").verdict.value_counts().unstack(fill_value=0).to_csv(config.RESULTS_DIR / f"live_{SET}_counts.csv")
    picks.append(rest.sample(N["R"], random_state=2).assign(stratum="R"))
    rest = rest.drop(picks[-1].index)
    hard = rest[(rest.verdict != "S") & (rest.tfidf >= 0.5)]
    picks.append(hard.sample(min(N["H"], len(hard)), random_state=3).assign(stratum="H"))
    sheet = pd.concat(picks).sample(frac=1, random_state=4).reset_index(drop=True)
    sheet["pair_id"] = [f"L{k:03d}" for k in range(len(sheet))]
    sheet[["pair_id", "stratum", "verdict", "tfidf", "query"]].to_csv(KEY, index=False)
    sheet[["pair_id", "a_title", "b_title", "a_store", "b_store"]].assign(label="").to_csv(SHEET, index=False)
    grids(sheet)
    print("pairs:", len(sheet), sheet.stratum.value_counts().to_dict(), "| sheet:", SHEET)


def grids(sheet, per=10, size=180):
    """Photo grids: each line = pair id, photo A, photo B (titles are in the sheet)."""
    out = config.RESULTS_DIR / "labelling"
    out.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(sheet), per):
        part = sheet.iloc[start:start + per]
        canvas = Image.new("RGB", (2 * size + 90, per * (size + 10)), "white")
        draw = ImageDraw.Draw(canvas)
        for k, r in enumerate(part.itertuples()):
            y = k * (size + 10)
            draw.text((5, y + size // 2), r.pair_id, fill="black")
            for c, url in enumerate((r.a_image, r.b_image)):
                data = download(url)
                if data:
                    import io
                    im = Image.open(io.BytesIO(data)).convert("RGB")
                    im.thumbnail((size, size))
                    canvas.paste(im, (80 + c * (size + 5), y))
        canvas.save(out / f"live_{SET}_{start // per:02d}.jpg", quality=85)


def search_bootstrap(df, hit, rng, n_boot=config.N_BOOT):
    """Rate of `hit` over the pairs in df, with a 95% CI that resamples whole SEARCHES (not pairs).

    Pairs from one search share offers and photos, so they are not independent; resampling the 8
    searches with replacement and pooling their pairs gives an honest (wider) interval."""
    groups = [g for _, g in df.groupby("query")]
    if not groups:
        return None
    rates = []
    for _ in range(n_boot):
        pick = [groups[k] for k in rng.integers(0, len(groups), len(groups))]
        n = sum(len(g) for g in pick)
        rates.append(sum(hit(g).sum() for g in pick) / n if n else np.nan)
    y = hit(df)
    return dict(n=len(df), hits=int(y.sum()), rate=round(float(y.mean()), 3), searches=len(groups),
                ci_search_bootstrap=[round(float(np.nanpercentile(rates, 2.5)), 3), round(float(np.nanpercentile(rates, 97.5)), 3)])


def stats():
    lab = pd.read_csv(SHEET).merge(pd.read_csv(KEY), on="pair_id")
    assert lab["label"].isin(["same", "variant", "different"]).all(), "fill every label first"
    rng, out = np.random.default_rng(config.SEED), {"set": SET, "labels": "Claude, an AI assistant, not humans"}
    same = lambda g: (g["label"] == "same").to_numpy(dtype=float)
    line = lambda g: g["label"].isin(["same", "variant"]).to_numpy(dtype=float)
    for key, s, hit, name in [("S_same", "S", same, "model says same -> truly the identical item"),
                              ("S_same_or_variant", "S", line, "model says same -> at least the same product line"),
                              ("R_same", "R", same, "random pair from a search -> identical item (naive baseline)"),
                              ("V_same_model", "V", line, "model says other variant -> at least the same product line"),
                              ("V_same", "V", same, "model says other variant -> actually the identical item"),
                              ("H_same", "H", same, "rejected look-alike -> actually identical (missed)"),
                              ("H_same_model", "H", line, "rejected look-alike -> actually the same product line (missed)")]:
        r = search_bootstrap(lab[lab.stratum == s], hit, rng)
        if r:
            out[key] = dict(what=name, **r)
            print(f"{key:<18} {r['hits']}/{r['n']} = {r['rate']:.3f}  CI (search bootstrap) {r['ci_search_bootstrap']}  - {name}")
    import json
    (config.RESULTS_DIR / "json" / f"live_eval_{SET}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    make_sheet() if "--sheet" in sys.argv else stats()
