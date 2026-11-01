"""Ablation on the final live round: trained model alone (LR) vs model + live-search rules (LR+rules).

"LR" = M1_v2 probability with the v2 verdict rules it was calibrated with (number conflict, combo, colour cut).
"LR+rules" adds the live-search rules in src/verdict.py (model words/codes, accessories, sizes, name words).
LR+rules "same" pairs are a subset of LR "same" pairs, so

    precision(LR) = [N_kept * p_kept + N_removed * p_removed] / (N_kept + N_removed)

N_kept / N_removed = how many pairs in the 8 final searches each group holds (counted exactly);
p_kept = the S stratum of the final round (already labelled);
p_removed = a NEW blind sample of pairs the rules removed, shuffled with decoy pairs so the labeller
cannot tell them apart (labels by Claude, an AI assistant, not humans).
CIs resample whole searches.

--sheet : results/ablation_sheet.csv + photo grids results/labelling/ablation_*.jpg, key cache/ablation_key.csv
--stats : results/json/ablation_final.json
"""
import json
import sys

import numpy as np
import pandas as pd

from src import config
from src.live import search
from src.live_eval import SETS, grids

SHEET = config.RESULTS_DIR / "ablation_sheet.csv"
KEY = config.CACHE_DIR / "ablation_key.csv"


def pools():
    """Every pair of every final search with its LR and LR+rules verdicts."""
    from src.service import Matcher
    m, rows = Matcher(), []
    for q in SETS["final"]:
        r = search(m, q)
        O, V, VL = r["offers"], np.array(r["verdict"]), np.array(r["verdict_lr"])
        a, b = np.triu_indices(len(O), 1)
        for i, j in zip(a, b):
            rows.append(dict(query=q, a_title=O[i]["title"], b_title=O[j]["title"], a_store=O[i]["store"],
                             b_store=O[j]["store"], a_image=O[i]["image"], b_image=O[j]["image"],
                             rules=V[i, j], lr=VL[i, j]))
    return pd.DataFrame(rows)


def make_sheet(n_removed=40, n_decoy=20):
    d = pools()
    removed = d[(d.lr == "S") & (d.rules != "S")]
    decoy = d[d.lr != "S"]
    pick = pd.concat([removed.sample(min(n_removed, len(removed)), random_state=11).assign(group="removed"),
                      decoy.sample(n_decoy, random_state=12).assign(group="decoy")]).sample(frac=1, random_state=13)
    pick = pick.reset_index(drop=True)
    pick["pair_id"] = [f"A{k:03d}" for k in range(len(pick))]
    pick[["pair_id", "group", "query", "rules", "lr"]].to_csv(KEY, index=False)
    pick[["pair_id", "a_title", "b_title", "a_store", "b_store"]].assign(label="").to_csv(SHEET, index=False)
    import src.live_eval as le
    le.SET = "ablation"
    grids(pick)
    counts = d.groupby("query").apply(lambda g: pd.Series(dict(lr_same=int((g.lr == "S").sum()),
                                                               kept=int((g.rules == "S").sum())))).reset_index()
    counts.to_csv(config.RESULTS_DIR / "ablation_counts.csv", index=False)
    print("pairs to label:", len(pick), "| removed pool:", len(removed), "| counts per search -> results/ablation_counts.csv")


def stats():
    counts = pd.read_csv(config.RESULTS_DIR / "ablation_counts.csv")
    counts["removed"] = counts.lr_same - counts.kept
    final = pd.read_csv(config.RESULTS_DIR / "live_final_sheet.csv").merge(pd.read_csv(config.CACHE_DIR / "live_final_key.csv"), on="pair_id")
    kept = final[final.stratum == "S"][["query", "label"]]
    abl = pd.read_csv(SHEET).merge(pd.read_csv(KEY), on="pair_id")
    assert abl.label.isin(["same", "variant", "different"]).all(), "fill every label first"
    removed = abl[abl.group == "removed"][["query", "label"]]

    def estimate(qs):
        c = counts.set_index("query").loc[qs]
        # a search drawn twice counts twice, so its labelled pairs are repeated too
        k = pd.concat([kept[kept["query"] == q] for q in qs])
        r = pd.concat([removed[removed["query"] == q] for q in qs])
        p_k = (k.label == "same").mean() if len(k) else np.nan
        p_r = (r.label == "same").mean() if len(r) else 0.0
        n_k, n_r = c.kept.sum(), c.removed.sum()
        lr = (n_k * p_k + n_r * p_r) / (n_k + n_r) if n_k + n_r else np.nan
        return p_k, lr, p_r

    qs = list(counts["query"])
    p_k, lr, p_r = estimate(qs)
    rng, boot = np.random.default_rng(config.SEED), []
    for _ in range(config.N_BOOT):
        boot.append(estimate(list(rng.choice(qs, len(qs)))))
    boot = np.array(boot, dtype=float)
    ci = lambda col: [round(float(np.nanpercentile(boot[:, col], 2.5)), 3), round(float(np.nanpercentile(boot[:, col], 97.5)), 3)]
    out = dict(labels="Claude, an AI assistant, not humans (blind; removed pairs mixed with decoys)",
               pairs_called_same=dict(lr=int(counts.lr_same.sum()), lr_plus_rules=int(counts.kept.sum())),
               precision_identical=dict(lr=round(float(lr), 3), lr_ci=ci(1), lr_plus_rules=round(float(p_k), 3), lr_plus_rules_ci=ci(0)),
               removed_by_rules_but_identical=dict(labelled=int(len(removed)), identical=int((removed.label == "same").sum()),
                                                   rate=round(float(p_r), 3), ci=ci(2)),
               ci_method="bootstrap over the 8 final searches")
    (config.RESULTS_DIR / "json" / "ablation_final.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    make_sheet() if "--sheet" in sys.argv else stats()
