"""Identifier-based labels for live-search pairs: an independent check on the Claude labels.

Serper Shopping offers carry no GTIN / EAN / UPC / MPN / ASIN field. They do carry Google's productId,
but it is (almost) unique per offer, so it cannot link one product across stores (counted below).
The usable identifier is a manufacturer model code written in the title (BT3221/15, WH-1000XM5, FX-991ES).

Label rule for a pair, only when BOTH titles contain a model code:
  same_model       the titles share a model code (colour / pack may still differ)
  different_model  the titles' model codes share nothing
Compared with the Claude labels on the same pairs: Claude same or variant = same model.
Run: python -m src.identifiers   -> results/json/identifiers.json, results/identifier_labels.csv
"""
import glob
import json
import re
from collections import Counter

import pandas as pd

from src import config
from src.verdict import SPEC

# A model code: letters and digits mixed, optionally joined by '-' (WH-1000XM5, FX-991ES, BT3221/15 -> BT3221).
CODE = re.compile(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b")


def model_codes(title):
    out = set()
    for tok in CODE.findall(title.lower()):
        tok = tok.split("/")[0]
        flat = tok.replace("-", "")
        if (re.search(r"[a-z]", flat) and re.search(r"\d", flat) and len(flat) >= 4 and not SPEC.match(flat)
                and not re.fullmatch(r"\d+(g|gb|mp|hz|k|w|v|x)\d*", flat)):
            out.add(flat)
    return out


def coverage():
    """Which identifier fields the cached Serper offers have, and how often productId repeats."""
    offers = [o for f in glob.glob(str(config.CACHE_DIR / "live" / "shopping_*.json"))
              for o in json.load(open(f, encoding="utf-8")).get("shopping", [])]
    keys = Counter(k for o in offers for k in o)
    pid = Counter(o.get("productId") for o in offers if o.get("productId"))
    with_code = sum(bool(model_codes(o["title"])) for o in offers)
    return dict(offers=len(offers), fields={k: v for k, v in keys.most_common()},
                gtin_ean_upc_mpn_asin_fields=sum(keys[k] for k in ("gtin", "ean", "upc", "mpn", "asin")),
                distinct_product_ids=len(pid), product_ids_shared_by_2_or_more_offers=sum(v > 1 for v in pid.values()),
                offers_with_model_code_in_title=with_code)


def label_final():
    d = pd.read_csv(config.RESULTS_DIR / "live_final_sheet.csv").merge(
        pd.read_csv(config.CACHE_DIR / "live_final_key.csv"), on="pair_id")
    a, b = d.a_title.map(model_codes), d.b_title.map(model_codes)
    both = (a.map(bool) & b.map(bool))
    d["id_label"] = [("same_model" if x & y else "different_model") if ok else "" for x, y, ok in zip(a, b, both)]
    d["claude_model_level"] = d.label.map({"same": "same_model", "variant": "same_model", "different": "different_model"})
    d[["pair_id", "query", "stratum", "a_title", "b_title", "id_label", "label"]].to_csv(
        config.RESULTS_DIR / "identifier_labels.csv", index=False)
    lab = d[both]
    agree = (lab.id_label == lab.claude_model_level)
    return dict(pairs=len(d), pairs_with_code_in_both_titles=int(both.sum()),
                id_label_counts=lab.id_label.value_counts().to_dict(),
                agreement_with_claude_model_level=f"{int(agree.sum())}/{len(lab)}",
                disagreements=lab[~agree][["pair_id", "a_title", "b_title", "id_label", "label"]].to_dict("records"))


if __name__ == "__main__":
    out = dict(coverage=coverage(), final_round=label_final())
    (config.RESULTS_DIR / "json" / "identifiers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
