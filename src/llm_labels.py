"""Second, independent labeller for the final live round: a non-Anthropic vision model.

Each pair from results/live_final_sheet.csv is sent with ONLY the two titles, the two store names and
the two product photos (no model output, no Claude labels) to an OpenAI-compatible chat API.
Default: Groq, model qwen/qwen3.8-27b (accepts images; checked on console.groq.com/docs/vision, 2026-11-01).
Env vars: GROQ_API_KEY (or LABEL_API_KEY), optional LABEL_API_BASE, LABEL_MODEL.

--label : results/llm_labels_final.csv (resumable: pairs already labelled are skipped)
--stats : Cohen's kappa vs the Claude labels (3-class and same-vs-not), and precision of the model's
          "same" verdicts on pairs where both labellers agree -> results/json/llm_agreement_final.json
"""
import base64
import json
import sys
import time

import pandas as pd
import requests

from src import config
from src.live import download, search

API_BASE = config.env("LABEL_API_BASE") or "https://api.groq.com/openai/v1"
MODEL = config.env("LABEL_MODEL") or "qwen/qwen3.8-27b"
OUT = config.RESULTS_DIR / "llm_labels_final.csv"

PROMPT = """You compare two online shopping listings from India and decide whether they sell the same item.
Answer with exactly one label:
- "same": the same product model AND the same colour/shade/scent, size, pack count and bundle contents.
- "variant": the same product model or product line, but a different colour/shade/scent, size, pack count,
  bundle or combo.
- "different": a different product (another model, another product line, an accessory or spare part for it,
  or a different kind of thing).
Use both the titles and the photos. Titles may be in Hindi or English.
Listing A (store: {a_store}): {a_title}
Listing B (store: {b_store}): {b_title}
The first image is listing A, the second is listing B.
Reply as JSON: {{"label": "same" | "variant" | "different", "reason": "<one short sentence>"}}"""


def image_part(url):
    data = download(url)
    if not data:
        return None
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode()}}


def ask(pair, key):
    content = [{"type": "text", "text": PROMPT.format(**pair)}]
    content += [p for p in (image_part(pair["a_image"]), image_part(pair["b_image"])) if p]
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "temperature": 0,
            "response_format": {"type": "json_object"}, "max_completion_tokens": 300}
    for attempt in range(6):  # free tiers rate-limit; wait and retry
        r = requests.post(f"{API_BASE}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=90)
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        reply = json.loads(r.json()["choices"][0]["message"]["content"])
        label = str(reply.get("label", "")).strip().lower()
        return label if label in ("same", "variant", "different") else "unparsed", reply.get("reason", "")
    raise RuntimeError("rate limited too many times")


def pairs_with_images():
    """Final-round pairs with their photo addresses, rebuilt from the saved searches (query only from the key)."""
    from src.service import Matcher
    sheet = pd.read_csv(config.RESULTS_DIR / "live_final_sheet.csv").drop(columns=["label"])
    query = pd.read_csv(config.CACHE_DIR / "live_final_key.csv")[["pair_id", "query"]]
    sheet = sheet.merge(query, on="pair_id")
    m, images = Matcher(), {}
    for q in sheet["query"].unique():
        for o in search(m, q)["offers"]:
            images.setdefault((q, o["title"]), o["image"])
    sheet["a_image"] = [images[(q, t)] for q, t in zip(sheet["query"], sheet.a_title)]
    sheet["b_image"] = [images[(q, t)] for q, t in zip(sheet["query"], sheet.b_title)]
    return sheet


def label():
    key = config.env("LABEL_API_KEY") or config.env("GROQ_API_KEY")
    if not key:
        raise SystemExit("Set GROQ_API_KEY (or LABEL_API_KEY) first.")
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["pair_id", "llm_label", "llm_reason", "model"])
    rows = done.to_dict("records")
    for p in pairs_with_images().to_dict("records"):
        if p["pair_id"] in set(done.pair_id):
            continue
        lab, why = ask(p, key)
        rows.append(dict(pair_id=p["pair_id"], llm_label=lab, llm_reason=why, model=MODEL))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(p["pair_id"], lab, "|", why[:80], flush=True)


def stats():
    from sklearn.metrics import cohen_kappa_score
    claude = pd.read_csv(config.RESULTS_DIR / "live_final_sheet.csv")[["pair_id", "label"]]
    key = pd.read_csv(config.CACHE_DIR / "live_final_key.csv")[["pair_id", "stratum", "query"]]
    d = pd.read_csv(OUT).merge(claude, on="pair_id").merge(key, on="pair_id")
    d = d[d.llm_label.isin(["same", "variant", "different"])]
    agree = d[d.llm_label == d.label]
    s_agree = agree[agree.stratum == "S"]
    out = dict(model=str(d.model.iloc[0]), pairs=len(d),
               kappa_3class=round(float(cohen_kappa_score(d.label, d.llm_label)), 3),
               kappa_same_vs_not=round(float(cohen_kappa_score(d.label == "same", d.llm_label == "same")), 3),
               raw_agreement=f"{len(agree)}/{len(d)}",
               confusion=pd.crosstab(d.label, d.llm_label).to_dict(),
               model_same_precision_llm_labels=f"{int((d[d.stratum == 'S'].llm_label == 'same').sum())}/{int((d.stratum == 'S').sum())}",
               model_same_precision_on_agreed_pairs=f"{int((s_agree.label == 'same').sum())}/{len(s_agree)}")
    (config.RESULTS_DIR / "json" / "llm_agreement_final.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    label() if "--label" in sys.argv else stats()
