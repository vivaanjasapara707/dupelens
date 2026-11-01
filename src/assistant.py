"""Ask DupeLens: answers shopping questions about the CURRENT search with a Groq-hosted LLM.

The page sends the data it is showing (exact-item sellers, prices, ratings, seller scores, variants,
cheapest price per store) as JSON; the LLM must answer from that data only. Numbers such as
"Amazon vs Flipkart" are computed by the page, not by the LLM, so the model only has to read them.

Needs a free Groq key (console.groq.com -> API Keys) in .env: GROQ_API_KEY=...
Optional: GROQ_MODEL (default openai/gpt-oss-120b).
"""
import json

import requests

from src import config

URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_CONTEXT_CHARS = 24000  # keeps each request small and cheap

SYSTEM = """You are the shopping assistant inside DupeLens, a site that finds every online store in India selling
the exact same product and tells the buyer where to buy it.

Rules:
- Answer ONLY from the search data given below (JSON). Never invent stores, prices, ratings, delivery times or links.
- Every number you mention must appear in the data. If the data cannot answer the question, say so and suggest
  what the user could check (for example, open the store page).
- Prices are what Google Shopping showed at search time; say so when it matters.
- "exact_item_sellers" are listings the DupeLens model judged to be the identical item. "variants" are the same
  product in another colour, size or pack. "look_alikes" are other products. Only compare prices of exact-item
  sellers, unless the user asks about variants.
- To compare stores (for example Amazon vs Flipkart), use "cheapest_by_store" and "store_comparisons".
- For "which should I buy", start from "recommended" and explain with its score parts; mention any
  "suspiciously cheap" flag (a price far below the fair price can mean a fake, used or wrong item).
- Plain, friendly English for a student. Use Rs or the rupee sign for prices. At most about 120 words unless asked
  for more. No tables.
- If the question is not about shopping for this product, say briefly that you can only help with this search."""


def build_messages(question, context, history=()):
    """System rules + the search data + earlier turns + the new question."""
    data = json.dumps(context, ensure_ascii=False)[:MAX_CONTEXT_CHARS]
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Search data (JSON):\n{data}"},
                {"role": "assistant", "content": "I have the search data and will answer only from it."}]
    for turn in list(history)[-6:]:  # the last few turns, so follow-up questions work
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": str(turn["content"])[:2000]})
    messages.append({"role": "user", "content": question})
    return messages


def answer(question, context, history=()):
    key = config.env("GROQ_API_KEY")
    if not key:
        raise RuntimeError("No Groq key: put GROQ_API_KEY=... in .env (free at console.groq.com)")
    model = config.env("GROQ_MODEL") or DEFAULT_MODEL
    body = {"model": model, "messages": build_messages(question, context, history), "temperature": 0.2,
            "max_completion_tokens": 1500}
    if model.startswith("openai/gpt-oss"):  # reasoning model: think briefly, return only the answer
        body.update(reasoning_effort="low", include_reasoning=False)
    r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=60)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"].get("content") or ""
    return {"answer": text.strip() or "Sorry, I could not produce an answer. Please ask again.", "model": model}
