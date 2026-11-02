"""Ask DupeLens: answers shopping questions about the CURRENT search with a Groq-hosted LLM.

The page sends the data it is showing (exact-item sellers, prices, ratings, seller scores, variants,
cheapest price per store) as JSON. Prices, stores and links come from that data only; numbers such as
"Amazon vs Flipkart" are computed by the page, not by the LLM, so the model only has to read them. General
questions (specs, "is it good for gaming") may use general knowledge, labelled as such.
The reply is JSON: {"answer": text, "action": {...} or null}. The action changes what the page shows
(keep some stores, a price range, words the title must have, big stores only, new only, sort order, a new
search, or clear): the page applies it, so the list and the best price follow what the user asked.

Needs a free Groq key (console.groq.com -> API Keys) in .env: GROQ_API_KEY=...
Optional: GROQ_MODEL (default openai/gpt-oss-120b).
"""
import json
import re
import time

import requests

from src import config

URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"   # its own rate limit: used when the main model is busy (HTTP 429)
MAX_CONTEXT_CHARS = 10000  # about 3,000 tokens: the free Groq tier allows 8,000 tokens a minute per model

SYSTEM = """You are the shopping assistant inside DupeLens, a site that finds every online store in India selling
the exact same product and tells the buyer where to buy it.

Rules:
- Stores, prices, ratings, delivery times and links come ONLY from the search data below (JSON). Never invent them;
  every price you mention must appear in the data.
- You can answer any question. About this search: from the data. General questions (specs, battery, camera, is it
  good for gaming, how it compares with another model, what to look out for): answer from general knowledge and
  start with "In general," so the user knows it is not from this search. State only specs you are sure of for this
  exact model (do not mix it up with its Pro / Plus / older versions); if unsure, say so and suggest the brand's
  spec page.
- Prices are what Google Shopping showed at search time; say so when it matters.
- "listings_of_this_model" are listings the DupeLens model judged to be the same model, big stores first; each has
  "same_item_as_top_pick" (true = likely the identical item; false = another colour, size or pack) and the
  "attributes" read from its title (storage, colour, size, pack). The model is right more often than not, not
  always. "look_alikes" are other products. Compare prices only between listings with the same attributes, and
  say which variant a price is for.
- To compare stores (for example Amazon vs Flipkart), use "cheapest_by_store" and "store_comparisons".
- Each listing has "store_type" (big, authorised, brand = trusted; other = less-known shop) and "condition"
  (new or refurbished = refurbished / used / graded). Never recommend a less-known shop or a refurbished listing
  as the main choice; mention them only as riskier options. If "best_price" is null, say no trusted store sells
  it new in these results and suggest checking Amazon, Flipkart or Croma directly.
- "genuine_look_pct" (less-known shops only; PCA: how much the photo looks like the real product) and
  "risky_small_shop" answer "is this fake / which look fake?". Trusted stores (big, authorised, brand) are never
  called risky or fake; a trusted listing far below the fair price is "probably another variant, size or pack".
- For "which should I buy", start from "best_price" (the cheapest listing from a big store that is not
  suspiciously cheap) and say why; mention any "suspiciously cheap" flag (a price far below the typical price
  can mean a fake, used or wrong item) and any cheaper listing from a less-known store as a riskier option.
- Plain, friendly English for a student. Use Rs or the rupee sign for prices. At most about 120 words unless asked
  for more. No tables.
- "active_filters" are the filters already applied to the list (by earlier questions); the data shows the
  filtered list. "all_stores_in_search" lists every store in the search, filtered or not.

Reply with a JSON object only: {"answer": "<your answer>", "action": <action or null>}.
Set "action" whenever the question says what the user wants to see or prefers, even indirectly, so the list on
the page follows the conversation. Examples:
  "which is the safest?" / "I only trust big sites"   -> {"big_stores_only": true, "new_only": true, "sort": "score"}
  "cheapest?" / "lowest price"                        -> {"sort": "price"}
  "best reviewed?" / "most popular"                   -> {"sort": "rating"}
  "what colours does it come in?"                     -> answer from "colours_available" (no action, unless one colour
                                                         is wanted: "show the black one" -> {"must_include": ["black"]})
  "only 256 GB" / "the 1 kg pack"                     -> {"must_include": ["256gb"]} / {"must_include": ["1kg"]}
  "under 50k" / "between 1000 and 2000"               -> {"max_price": 50000} / {"min_price": 1000, "max_price": 2000}
  "only Amazon" / "not Meesho"                        -> {"stores": ["Amazon"]} / {"exclude_stores": ["Meesho"]}
  "show me the Pro version" / "compare with S24"      -> {"new_search": "<that product>"}
  "show everything" / "reset"                         -> {"reset": true}
Questions about safety, trust, cheapest, best rated or a variant ALWAYS get the action above, so the list shows
the answer. Only plain information questions ("is it good for gaming?", "why is this first?") need no action.
The action is an object with any of these keys (leave out the rest):
  "stores": ["Amazon", "Flipkart"]      keep only these stores (names as in all_stores_in_search)
  "exclude_stores": ["Meesho"]          hide these stores
  "max_price": 70000, "min_price": 1000 rupees
  "must_include": ["256gb", "black"]    words or numbers the listing title must contain (variant: storage,
                                        colour, size, pack); use the spelling a title would use
  "big_stores_only": true               only big stores, authorised resellers and brand stores
  "new_only": true                      hide refurbished / used / bulk / copies
  "sort": "price" | "score" | "rating"  cheapest first, sort-order score, or best rated first
  "new_search": "iphone 16 pro"         search for another product (when the user asks about a different one)
  "reset": true                         clear all filters (with no other keys)
A new action replaces the old filters, so repeat any earlier filter the user still wants. When you set an action,
say in the answer what the list now shows."""


def build_messages(question, context, history=()):
    """System rules + the search data + earlier turns + the new question."""
    data = json.dumps(context, ensure_ascii=False)[:MAX_CONTEXT_CHARS]
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Search data (JSON):\n{data}"},
                {"role": "assistant", "content": '{"answer": "I have the search data.", "action": null}'}]
    for turn in list(history)[-6:]:  # the last few turns, so follow-up questions work
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": str(turn["content"])[:2000]})
    messages.append({"role": "user", "content": question})
    return messages


def answer(question, context, history=()):
    key = config.env("GROQ_API_KEY")
    if not key:
        raise RuntimeError("No Groq key: put GROQ_API_KEY=... in .env (free at console.groq.com)")
    main = config.env("GROQ_MODEL") or DEFAULT_MODEL
    messages = build_messages(question, context, history)
    # free tier: when the main model is busy (429), the smaller model (own limit) answers; if both are busy,
    # wait as long as Groq asks (at most 15 s) and try the main model once more
    for model in (main, FALLBACK_MODEL, main, FALLBACK_MODEL):
        body = {"model": model, "messages": messages, "temperature": 0.2, "max_completion_tokens": 1500,
                "response_format": {"type": "json_object"}}
        if model.startswith("openai/gpt-oss"):  # reasoning model: think briefly, return only the answer
            body.update(reasoning_effort="low", include_reasoning=False)
        r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=60)
        if r.status_code == 429:
            last = r
            if model == FALLBACK_MODEL:   # both busy: wait until Groq frees tokens (at most 20 s), then try again
                time.sleep(seconds_to_wait(r.headers))
            continue
        r.raise_for_status()
        reply = parse_reply(r.json()["choices"][0]["message"].get("content") or "")
        return {**reply, "model": model}
    last.raise_for_status()


def seconds_to_wait(headers):
    """Groq says when tokens free up: retry-after "7" or x-ratelimit-reset-tokens "7.66s" / "1m2s"."""
    text = headers.get("retry-after") or headers.get("x-ratelimit-reset-tokens") or "8"
    m = re.fullmatch(r"(?:(\d+)m)?([\d.]+)s?", text.strip())
    secs = (int(m.group(1) or 0) * 60 + float(m.group(2))) if m else 8.0
    return min(20.0, secs + 0.5)


def number(v):
    """70000, "70000", "₹70,000" or "70k" -> 70000.0; anything else -> None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.fullmatch(r"[^\d]*([\d,]+(?:\.\d+)?)\s*(k|thousand|lakh|l)?", str(v).strip().lower())
    if not m:
        return None
    return float(m.group(1).replace(",", "")) * {"k": 1e3, "thousand": 1e3, "lakh": 1e5, "l": 1e5}.get(m.group(2), 1)


ACTION_KEYS = {"stores": list, "exclude_stores": list, "must_include": list, "max_price": (int, float),
               "min_price": (int, float), "big_stores_only": bool, "new_only": bool, "sort": str,
               "new_search": str, "reset": bool}


def parse_reply(text):
    """The model's JSON reply -> {"answer", "action"}; only known action keys with the right types are kept, so a
    bad reply can never break the page. Plain text (no JSON) is used as the answer with no action."""
    try:
        data = json.loads(text)
    except ValueError:
        return {"answer": text.strip() or "Sorry, I could not produce an answer. Please ask again.", "action": None}
    answer = str(data.get("answer") or "").strip() or "Sorry, I could not produce an answer. Please ask again."
    raw = data.get("action") if isinstance(data.get("action"), dict) else {}
    def right_type(k, v):   # True / False are not prices, and "yes" is not True
        want = ACTION_KEYS[k]
        return isinstance(v, bool) if want is bool else isinstance(v, want) and not isinstance(v, bool)
    raw = {k: number(v) if k in ("max_price", "min_price") else v for k, v in raw.items()}
    action = {k: v for k, v in raw.items() if k in ACTION_KEYS and v is not None and right_type(k, v)}
    if action.get("sort") not in (None, "price", "score", "rating"):
        action.pop("sort")
    for k in ("stores", "exclude_stores", "must_include"):
        if k in action:
            action[k] = [str(x)[:40] for x in action[k] if str(x).strip()][:10]
    return {"answer": answer, "action": action or None}
