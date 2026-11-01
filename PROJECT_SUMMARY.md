# DupeLens: complete project summary for review

Status on 1 November 2026: built and running locally; not yet deployed online.
Every number below was produced by code in the repository
(https://github.com/figureitoutwithkrish/dupelens; the latest local commits are not pushed yet).

## 1. What it is
DupeLens is a student project for a Fundamentals of Machine Learning course, built by a team of 2.

**Problem:** on Indian shopping sites, one product is listed by many sellers with different titles, photos and
prices, and search results mix in look-alikes (other models, colours, pack sizes, accessories).

**What it does:** the user gives a product name, an Amazon/Flipkart link, or a product photo. DupeLens then:
- collects live offers from Indian online stores
- uses a trained matching model plus rules to decide which offers are exactly the same item
- reports the fair price (median), the cheapest store, a recommended seller, and links to buy

## 2. How it was built (disclosure)
- The two students set the goal and the direction.
- Most of the code, the experiments and the write-up were done by an AI coding assistant (Claude, in Claude Code),
  working under rules the team set:
  - never invent numbers
  - use the Shopee test split only once
  - never use price as a matching feature
  - label blind
  - disclose that labels are AI-made
- **All evaluation labels were made by Claude, an AI assistant, not by humans.** There is no human-labelled
  data and no measurement of agreement between two labellers.

## 3. User-facing features (local web app, FastAPI + one HTML page)
- **Three ways in:**
  - product name
  - Amazon or Flipkart link: the product name is read from the address
  - photo address: Google Lens, called through Serper, names the product, and the user's photo is then compared
    with every offer by the model
  - Uploading a photo file needs a public URL, so it only works once the app is hosted.
- **Offers:** up to 40 per search from Google Shopping India, via the paid-per-call Serper API (free tier
  2,500 credits; a search costs 2 credits).
- **Results:**
  - **likely-same-item sellers:** listed in a sort order, with fair price and price range
  - **variants:** same product, another colour, size or pack
  - **look-alikes kept apart**
  - a "suspiciously cheap" flag when the price is more than 40% below the median
- **"Behind the scenes" panel:** a 2-D PCA map of the offers, the pair features behind a match, and timing per step.
- **Well-known stores only (default; toggle).**
  - In 28 saved searches, 932 offers came from 247 different stores.
  - 511 (55%) were classed as well-known: big retailers and marketplaces, quick-commerce apps, pharmacies, or the
    brand's own store.
  - The model still compares every offer; the filter only changes what is shown and recommended.
- **Sort order:** a fixed hand-set rule, not learned and not a trust rating.
  - Weights: model's same-item confidence 35, fair price 25, star rating 15, number of reviews 10, well-known store 15.
- **Offline backup:** 6 example searches are saved for a demo without internet.
- **Speed:** 2–5 s per search once models are loaded (measured 2.2–4.6 s on a Windows laptop CPU).
  The first start downloads 3.3 GB of pretrained models.

## 4. Machine-learning pipeline
1. **Embeddings (pretrained, not fine-tuned)**, computed for each listing:
   - CLIP ViT-B/32 image vector (512-d)
   - multilingual MPNet title vector (768-d)
   - DINOv2-small image vector (384-d)
   - perceptual hash
   - 32-bin hue×saturation colour histogram of non-background pixels
2. **PCA**, one per modality, fitted on Shopee train only:
   - shrinkage whitening (Λ + εI)^-1/2 with ε = 10% of the mean eigenvalue, then re-normalised
   - 32 dimensions kept per modality: the smallest size whose validation F1 fell inside the full-dimension CI
3. **Candidates:** the 50 nearest neighbours by cosine similarity on the concatenated PCA vectors.
4. **Pair features (symmetric):**
   - image cosine, text cosine, TF-IDF char 3–5-gram cosine, phash distance
   - number-token match and conflict ("500 ml" vs "1 l")
   - mutual-neighbour flag and min/max rank
   - title length ratio
   - DINOv2 cosine, colour distance
   - **Price is never a feature.**
5. **Model (M1_v2):** logistic regression on standardised features, balanced class weights, threshold tuned on
   validation.
6. **Verdict rules** (same / variant / different), added during this work for live search:
   - model words (Pro, Neo, Gen, ANC, UV, refurbished)
   - model numbers and codes (PIC 20 vs PIC 3.1, V2 vs V3, DX2 = DX 2)
   - accessory words (case, cover, replacement)
   - sizes (W, L, ml, g, jars)
   - "only one title names a model"
   - per-search rare "name words" (model lines, shades) that two titles do not share
   - These rules are hand-written heuristics, tuned on AI-labelled data (see section 6).

## 5. Offline results (Shopee Price Match Guarantee, Kaggle)
**Data:** 34,250 listings in 11,014 product groups.

**Split by product group, 70/15/15, seed 42:**

| Split | Listings | Groups |
|---|---|---|
| train | 24,065 | 7,710 |
| validation | 5,045 | 1,652 |
| test | 5,140 | 1,652 |

**Metric:** per-listing F1 (Kaggle convention). 95% CIs bootstrap product groups (1,000 resamples).
The test split was used exactly once.

| Model | Test F1 | 95% CI |
|---|---|---|
| Self-only floor | 0.461 | 0.442–0.481 |
| Exact Shopee image hash | 0.547 | 0.527–0.569 |
| Best single feature (TF-IDF) | 0.794 | 0.780–0.807 |
| M1 without number tokens | 0.833 | 0.821–0.846 |
| **M1 logistic regression** | **0.838** | **0.825–0.851** |
| Small MLP | 0.843 | 0.830–0.855 |
| M1 + connected-components grouping | 0.816 | 0.802–0.830 |

**Other offline results:**
- Candidate recall on test: 0.906.
- **PCA:** validation F1 was 0.843 at 32-d vs 0.846 with raw full-dimension vectors (inside the raw CI).
- **M1_v2 (adds DINOv2 + colour):** validation F1 0.864 (CI 0.855–0.874) vs 0.843 for M1.
  - Paired gain: +0.022 (CI +0.017 to +0.026).
  - This was compared on validation only, because the test split had already been used.

## 6. Live-search evaluation (the main real-world result)
**Method:**
- Run 8 product searches.
- Sample offer pairs in three strata:
  - **S:** the model says "same"
  - **R:** a random pair from the same search
  - **H:** a pair the model rejected but whose titles are similar
- Label each pair same / variant / different from titles and photos, blind to model output (labelled by Claude).
- **Same** = same model and same colour/size/pack.

**Three rounds, each on fresh products:**

| Round | Products | When the model said "same", truly identical | What happened next |
|---|---|---|---|
| Tune | 8 | 20/40 | Rules were redesigned from these errors |
| Test | 8 new | 4/40 (shades, scents, model-line names) | Rules were redesigned again, so this round is also tuning data |
| **Final** | 8 new, rules frozen first, run once | **23/40 = 57.5%** (CI 28.6–81.4%) | Reported result |

**Final round in full:**

| Measure | Result | 95% CI |
|---|---|---|
| "Same" verdicts that are the identical item | 23/40 = 57.5% | 28.6–81.4% |
| "Same" verdicts that are at least the same product line (identical or variant) | 33/40 = 82.5% | 63.9–94.3% |
| Random pairs from the same search that are identical (naive baseline) | 3/30 = 10% | 3.4–15.4% |
| Rejected look-alike pairs that were actually identical (misses) | 2/30 = 6.7% | 0–15.4% |

**Caveats:**
- The samples are small: 40/30/30 pairs.
- The labels are AI-made.
- The final round used 1 page of results (40 offers) and the then-current display; later changes affect display only.

## 7. Earlier evaluation on a fixed catalog (Amazon India 2023, Kaggle)
- **Catalog:** 698 personal-care listings.
- **Version 1 model:** held-out precision 0.50 (CI 0.37–0.63) under a strict rule.
- **Version 2 model:** "same" verdicts correct 33/40 = 82.5% (CI 70–92.5%); "variant" verdicts correct 11/20 (55%).
- **Known wrong rows:** the catalog's "biggest price gaps" list contains two visible wrong matches (a toothbrush
  2-pack, a straightener). They are flagged on the evidence page.

## 8. Engineering
- **Stack:** Python 3.11; torch, sentence-transformers, transformers, scikit-learn, FAISS, FastAPI.
- **Front end:** a single HTML/JS file (about 460 lines, no framework).
- **Code:** about 2,200 lines of Python in `src/`. Main files for live search:
  - `src/live.py` (323 lines)
  - `src/verdict.py` (110)
  - `src/live_eval.py` (100)
- **Tests:** 36 unit tests (metrics, number tokens, verdict rules, store and accessory checks), all passing.
- **Reproducibility:** the split is fixed and every experiment is logged to `results/results.csv` and
  `results/json/*.json`. The labelled sheets (`results/live_*_sheet.csv`) and hidden keys
  (`cache/live_*_key.csv`) are in the repo; Serper responses are cached locally only.
- **Project docs:**
  - `HANDOFF.md`: progress log
  - `PRESENTATION.md`: demo script and Q&A
  - `CLAUDE.md`: rules for the AI assistant

## 9. Known weaknesses
1. **Exact-item precision on live search is moderate (57.5%).** The main errors are shades, scents and pack sizes
   that differ only in the photo, or vague titles.
2. **No human labels.** Every accuracy figure after Shopee rests on AI labels with 30–40 pairs per stratum, so the
   CIs are wide.
3. **The matching rules are hand-written heuristics** layered on the learned model and tuned on AI-labelled rounds.
   Part of the gain comes from rules, not learning.
4. **No model is fine-tuned on product-matching data.** The embeddings are used as is, and the logistic regression
   was trained on Shopee (Indonesian marketplace), not Indian listings.
5. **The sort-order weights are a design choice.** They are not validated against any outcome (returns, fraud,
   satisfaction).
6. **Data source:** Google Shopping via a third-party API (Serper).
   - It does not see every seller on Amazon or Flipkart.
   - Often only 1–2 well-known stores sell the exact item.
   - In one tested search only 13 of 40 offers came with any link, and those open Google's product page; the app
     finds the store's own page with an extra search when the user clicks.
7. **Not deployed:** photo upload needs hosting; the public Vercel page is an older static demo.
8. **Single-machine prototype:** no user accounts, no price history, no monitoring.

## 10. Where to check
- **Code:** `src/live.py`, `src/verdict.py`, `src/service.py`, `src/train.py`, `src/train_v2.py`, `src/live_eval.py`
- **Results:** `results/results.csv`, `results/json/live_eval_final.json`, `results/live_final_sheet.csv` with
  `cache/live_final_key.csv`
- **Docs:** `README.md`, `HANDOFF.md` section 9

## 11. Added after the first review (see README.md for every number and its file)
- CIs for live metrics now resample whole searches (the table above uses them).
- Ablation (final round): model alone 26.4% identical-item precision on 995 pairs (CI 7.8–49.5%) vs 57.5% with the rules on 171 pairs; 8 of 40 sampled removed pairs were identical.
- Error analysis of the 17 wrong "same" verdicts (types assigned by Claude); identifier check (no GTIN/ASIN fields; model codes in 228/978 titles; 25/26 agreement on code-bearing pairs).
- Store insights over 60 searches, a PCA product map (58.0% same-category nearest neighbours vs 12.1% chance), a price tracker with a rule-based buy-or-wait hint and alerts, a Groq-hosted assistant answering from on-screen data.
- Deployment files (Dockerfile, Hugging Face Spaces script) and a GitHub Actions test workflow; 42 tests.
- Pending: second labeller via a non-Anthropic vision model (script ready; needs a Groq key) and calibration on Shopee validation (script ready; needs the Kaggle data). Not deployed yet.
