# DupeLens

**Same product, many sellers: know where to buy it.** Type a product name, paste an Amazon or Flipkart link, or give a product photo. DupeLens collects offers from Indian online stores, uses a trained matcher to keep the listings that are likely the same item (not a look-alike, another colour or another pack), and shows the fair price, the sellers in a transparent sort order, and where to buy.

Status (1 Nov 2026): runs locally; not yet hosted (a Docker / Hugging Face Spaces setup is included). Every number in this file comes from code in this repo; the files are named next to each number. Accuracy labels after the Shopee stage were made by **Claude, an AI assistant, not humans**.

## What you can do with it

| Page | What it does | Code |
|---|---|---|
| **Search** (`/`) | Name, store link or photo address -> up to 40 live offers; every listing of the same model, big stores first, each tagged "same item" or "other variant" with storage / colour / size / pack read from the title and filter chips per variant; fair price, cheapest, sort order, "suspiciously cheap" flag; look-alikes shown apart; "big stores only" toggle; "Behind the scenes" panel with a per-search PCA map | `src/live.py`, `src/verdict.py`, `dashboard/index.html` |
| **Photo X-ray** (on every search) | PCA on the listing photos of one model ("eigen-products"): the average photo, eigen-images (how the photos differ), each photo rebuilt from the other photos' main directions, a heatmap of what does not fit, and an odd-one-out score that flags photos unlike the rest | `src/photo_xray.py` |
| **Ask DupeLens AI** | Questions about the search on screen ("Is Amazon cheaper than Flipkart here?"), answered by a Groq-hosted LLM from that data only; store-vs-store numbers are computed by the page, not the LLM | `src/assistant.py` |
| **Store insights** (`/insights.html`) | 60 live searches: price spread for the same item, cheapest store, Amazon vs Flipkart, and a PCA map of 2,300 offers | `src/insights.py` |
| **Price tracker** (`/tracker.html`) | Track an item; a daily job re-runs the search, the model finds the item again, history + buy-now-or-wait advice + price-drop alerts (in-app, optional Telegram) | `src/tracker.py` |
| **How we tested** (`/evidence.html`) | The original Shopee model and the 698-listing 2023 Amazon India test catalog | `src/service.py` |

## Run it

```bash
python3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt        # Windows (Mac/Linux: .venv/bin/pip); versions pinned for the saved models
.venv\Scripts\python -m pytest tests -q               # 42 tests
.venv\Scripts\python -m src.app                       # http://127.0.0.1:8000
```
Keys go in a file `.env` (never committed): `SERPER_API_KEY` (live search, free tier at serper.dev; a search costs 2 credits), `GROQ_API_KEY` (AI assistant, console.groq.com), optional `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` (alerts). The first start downloads 3.3 GB of pretrained encoders. Daily price checks: `scripts\update_prices.bat` (register it once with the `schtasks` line inside the file).
Online: `docker build .` runs anywhere; `python scripts/deploy_hf.py` creates a Hugging Face Space that runs the full app and stores the keys as Space secrets (needs `HF_TOKEN`); `python scripts/deploy_vercel.py <space-url>` publishes the pages on Vercel and forwards `/api/*` to that Space (needs `VERCEL_TOKEN`). The models (3.3 GB) cannot run on Vercel itself. Tests also run on GitHub Actions (`.github/workflows/tests.yml`).

## How it works

1. **Understand the input.** A name is searched as is; a store link gives the name from its address (`name_from_link`); a photo address goes to Google Lens through Serper, which names the product, and the photo itself is compared with every offer.
2. **Collect offers.** Up to 40 offers from Google Shopping India through Serper (title, photo, price, store, rating). Offers are not limited to big stores; 50.3% of 2,260 offers in the insights searches came from well-known stores (`src/insights.py`).
3. **Read photos and titles.** Three pretrained neural networks: CLIP ViT-B/32 (photo, 512 numbers), DINOv2-small (exact object, 384), multilingual MPNet (title, 768); plus a colour histogram and a perceptual hash.
4. **PCA (used three ways).**
   - *Compression inside the matcher:* one PCA per modality, fitted on 24,065 Shopee training listings with shrinkage whitening, keeps 32 of 512 (photo) and 32 of 768 (title) numbers. Validation F1 stayed 0.843 vs 0.846 at full size, and whitening beat plain PCA on candidate recall at every size (table below).
   - *Per-search map:* a second PCA puts each search's offers on a 2-D map in the "Behind the scenes" panel.
   - *Photo X-ray:* PCA directly on the listing photos (48×48 cut-outs, 6,912 numbers each) gives the average photo and eigen-images; leave-one-out PCA residuals (on DINOv2 features) score how unlike the other listings each photo is. Injection test on 48 real listings (`results/json/photo_xray_test*.json`, CIs resample searches): a photo of a different product added to a listing was ranked the most unusual **95.8%** of the time (CI 89.6–100%; raw-pixel PCA 54.2%; chance 12.4%); a look-alike from the same search only **31.2%** (CI 16.7–45.8%; pixels 22.9%). It catches wrong products, not close versions.
   - *Product map across 60 searches:* PCA of 2,300 offers' 64 numbers; the first two directions explain 18.2% of the variation, yet 58.0% of each offer's 10 nearest neighbours on the 2-D map share its category, against 12.1% by chance; PCA never saw the categories (`src/insights.py`).
5. **Judge every pair.** Logistic regression M1_v2 (trained on Shopee pairs, features below) gives each pair a probability; same / variant / different comes from the v2 rules (probability >= 0.90, number/size conflict, combo, colour distance) plus live-search rules in `src/verdict.py`: model words (Pro, Neo, Gen, ANC, UV, refurbished), model numbers and codes (PIC 20 vs PIC 3.1, V2 vs V3, DX2 = DX 2), accessory words, sizes (W, L, ml, g, jars), "only one title names a model", and per-search rare name words (Matchless vs Xeno, Plum Pick vs Red Coat). **Price is never a matching feature.**
6. **Sort the sellers** of the likely same item: fair price = median; sort order = same-item probability 35 + fair price 25 + star rating 15 + number of reviews 10 + well-known store 15 (a fixed rule, not a trained model or a trust rating). More than 40% below the median is flagged "suspiciously cheap". The probability shown is calibrated once `src/calibrate.py` has run (see Pending); until then it is labelled "model score".

## Exact-product search, store pages and photo checks (latest)

- **Exact product only.** For a name or link search, every offer is tagged against what the user typed (`src/verdict.py: query_match`): *exact* (all model numbers, the name in front of each number, the model words such as Pro/Max, no extra model word, no "18e" for "18"), *related* (another model of the same line), *accessory* (case, cover, "... for iPhone 18") or *other*. The main list shows only exact listings; related models are recommendations; accessories are hidden. If no listing names the product, the page says "Not found" and shows the closest product only as a recommendation.
- **Big-store product pages.** One Google web search per query (`store_pages`) adds the product pages of big / official stores (Amazon, Flipkart, Croma, Reliance Digital, JioMart, Apple, brand stores, authorised resellers) whose title is an exact match, with a direct link and the price from the page snippet (EMI, exchange and cashback amounts skipped; prices far from the listings' median dropped to "see price on the page").
- **Best price** = cheapest new listing from a trusted store (big store, authorised reseller, brand store); never refurbished/used, bulk lots, accessories, a price more than 40% below the median, or a bait listing.
- **PCA bait-listing check.** Among exact listings (5+), a photo with a leave-one-out PCA residual (DINOv2) robust z > 2.5 is flagged "Photo unlike the other listings" and cannot be the best price. Evidence: the injection test above (a different product's photo ranked most unusual 95.8% of the time).
- **PCA Colour Decoder** (`src/colour_pca.py`): CIELAB colour profiles of the product pixels -> PCA -> k-means (k by silhouette, >= 0.35) -> named colours, offered as a "Colour (from photo)" filter. Tested as a same-item rule on the labelled "same" pairs (`results/json/colour_pca_eval.json`) it did **not** help: final round 57.5% -> 56.2% identical (8 pairs removed, 3 of them wrong), round 4 32.5% -> 34.8% (17 removed, 12 wrong), so it is used only as a filter.
- These changes came after round 4; a new fresh round is needed before quoting accuracy for them.

## Live-search accuracy

Three rounds of 8 product searches each; pairs of offers labelled blind (same / variant / different, from titles and photos, without model output) by Claude, an AI assistant, not humans. Strata: **S** = pairs the system calls same, **R** = random pairs from the same search, **H** = rejected pairs with similar titles. 95% CIs resample whole searches (pairs within a search are not independent).

| Round | Use | S: identical item | S: at least same product line |
|---|---|---|---|
| Tune (`live_eval_tune.json`) | rules designed from its errors | 20/40 = 50.0% (CI 15.4–74.6%) | 23/40 |
| Test (`live_eval_test.json`) | rules redesigned again (shades, scents, line names), so also tuning data | 4/40 = 10.0% (CI 0–38.1%) | 22/40 |
| **Final** (`live_eval_final.json`) | 8 new products, rules frozen first, run once | **23/40 = 57.5% (CI 28.6–81.4%)** | **33/40 = 82.5% (CI 63.9–94.3%)** |

Final round baselines: a random pair from the same search was the identical item 3/30 = 10.0% (CI 3.4–15.4%); among rejected look-alikes, 2/30 = 6.7% were actually identical (CI 0–15.4%).

**Round 4: the model-level "where to buy" list** (`live_eval_round4.json`; 8 more new products, run once after the rules v3 and the model-level view were frozen: hyphenated codes, colours, packs, storage stated on one side only, vague-vs-named titles as variants). The search page now lists every listing of the same model (pairs called *same* or *other variant*), big stores first:

| Round 4 | Result | 95% CI |
|---|---|---|
| listing pairs (S + V, weighted by the 97 S and 525 V pairs in the 8 searches) that are the same model | **84.0%** | – |
| "same item" pairs that are at least the same model | 35/40 = 87.5% | 46.7–97.6% |
| "other variant" pairs that are at least the same model | 25/30 = 83.3% | 41.7–96.6% |
| "same item" pairs that are the identical item | 13/40 = 32.5% | 16.0–59.3% |
| rejected look-alikes that are actually the same model (missed) | 13/30 = 43.3% | 18.7–75.0% |

The list is usually the right model; telling the exact version apart (shampoo variants, kajal versions, colours) is the weak point and depends heavily on the product (57.5% identical on the final round, 32.5% on round 4).

**Ablation, final round** (`src/ablation.py`, `results/json/ablation_final.json`). The live rules only remove "same" verdicts, so a fresh blind sample of the removed pairs (40, mixed with 20 decoys) gives the model-alone precision:

| | Pairs called same (8 searches) | Identical-item precision | 95% CI |
|---|---|---|---|
| M1_v2 + v2 rules only | 995 | 26.4% | 7.8–49.5% |
| + live-search rules | 171 | 57.5% | 28.6–81.4% |

The rules double precision but drop real matches: 8 of 40 removed pairs (20%) were identical.

**Error analysis, final round** (`results/error_analysis_final.csv`; types assigned by Claude). Of the 17 "same" verdicts that were not the identical item: colour/finish visible only in the photo or one title 6, bundle/combo/pack not detected 4, other model of the same product line sharing a name word 4, different formulation 1, vague vs specific title 1, hyphenated code read wrongly (WH-1000XM5 vs WF-1000XM5) 1. The 2 missed matches: code in only one title (strict rule) 1, verdict "variant" 1.

**Identifier check** (`src/identifiers.py`, `results/json/identifiers.json`). Serper offers carry no GTIN/EAN/UPC/MPN/ASIN field; Google's productId is almost unique per offer (971 distinct among 978), so it cannot link stores. A model code appears in 228 of 978 titles. In the final round, 26 of 100 pairs have a code in both titles; code-based same/different model labels agree with Claude's model-level labels on 25 of 26 (the exception: replacement ear pads named after the WH-1000XM5).

## Store insights (60 searches, 1 Nov 2026)

From `results/json/insights.json` (CIs resample searches; "same item" = the model's largest non-accessory group per search):
- Price spread for the likely same item (middle half of prices, 75th / 25th percentile, searches with 4+ prices): median **37.9%** (CI 30.0–69.6%, 43 searches); over 20% in 79.1% of searches. Max vs min was dominated by matching mistakes (bulk packs, accessories), so it is not used.
- Amazon vs Flipkart on the same item: only **6** searches had both; Amazon was cheaper in 5 (median gap -18.8%, CI -63.8% to -0.3%). Too few to generalise.
- Cheapest well-known store, in 23 searches with 2+ of them: Amazon 7, JioMart 5, BigBasket 4, Zepto 2, Myntra 2, 1mg 2, Blinkit 1.

## Pending (scripts ready, results not produced yet)

- **Second, non-Anthropic labeller** (`src/llm_labels.py`): sends the final-round pairs (titles, stores, photos only) to a Groq vision model (`qwen/qwen3.8-27b`) and reports Cohen's kappa vs the Claude labels and precision on agreed pairs. Needs `GROQ_API_KEY`.
- **Calibration of M1_v2** (`src/calibrate.py`): Platt and isotonic, cross-fitted on Shopee validation, reliability plot, calibrated probability in the sort order. Needs the Shopee data (Kaggle) on this machine.

## How the matching model was built and tested

The model below was trained and tested before live search existed; its numbers come from the Shopee data and a 2023 Amazon India catalog.

A logistic-regression matcher over image, text and variant-aware features reaches **per-listing F1 0.838 (95% CI 0.825–0.851)** on a held-out set of Shopee product groups, against 0.461 for a self-only floor and 0.794 for the best single feature. An upgraded model (v2: adds DINOv2 exact-item photo matching and a colour check) raises Shopee validation F1 from 0.843 to 0.864 and, on fresh Amazon India pairs, is right **82.5% of the time (95% CI 70–92.5%)** when it says two listings are the same product, up from 50%. It also tells exact duplicates apart from colour/pack variants.

**Older static demo (2023 catalog only): https://dupelens.vercel.app**

Every number below comes from code in this repo; the raw rows are in `results/results.csv`.

## Check one listing against the 2023 test catalog

- **Online demo:** https://dupelens.vercel.app — pick any of the 698 catalog listings; it is hidden from the catalog and checked as if new (real model output, precomputed).
- **Check a brand-new listing against the catalog (runs the models locally):**

```bash
.venv/bin/python -m src.app      # then open /evidence.html
```

Open http://127.0.0.1:8000/evidence.html, choose **Check a new listing**, paste the Amazon title, paste the photo address (right-click the product photo → Copy image address) or drop the photo file, optionally add your price, and press **Check listing**.

Example (a real Amazon listing that is *not* in the catalog): "FIT & SHAPE Hair Straightener Comb…", ₹390 → **Already listed by 12 other sellers**, same product sells for ₹395–₹1,449, typical ₹498, so ₹390 is 21.7% below typical. The first check takes ~10 s while the models load; later checks took 0.13–0.48 s (measured, photo download included).

The app works best for products in the trained category (personal-care appliances: hair straighteners, dryers, curlers, trimmers). To use another category, change `INDIA_CATEGORY` in `src/config.py` and re-run the India steps.

### The problem

On Amazon and Flipkart one product often appears as many separate listings from different sellers, with different titles, photos and prices (for example the same hair-straightener comb at ₹299, ₹399 and ₹499). Buyers cannot tell which listings are the same item, or whether a cheaper one is a different variant or pack size.

### How the model works

1. **Data.** Shopee Price Match Guarantee (Kaggle): 34,250 listings in 11,014 product groups (`label_group` = same product). Split **by product group** 70/15/15 (seed 42): train 24,065 listings / 7,710 groups, validation 5,045 / 1,652, test 5,140 / 1,652. No product appears in two splits, so the test score measures matching *new* products.
2. **Embeddings.** CLIP ViT-B/32 image vectors (512-d) and multilingual MPNet title vectors (768-d), unit length; perceptual hash (`imagehash.phash`) recomputed for every photo.
3. **PCA.** One PCA per modality, fitted on train only, with shrinkage whitening (Λ + εI)^-1/2, ε = 10% of the mean eigenvalue, then re-normalised. Kept **32 dims per modality**, the smallest setting whose validation F1 fell inside the full-dimension confidence interval.
4. **Candidates.** FAISS exact inner-product search, top k = 50 neighbours within a split (k = 50 beat k = 20 by more than 2 recall points on validation).
5. **Pair features (symmetric):** image cosine, text cosine, TF-IDF char 3–5-gram cosine, phash Hamming distance, number-token match/conflict (e.g. "500 ml" vs "1 l"), mutual-neighbour flag, min/max rank, title length ratio. **Price is never a feature**, otherwise the price comparison would be circular.
6. **Model.** Logistic regression on standardised features with balanced class weights (positive share of train candidate pairs: 6.0%), probability threshold tuned on validation (0.82). **Test was used exactly once.**

## Results (Shopee test, single run)

| Step | Model | Test F1 | 95% CI | Pair precision | Pair recall |
|---|---|---|---|---|---|
| B0 | Self-only floor | 0.461 | 0.442–0.481 | – | 0.000 |
| B1 | Exact Shopee image hash | 0.547 | 0.527–0.569 | 0.994 | 0.107 |
| B2 | Best single feature (TF-IDF, tuned cutoff) | 0.794 | 0.780–0.807 | 0.782 | 0.694 |
| Ablation | M1 without number tokens | 0.833 | 0.821–0.846 | 0.827 | 0.708 |
| **M1** | **Logistic regression** | **0.838** | **0.825–0.851** | 0.833 | 0.717 |
| M2 | Small MLP (32-16) | 0.843 | 0.830–0.855 | 0.846 | 0.715 |
| M1 + components | Grouping rule (connected components, conflict edges dropped) | 0.816 | 0.802–0.830 | 0.760 | 0.817 |

F1 is per listing with the self-match counted (Kaggle convention); CIs bootstrap product groups (1,000 resamples). Candidate recall on test was 0.906 (CI 0.882–0.934), which caps pairwise recall. M2 is within M1's CI, so the simpler, interpretable M1 is the main model.

**Compression (validation).** Candidate recall with raw full-dimension vectors was 0.954; with shrinkage-whitened PCA it was 0.935 at 32 dims and 0.982 at 256 dims. Whitening raised recall over plain PCA at every size (e.g. 0.970 vs 0.940 at 64 dims), so compression with whitening is not just smaller but better at 64+ dims. M1 validation F1: raw 0.846, 32-d 0.843 (inside the raw CI 0.836–0.856).

**What the model weighs (standardised coefficients):** TF-IDF cosine +2.01, phash distance −0.54, number conflict −0.53, image cosine +0.42, text cosine −0.19. Text cosine is negative because it is strongly correlated with TF-IDF cosine, so the two split the weight; it does not mean similar meaning hurts.

**Where it fails.** Of 15 sampled false matches, 14 were variants (same brand or product line, different item); of 15 misses, 5 were variants (labelled as one product despite different size/shade), 3 garbled or abbreviated titles, 3 stock photo vs product photo, 2 bundles, 2 label noise.

![Recall vs dims](results/plots/recall_vs_dims.png) ![PR curve](results/plots/pr_curve_test.png) ![Explained variance](results/plots/explained_variance.png)

## Model v2: exact-item photos, colour, and a three-way verdict

The audit found that most wrong Indian matches were the same model in another colour or pack, and that CLIP recognises *what kind of thing* is in a photo better than *which exact item*. v2 adds two pair features and a three-way verdict:

- **dino_cos:** cosine of DINOv2-small image vectors (384-d), a model trained to recognise the same object across photos.
- **colour_dist:** 1 − histogram intersection of hue × saturation of the product pixels (near-white background ignored).
- **Verdict:** *same* (same model, same colour/size/pack) · *variant* (same model, other colour, size, pack or combo) · *different*. The model probability decides "same model" (≥ 0.90); colour distance > 0.45, a size/pack conflict, or a combo mismatch turns it into a variant. These two cut-offs were chosen on the 210 earlier Indian labels only.

| Shopee validation (same candidate pairs) | F1 | 95% CI | vs M1 (paired) |
|---|---|---|---|
| M1 (original) | 0.843 | 0.833–0.852 | – |
| **M1_v2** | **0.864** | 0.855–0.874 | +0.022 (+0.017 to +0.026) |
| v2 without DINOv2 | 0.850 | 0.841–0.860 | +0.007 |
| v2 without colour | 0.861 | 0.852–0.870 | +0.018 |

The Shopee test split was already used once, so v2 is compared on validation and checked on a **fresh** Indian held-out set (60 pairs never labelled before, blind): **"same product" verdicts were correct for 33 of 40 (82.5%, CI 70–92.5%)** and at least the same model for 37 of 40; "variant" verdicts were correct for 11 of 20 (55%, CI 30–75%; 5 were actually identical, 4 different products). With v2, 434 of 698 catalog listings are already listed as the same product by another seller, 88 only as a colour/pack variant, 176 look new; 343 have a typical price and 62 are priced more than 20% above it.

## India price demo (v1 model) (Amazon India, Personal Care Appliances, 2023 snapshot)

- **Category choice.** Jewellery was tried first; its titles are templated and only 1 of 150 sampled predicted matches was truly the same product. A duplicate probe (`results/phase1_search/duplicate_probe.py`) then found Personal Care Appliances has by far the most true duplicate listings (599 pairs with near-identical photo and title under different ASINs). 698 listings kept after dropping 372 with dead images.
- **Distribution shift.** At 32 dims the frozen Shopee PCA explains 0.503 of Indian image variance (Shopee test 0.479) and 0.570 of text variance (0.563): little shift in the embeddings, but the Shopee threshold over-matches inside one narrow category.
- **Calibration.** 150 blind-labelled pairs: stratum-weighted precision at the Shopee threshold was 0.39 (CI 0.27–0.50), rising to 0.74 above probability 0.99. The India match threshold is therefore 0.99.
- **Held-out precision.** On 60 *new* labelled matches above 0.99: **precision 0.50 (CI 0.37–0.63)** under a strict rule (same model *and* same colour and pack). Of the 30 false matches, 11 were the same model in another colour, 7 combo packs with different contents, 7 a single item vs a multi-colour/multi-pack set, 5 other.
- **Price output.** Each listing is checked against all the others (leave-one-out). 417 of 698 are already listed by at least one other seller; 336 have a typical price (median of at least 2 other listings of the same product); 61 are priced more than 20% above it and 61 at least 10% below. Prices use each listing's direct matches; connected components chained into one 100+ listing group on Indian data.

**Labelling disclosure:** the Indian match labels (150 calibration + 60 held-out pairs) and the 30 Shopee error tags were made by **Claude, an AI assistant**, blind to model scores, from titles and photos. No human labels were collected, so no inter-annotator agreement could be measured.

## Limits

- All accuracy figures after the Shopee stage rest on labels made by Claude, an AI assistant, not humans, with 30–40 pairs per stratum; CIs are wide. No human agreement has been measured yet.
- Live search: identical-item precision 57.5% on the final round; the main errors are colours, shades and packs visible only in photos. Hand-written rules do much of the work (ablation above) and remove some true matches.
- Live offers come from Google Shopping India through Serper; it does not list every seller on Amazon or Flipkart, and prices are as shown at search time. Uploading a photo file needs the hosted version (Google Lens must reach it); locally, paste the photo's address.
- The sort order and the buy-or-wait advice are fixed rules, not trained models; price history starts on the day an item is first tracked.
- 2023 catalog demo: Shopee has no prices, so the catalog price comparison runs only on that 2023 snapshot; number tokens miss karat ("22 K"), and the brand "3M" is read as 3 metres.

## Reproduce

```bash
python3.11 -m venv --copies .venv && .venv/bin/pip install torch sentence-transformers faiss-cpu scikit-learn pandas numpy matplotlib imagehash pillow requests kaggle six pytest fastapi uvicorn python-multipart
kaggle competitions download -c shopee-product-matching -p data/ && unzip data/shopee-product-matching.zip -d data/
.venv/bin/python -m pytest tests            # 42 unit tests
.venv/bin/python -m src.data                # group split
.venv/bin/python -m src.india               # Indian listings + photos (category in src/config.py)
.venv/bin/python -m src.embed               # embeddings + phash (Apple GPU 'mps' if available)
.venv/bin/python -m src.candidates          # PCA sweep, candidate recall
.venv/bin/python -m src.train               # baselines, M1, M2, ablation (validation)
.venv/bin/python -m src.train --test        # the single test run (refuses a second run)
.venv/bin/python -m src.india --frozen      # India calibration sample
.venv/bin/python -m src.india --matches     # India price view + held-out sample
.venv/bin/python -m src.export              # plots + dashboard data
.venv/bin/python -m src.extra               # v2: DINOv2 + colour features
.venv/bin/python -m src.train_v2            # v2 model, compared on validation
.venv/bin/python -m src.variant --calibrate  # v2 verdict cut-offs on earlier labels (then --sheet, --stats)
.venv/bin/python -m src.service             # demo results for every catalog listing
.venv/bin/python -m src.app                 # live app at http://127.0.0.1:8000 (live search needs SERPER_API_KEY in .env)
.venv/bin/python -m src.live "boat airdopes 141"   # one live search in the terminal
.venv/bin/python -m src.live_eval --stats --final # live-search accuracy (search-level CIs); --sheet builds a new blind sheet
.venv/bin/python -m src.ablation           # LR vs LR+rules on the final round
.venv/bin/python -m src.identifiers        # identifier coverage + code-based labels
.venv/bin/python -m src.insights --collect # 60 searches (2 Serper credits each), then without --collect: insights + PCA map
.venv/bin/python -m src.llm_labels --label # second labeller (Groq), then without --label: kappa
.venv/bin/python -m src.calibrate          # Platt / isotonic on Shopee validation (needs Kaggle data)
.venv/bin/python -m src.tracker --update   # one round of price checks
.venv/bin/python -m http.server 8765 --directory dashboard   # open http://localhost:8765 (live: https://dupelens.vercel.app)
```

`notebooks/dupelens.ipynb` runs the same steps and shows the results. On an exFAT drive, delete macOS `._*` sidecar files after big writes (`find . -name '._*' -delete`). Do not import torch and faiss in one process unless faiss is imported first.

## Future work

- More Indian categories (one setting in `src/config.py`, then re-run the India steps).
- Read shade and colour from the photo (the main live error); split hyphenated model codes correctly.
- Human labels with two annotators for agreement; a learned (fine-tuned) matcher to replace hand-written rules.
- Live scraping, CLIP fine-tuning or ArcFace, approximate search (IVF/HNSW), price prediction, listing photos on the dashboard.

## Resume bullet

Built DupeLens, a duplicate-listing matcher for Indian e-commerce: logistic regression over CLIP, DINOv2 and MPNet embeddings (PCA-compressed 512/768 -> 32 dims with shrinkage whitening, validation F1 0.843 vs 0.846 uncompressed) reached Shopee test F1 0.838 (95% CI 0.825-0.851) vs 0.794 for the best single feature; built it into a live search over Google Shopping India where "same item" verdicts were identical 57.5% of the time on a frozen, blind-labelled final round (vs 10% for random pairs; AI-labelled), with an ablation showing rules doubled precision (26.4% -> 57.5%), plus a 60-search store-insights study, a PCA product map, a price tracker and a Groq-powered shopping assistant.
