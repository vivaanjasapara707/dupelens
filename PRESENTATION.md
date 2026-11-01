# DupeLens: presentation kit

## 1. The 30-second pitch
On Amazon and Flipkart the same product is sold by many sellers, under different titles, photos and prices.
Search results also mix in look-alikes: a newer model, another colour, a 2-pack, or just a cover for it.
A buyer cannot tell which listings are really the same item, so they cannot tell the fair price or whom to buy from.

**DupeLens:** type a product name, paste a store link, or give a photo. It collects offers from Indian online
stores, uses machine learning to keep only the listings that are exactly the same item, and tells you the fair
price, the seller to trust and where to buy.

## 2. Demo script (about 4 minutes)
Before you start: run `.venv\Scripts\python.exe -m src.app` in the project folder, wait about 20 seconds,
open http://127.0.0.1:8000 and check that the top right says **"Live search connected"**.

1. **Search by name.** Click the example **Sony WH-1000XM5 headphones**.
   - Point at "Top of the sort order": the likely same item, its fair price, and the sort-order parts.
   - Point at the price range: the same headphones cost ₹19,900 to ₹48,928 (146% apart) across 5 offers.
   - Point at **"Suspiciously cheap"** on the ₹19,900 offer: far below the fair price is a warning, not a bargain.
   - Scroll to **"Same product, different colour"** (silver, pink, blue) and **"Look-alikes kept apart"**
     (XM6, WF earbuds, replacement ear pads). Say: *"a normal price-comparison site would mix these in."*
2. **Behind the scenes.** Scroll to the PCA map and "Why the model matched".
   - The dots are the offers.
   - The model's pair scores are listed: title similarity, photo meaning, exact object, colour difference.
   - Time per step is shown.
3. **Search by link.** Paste an Amazon product link, for example
   `https://www.amazon.in/Veentus-Dealsure-Professional-Curler-Machine/dp/B0BSR13TJL/`.
   It reads the product name from the link and searches.
4. **Search by photo.** Paste a product photo's address (right-click a product photo, then "Copy image
   address"), for example `https://m.media-amazon.com/images/I/61rS3WRPWjL.jpg`.
   - Google Lens names the product.
   - Your photo becomes the white dot, and every offer is compared with it.
5. **Accuracy.** Scroll to the footer, or open "How we tested the model", and quote the numbers in section 4.

**If the internet fails:** the 6 example buttons still work, because those searches are saved in the project.

## 3. How it works, in plain words
| Step | What happens | Words to know |
|---|---|---|
| Collect | Serper returns Google Shopping (India) offers: title, photo, price, store, rating | API = a service our code calls for data |
| Read | Three pretrained neural networks turn each photo and title into a list of numbers | **embedding** = numbers that describe a thing, so that similar things get similar numbers |
| Compress | **PCA** keeps the 32 most informative directions of each embedding (512 → 32 for the photo, 768 → 32 for the title) | It was fitted on training data only; "whitening" evens out the directions |
| Judge | **Logistic regression** (our trained model) adds up weighted clues for each pair of offers and gives a probability that they are the same item | The clues are title similarity (TF-IDF), photo similarity (CLIP), exact-object similarity (DINOv2), colour difference, photo fingerprint, size numbers and neighbour ranks |
| Check | Rules catch what the model misses: different model words or numbers, accessories, sizes, and per-search name words (shades, model lines) | **same** = identical item; **variant** = other colour, size or pack; **different** = another product |
| Sort | Among offers of the likely same item: fair price (median), cheapest, and a sort order | Sort order = same-item probability 35 + fair price 25 + stars 15 + reviews 10 + known store 15 (a fixed rule we chose, not trained) |

**Where the model learned:** the Shopee Kaggle dataset, 34,250 listings in 11,014 product groups. The split
was by product group: 70% train, 15% validation, 15% test, so the test only contains products the model never saw.
**Price is never a clue for matching.** Otherwise the "fair price" answer would be circular.

## 4. Numbers you can quote (all measured)
- **Shopee test, used exactly once:** F1 0.838 (95% CI 0.825–0.851).
  - Compare: 0.794 for title similarity alone, and 0.461 for doing nothing.
  - F1 balances precision ("when it says match, is it right?") with recall ("does it find the matches?").
- **Upgraded model (adds DINOv2 and colour):** validation F1 0.843 → 0.864.
- **PCA:** at 32 numbers instead of 512/768, validation F1 was 0.843, vs 0.846 at full size.
- **Live search on 8 fresh products** (labelled blind by Claude, an AI assistant, not humans). When DupeLens says "same item":
  - it is the identical item 23 of 40 times: **57.5%** (CI 28.6–81.4%, resampling whole searches)
  - it is at least the same product line 33 of 40 times: **82.5%** (CI 63.9–94.3%)
  - baseline: a random pair from the same search is the identical item only 3 of 30 times (10%)
  - the model missed only 2 of 30 true matches among the look-alikes it rejected
- **Speed:** a live search takes 2–5 seconds once the models are loaded (measured 2.2–4.6 s).

**CI (confidence interval):** we resampled the labelled pairs 1,000 times. The true rate is very likely inside the range.
It is wide because we have only 40 pairs.

## 5. Honest limits (say them before you are asked)
- Exact-item precision is 57.5%. The main errors are shades, scents and pack sizes that appear only in the
  photo or are not named in the title.
- The labels were made by an AI (Claude), not by humans, and there are only 40–100 pairs per round.
- The sort-order weights are our design choice, not learned from data.
- Prices are what Google Shopping shows at search time. We do not see every seller on Amazon.
- Uploading a photo file needs the online version, because Google Lens must be able to reach the photo.
  On the laptop, paste the photo's address instead.

## 6. Likely questions
**Why not just compare titles?**
Title similarity alone scored F1 0.794 on Shopee; our model scored 0.838. On live searches, titles look alike
across models ("PIC 20" vs "PIC 3.1"), so the photo models and the rules matter.

**Why logistic regression and not a neural network?**
We tried a small neural net (MLP): it scored 0.843, inside the logistic regression's confidence interval. So we
kept the simpler, explainable model, whose coefficients we can read (title similarity has the biggest weight).

**What does PCA do here, and did it help?**
It compresses each embedding to 32 numbers. Validation F1 stayed at 0.843 vs 0.846 at full size, and whitening
improved candidate recall over plain PCA at 64+ dimensions. So the vectors are much smaller with almost no loss.

**How do you know it is accurate, if you tuned it?**
We used two rounds of searches to design the rules. Then we froze the rules and tested on 8 brand-new products
exactly once. Only that last round is reported.

**Why is price not used for matching?**
We want to say whether a price is fair compared with the same item. If price helped decide what counts as the
same item, the answer would be circular.

**Why split by product group?**
A random split would put listings of the same product in both train and test, so the model would be tested on
products it had already seen.

**Is scraping allowed?**
We do not scrape Amazon or Flipkart directly; Serper provides Google Shopping results through an API.
A store page is only read when the user pastes its link or clicks "Open store".

**What would you do next?**
- Human labels with two annotators, to measure agreement.
- Read shade and colour from the photo.
- Learn the sort order from real purchase outcomes.
- Host it online with photo upload.

## 7. New since the first version (all measured; files in results/)
- **Ablation:** the model alone called 995 pairs "same" in the final searches, 26.4% identical (CI 7.8–49.5%); with the rules, 171 pairs, 57.5% identical. The rules double precision but also drop real matches (8 of 40 removed pairs were identical).
- **Error analysis (17 wrong "same" verdicts):** colour/finish 6, bundle/pack 4, other model of the same line 4, formulation 1, vague title 1, hyphenated code 1.
- **Identifier check:** no barcodes in the data; model codes in 228 of 978 titles; code-based labels agree with Claude on 25 of 26 pairs.
- **Store insights (60 searches):** the middle half of prices for the likely same item spreads by a median 37.9%; Amazon was cheaper than Flipkart in 5 of the only 6 searches where both sold the item.
- **PCA, three ways:** compression inside the matcher (512/768 -> 32), a per-search map, and a product map of 2,300 offers. On that map, 58.0% of each offer's 10 nearest neighbours share its category, vs 12.1% by chance, although PCA never saw the categories.
- **New pages:** Store insights, Price tracker (buy-now-or-wait rule, alerts), Ask DupeLens AI (Groq).
- **Likely question, "Why does it still make mistakes?":** "Colours and packs that only show in the photo. Our error analysis counted them; the next step is human labels and a learned model."
