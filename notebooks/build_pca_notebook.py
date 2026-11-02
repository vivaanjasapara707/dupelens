"""Builds notebooks/pca_dupelens.ipynb (run from the repo root; then execute it with nbconvert)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# PCA in DupeLens: done by hand, step by step

DupeLens decides whether two online listings are the **same product**. For every pair of listings it measures
**12 features** (how similar the photos are, how similar the titles are, whether the model numbers match, ...),
and a logistic-regression model turns them into a decision.

This notebook does PCA on those 12 features **manually** (mean, centring, covariance matrix, eigenvalues,
eigenvectors, projection), checks the result with `sklearn`, and finds **which features matter most**.
Then it shows the two other places PCA works inside DupeLens:
- compressing the CLIP photo vectors (512 numbers) and the MPNet title vectors (768 numbers) to 32,
- the **Genuine Look** check: a photo that PCA cannot rebuild from the other listings is not the real product.

Data: 430 pairs of real listings from live Indian searches, labelled *same*, *variant* (same product, another
colour/size/pack) or *different* (labelled by Claude, an AI assistant, not humans).
""")

md("## 1. Load the data")
code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/pair_features.csv")
features = ["img_cos", "txt_cos", "tfidf_cos", "phash_dist", "num_match", "num_conflict",
            "mutual", "rank_min", "rank_max", "len_ratio", "dino_cos", "colour_dist"]
A = df[features]
Y = df["label"]
print(A.shape)
print(Y.value_counts())
A.head()
""")
md("""
| Feature | What it measures for a pair of listings |
| --- | --- |
| img_cos | photo similarity (CLIP neural network, after PCA to 32) |
| txt_cos | title meaning similarity (MPNet neural network, after PCA to 32) |
| tfidf_cos | title word overlap (TF-IDF) |
| phash_dist | perceptual-hash distance of the photos (0 = identical photo) |
| num_match | model numbers / codes the two titles share |
| num_conflict | model numbers that disagree (16 vs 16e, 128 GB vs 256 GB) |
| mutual | 1 if each listing is in the other's 50 nearest neighbours |
| rank_min, rank_max | neighbour rank of one listing in the other's list (1 = most similar) |
| len_ratio | shorter title length / longer title length |
| dino_cos | photo similarity (DINOv2 neural network: the exact object) |
| colour_dist | difference between the colour histograms of the photos |
""")

md("""
**A feature that never changes carries no information.** Let us check each feature's spread.
""")
code("""
print(A.std().round(4))
constant = [f for f in features if A[f].std() == 0]
print("constant features:", constant)
""")
md("""
`mutual` is 1 for every pair: in a live search of about 40 listings, every listing is always among another's 50
nearest neighbours. A feature with zero variance cannot help PCA (and dividing by its standard deviation of 0 is
impossible), so we drop it. **11 features remain.**
""")
code("""
features = [f for f in features if f not in constant]
A = df[features]
print(A.shape)
""")

md("""
## 2. Why we standardise first

PCA looks for the directions of **largest variance**. Our features have very different scales: a cosine
similarity is between 0 and 1, but `rank_max` goes up to 51 and `phash_dist` up to 64. Without scaling, PCA
would simply pick the features with the biggest numbers (just like *Insulin* dominated PC1 in the diabetes
example, because its values are in the hundreds).
""")
code("""
print(A.std().round(3).sort_values(ascending=False))
""")
code("""
from sklearn.decomposition import PCA

raw = PCA(2).fit(A)                      # PCA WITHOUT scaling (only to show the problem)
print(pd.Series(raw.components_[0], index=features).abs().sort_values(ascending=False).round(3))
""")
md("""
PC1 of the unscaled data is almost only `rank_max`, `rank_min` and `phash_dist`, the three features with the
biggest numbers: the biggest numbers win, not the most informative features. So every feature is **standardised** (z-score): subtract its mean and divide by its standard
deviation. Then every feature has mean 0 and standard deviation 1, and PCA compares them fairly.
""")
code("""
S = (A - A.mean()) / A.std()
print(S.mean().round(3).values)
print(S.std().round(3).values)
""")

md("## 3. PCA by hand")
md("**Step 1: mean of every feature, and centre the data**")
code("""
M = np.mean(S.values.T, axis=1)
C = S.values - M
print(M.round(6))
print(C.shape)
""")
md("**Step 2: covariance matrix (11 x 11)**")
code("""
V = np.cov(C.T)
print(V.shape)
print(pd.DataFrame(V, index=features, columns=features).round(2))
""")
md("**Step 3: eigenvalues and eigenvectors of the covariance matrix, sorted from largest to smallest**")
code("""
eigvalues, eigenvector = np.linalg.eig(V)
idx = eigvalues.argsort()[::-1]
eigvalues = eigvalues[idx]
eigenvector = eigenvector[:, idx]
print(eigvalues.round(4))
""")
md("""
Each eigenvector is a **principal component** (a direction made of all 12 features); its eigenvalue is the
variance of the data along that direction. The sum of the eigenvalues equals the total variance, which is 11 here
(11 standardised features, each with variance 1).
""")
code("""
print("sum of eigenvalues:", eigvalues.sum().round(4))
ratio = eigvalues / eigvalues.sum()
cumulative = np.cumsum(ratio)
print(pd.DataFrame({"eigenvalue": eigvalues.round(3), "explained %": (100 * ratio).round(1),
                    "cumulative %": (100 * cumulative).round(1)}, index=[f"PC{i+1}" for i in range(len(features))]))
""")
md("**Step 4: choose how many components to keep (90% of the variance, like `PCA(0.90)`)**")
code("""
k = int(np.argmax(cumulative >= 0.90)) + 1
print("components needed for 90% of the variance:", k)

plt.figure(figsize=(7, 4))
n = len(features)
plt.bar(range(1, n + 1), 100 * ratio, label="explained % per component")
plt.plot(range(1, n + 1), 100 * cumulative, "o-", color="orange", label="cumulative %")
plt.axhline(90, color="grey", linestyle="--")
plt.xlabel("principal component"); plt.ylabel("% of variance"); plt.legend(); plt.title("Scree plot")
plt.show()
""")
md("**Step 5: project the data onto the kept components (11 features -> k numbers per pair)**")
code("""
W = eigenvector[:, :k]
Z = np.dot(C, W)
print(C.shape, "->", Z.shape)
print(Z[:5].round(3))
""")
code("""
colours = {"same": "tab:green", "variant": "tab:orange", "different": "tab:grey"}
plt.figure(figsize=(7, 5))
for lab, col in colours.items():
    part = Z[(Y == lab).values]
    plt.scatter(part[:, 0], part[:, 1], s=14, color=col, label=lab, alpha=0.7)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(); plt.title("Every pair of listings on the first two components")
plt.show()
""")
md("""
**What PC1 means:** its loadings (section 5) are positive for the distance-like features (`rank_min`, `rank_max`,
`phash_dist`) and negative for all the similarities (`img_cos`, `dino_cos`, `txt_cos`, `tfidf_cos`). So PC1 is a
**"how different do these two listings look and read"** axis. The average PC1 of each group:
""")
code("""
print(pd.Series(Z[:, 0], index=df.index).groupby(Y).mean().round(2))
""")
md("""
*Same* pairs sit furthest left, *different* pairs furthest right, *variants* in between, but the groups overlap a
lot: every pair comes from the same search (two iPhone 16 listings, or an iPhone 16 and an iPhone 16 Pro), so
these are hard cases, and one direction alone cannot separate them.
""")

md("## 4. Check with sklearn")
code("""
pca = PCA(0.90)
pca.fit(S)
print("components kept:", pca.n_components_)
print("explained variance (sklearn):", pca.explained_variance_.round(4))
print("explained variance (by hand):", eigvalues[:pca.n_components_].round(4))
print("ratio:", pca.explained_variance_ratio_.round(4))
""")
code("""
B = pca.transform(S)
# an eigenvector may come out with the opposite sign (v and -v are the same direction),
# so we compare absolute values
print("same components:", np.allclose(np.abs(pca.components_.T), np.abs(W)))
print("same projected data:", np.allclose(np.abs(B), np.abs(Z)))
""")

md("""
## 5. Which features matter most (loadings)

The **loadings** of a component are the entries of its eigenvector: how much each original feature contributes.
A large absolute loading = an important feature for that component.
""")
code("""
loadings_pc1 = pca.components_[0]
contributions = pd.Series(loadings_pc1, index=features)
print(contributions.abs().sort_values(ascending=False).round(3))
""")
code("""
loadings_pc2 = pca.components_[1]
contributions = pd.Series(loadings_pc2, index=features)
print(contributions.abs().sort_values(ascending=False).round(3))
""")
code("""
loadings_pc3 = pca.components_[2]
contributions = pd.Series(loadings_pc3, index=features)
print(contributions.abs().sort_values(ascending=False).round(3))
""")
code("""
loadings = pd.DataFrame(pca.components_.T, index=features,
                        columns=[f"PC{i+1}" for i in range(pca.n_components_)])
loadings.round(2)
""")
md("""
**Overall importance of each feature.** PCA does not delete features: every component is a mix of all 11.
(The ranking below is fairly even once the features are standardised: no single feature dominates.)
To rank the original features we weight each feature's absolute loading by how much variance its component
explains, and add over the kept components:

importance(feature) = sum over kept PCs of |loading| x explained-variance ratio
""")
code("""
importance = (loadings.abs() * pca.explained_variance_ratio_).sum(axis=1)
importance = (importance / importance.sum() * 100).sort_values(ascending=False)
print(importance.round(1))

importance.sort_values().plot(kind="barh", figsize=(7, 4.5), title="Feature importance from PCA (% of total)")
plt.xlabel("%"); plt.show()
""")

md("""
## 6. Should we reduce these features? (logistic regression before and after)

DupeLens's matcher is a logistic regression. We train it to tell *same product* (same or variant) from
*different*, once on all 11 standardised features and once on only the k principal components, and test both
on pairs it never saw (70% train / 30% test). The scaling and the PCA are fitted on the training part only.
""")
code("""
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

y = (Y != "different").astype(int)
X_train, X_test, y_train, y_test = train_test_split(A, y, test_size=0.3, random_state=42, stratify=y)

full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
reduced = make_pipeline(StandardScaler(), PCA(k), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
print("accuracy with all 11 features:", round(100 * full.score(X_test, y_test), 1), "%")
print(f"accuracy with {k} principal components:", round(100 * reduced.score(X_test, y_test), 1), "%")
""")
code("""
acc = []
for n in range(1, len(features) + 1):
    model = make_pipeline(StandardScaler(), PCA(n), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    acc.append(100 * model.score(X_test, y_test))
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(features) + 1), acc, "o-")
plt.xlabel("number of principal components kept"); plt.ylabel("test accuracy %")
plt.title("Accuracy against the number of components"); plt.show()
print([round(a, 1) for a in acc])
""")
md("""
**Result: for these features, PCA should not be used to reduce them.** Keeping 90% of the variance (8
components) lowers the test accuracy, and only all 11 components (which contain exactly the same information as
the 11 features, just rotated) give back the full accuracy.

**Why:** PCA is *unsupervised*: it keeps the directions where the data **varies** most, without looking at the
labels. The last, small-variance directions (PC9 to PC11, only 7.7% of the variance) still carry information that
decides *same vs different*: accuracy climbs from 61.2% with 8 components to 72.1% with all 11. So DupeLens's
logistic regression uses all 11 features as they are.
(Test set: 129 pairs, so a few points of difference is within normal variation; the drop of about 11 points is not.)

PCA **does** help where the data has hundreds of redundant numbers: the neural-network vectors (section 7).
""")

md("""
## 7. PCA inside DupeLens: compressing the neural-network vectors

Every listing photo is read by **CLIP** (512 numbers) and every title by **MPNet** (768 numbers). DupeLens
compresses each to **32 numbers** with PCA before comparing listings: fewer numbers, less noise, faster.
Here is the same manual PCA on the 698 Indian listings in our catalogue. (For a symmetric matrix such as a
covariance matrix, `np.linalg.eigh` gives the same result as `eig`, but always as real numbers.)
""")
code("""
def manual_pca_variance(X):
    M = np.mean(X.T, axis=1)
    C = X - M
    V = np.cov(C.T)
    eigvalues, eigenvector = np.linalg.eigh(V)
    eigvalues = eigvalues[::-1]                      # largest first
    return np.cumsum(eigvalues) / eigvalues.sum()

clip = np.load("data/clip_photo_vectors.npy")
mpnet = np.load("data/mpnet_title_vectors.npy")
print(clip.shape, mpnet.shape)
cum_clip = manual_pca_variance(clip)
cum_mpnet = manual_pca_variance(mpnet)
print("CLIP photos:  32 components keep", round(100 * cum_clip[31], 1), "% of the variance")
print("MPNet titles: 32 components keep", round(100 * cum_mpnet[31], 1), "% of the variance")
""")
code("""
plt.figure(figsize=(7, 4))
plt.plot(range(1, 201), 100 * cum_clip[:200], label="CLIP photo vectors (512)")
plt.plot(range(1, 201), 100 * cum_mpnet[:200], label="MPNet title vectors (768)")
plt.axvline(32, color="grey", linestyle="--")
plt.xlabel("number of components"); plt.ylabel("cumulative % of variance"); plt.legend()
plt.title("How much information 32 components keep"); plt.show()
""")
md("""
32 components (out of 512 or 768) keep about **71%** of the variance: the curve rises steeply for the first
components and then flattens, so the remaining hundreds of directions each add very little (mostly fine detail
and noise). The vectors become 16 to 24 times smaller, which makes comparing every pair of listings fast. 32 is the
size used by the trained matcher (validation F1 0.864). In the live system this PCA is fitted once on the 34,250
training listings, not on these 698.
""")

md("""
## 8. PCA inside DupeLens: the Genuine Look check

A scam listing often uses the real product's title with **another product's photo**. For each listing of a
product, PCA is fitted on the photos of all the **other** listings (leave-one-out), and we try to rebuild this
photo from the top 3 components. A photo of the real product is rebuilt well; another product's photo is not.

Data: the DINOv2 photo vectors (384 numbers) of the "iPhone 16" listings, plus one photo of Sony headphones that
we slipped in under the iPhone's name.
""")
code("""
G = pd.read_csv("data/genuine_demo.csv")
D = G[[f"d{i}" for i in range(384)]].values
D = D / np.linalg.norm(D, axis=1, keepdims=True)
print(D.shape)

def rebuild_error(x, others, k=3):
    M = np.mean(others.T, axis=1)
    C = others - M
    V = np.cov(C.T)
    eigvalues, eigenvector = np.linalg.eigh(V)
    W = eigenvector[:, ::-1][:, :k]                  # top-k components
    z = x - M
    rebuilt = np.dot(np.dot(z, W), W.T)               # project onto the components and back
    return np.linalg.norm(z - rebuilt)

errors = np.array([rebuild_error(D[i], np.delete(D, i, axis=0)) for i in range(len(D))])
""")
code("""
median = np.median(errors)
spread = np.median(np.abs(errors - median)) * 1.4826   # robust standard deviation (median absolute deviation)
z = (errors - median) / spread
genuine = 100 / (1 + np.exp(z - 2.5))                  # 50% exactly at the warning line z = 2.5

result = pd.DataFrame({"store": G["store"], "rebuild error": errors.round(4), "z": z.round(2),
                       "genuine look %": genuine.round(0), "flagged": z > 2.5})
result.sort_values("z", ascending=False)
""")
md("""
The photo we slipped in has the **largest** rebuild error and is flagged. Three real listings are flagged too, all
from less-known shops (an Etsy item, an import store and a small shop), whose photos are unlike the others. The
big store and the Apple authorised resellers (Reliance Digital, iCrescent, Unicorn) score 66 to 96%. On the website
this is the **Genuine Look %** and the x-axis of the **Scam Radar**; only less-known shops are ever warned about.
(Tested on 13 searches: wrong photos caught 98.4% of the time.)
""")

md("## 9. Summary")
code("""
print("Features: 12 measured, 1 constant dropped ->", len(features), "-> principal components kept (90% variance):", k)
print("Most important features (PCA):")
for name, value in importance.head(5).items():
    print(f"  {name:12s} {value:5.1f} %")
print("Least important features:")
for name, value in importance.tail(3).items():
    print(f"  {name:12s} {value:5.1f} %")
print("Accuracy, all 11 features:", round(100 * full.score(X_test, y_test), 1), "% | 8 components:",
      round(100 * reduced.score(X_test, y_test), 1), "% -> the matcher keeps all 11 features")
print("CLIP 512 -> 32 keeps", round(100 * cum_clip[31], 1), "% | MPNet 768 -> 32 keeps", round(100 * cum_mpnet[31], 1), "%")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/pca_dupelens.ipynb")
print("written")
