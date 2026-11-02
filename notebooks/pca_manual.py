"""PCA by hand on DupeLens's pair features (the same steps as in class, no library PCA).
Run from the notebooks folder:  python pca_manual.py
"""
import numpy as np
import pandas as pd

# 1. data: 430 pairs of listings, 12 features each
df = pd.read_csv("data/pair_features.csv")
features = ["img_cos", "txt_cos", "tfidf_cos", "phash_dist", "num_match", "num_conflict",
            "mutual", "rank_min", "rank_max", "len_ratio", "dino_cos", "colour_dist"]
A = df[features]

# 2. drop features that never change (zero variance = no information)
features = [f for f in features if A[f].std() > 0]
A = df[features]
print("features used:", len(features), features)

# 3. standardise: every feature gets mean 0 and standard deviation 1
S = (A - A.mean()) / A.std()

# 4. mean and centring
M = np.mean(S.values.T, axis=1)
C = S.values - M

# 5. covariance matrix
V = np.cov(C.T)
print("covariance matrix:", V.shape)

# 6. eigenvalues and eigenvectors, largest first
eigvalues, eigenvector = np.linalg.eig(V)
idx = eigvalues.argsort()[::-1]
eigvalues = eigvalues[idx]
eigenvector = eigenvector[:, idx]
print("eigenvalues:", eigvalues.round(3))

# 7. explained variance and how many components keep 90%
ratio = eigvalues / eigvalues.sum()
cumulative = np.cumsum(ratio)
for i in range(len(ratio)):
    print(f"PC{i + 1}: {100 * ratio[i]:5.1f} %   cumulative {100 * cumulative[i]:5.1f} %")
k = int(np.argmax(cumulative >= 0.90)) + 1
print("components for 90% of the variance:", k)

# 8. project the data onto the first k components
W = eigenvector[:, :k]
Z = np.dot(C, W)
print("data:", C.shape, "->", Z.shape)

# 9. loadings: which features make up each component
for i in range(3):
    contributions = pd.Series(eigenvector[:, i], index=features)
    print(f"\nPC{i + 1} loadings (largest first):")
    print(contributions.abs().sort_values(ascending=False).round(3))

# 10. overall importance of each feature over the kept components
loadings = pd.DataFrame(np.abs(eigenvector[:, :k]), index=features)
importance = (loadings * ratio[:k]).sum(axis=1)
importance = (importance / importance.sum() * 100).sort_values(ascending=False)
print("\nfeature importance (%):")
print(importance.round(1))
