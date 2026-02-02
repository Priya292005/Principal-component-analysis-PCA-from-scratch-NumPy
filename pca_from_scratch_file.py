import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SklearnPCA

# -----------------------------
# 1. Generate synthetic dataset
# -----------------------------
np.random.seed(42)
n_samples = 200
n_features = 10

X = np.random.randn(n_samples, n_features)

# -----------------------------
# 2. Standardization
# -----------------------------
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_stdized = (X - X_mean) / X_std

# -----------------------------
# 3. Covariance matrix
# -----------------------------
cov_matrix = np.cov(X_stdized, rowvar=False)

# -----------------------------
# 4. Eigen decomposition
# -----------------------------
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

# -----------------------------
# 5. Explained variance ratio
# -----------------------------
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

print("Scratch PCA Explained Variance Ratio:")
print(explained_variance_ratio)

# -----------------------------
# 6. Projection for K=2
# -----------------------------
K = 2
W_2 = eigenvectors[:, :K]
X_pca_2 = X_stdized @ W_2

# -----------------------------
# 7. Projection for K=3 (validation path)
# -----------------------------
K3 = 3
W_3 = eigenvectors[:, :K3]
X_pca_3 = X_stdized @ W_3

# -----------------------------
# 8. Compare with sklearn PCA
# -----------------------------
sk_pca = SklearnPCA(n_components=3)
X_sk = sk_pca.fit_transform(X_stdized)

print("\nSklearn PCA Explained Variance Ratio:")
print(sk_pca.explained_variance_ratio_)

print("\nDifference between Scratch and Sklearn (first 3 components):")
print(explained_variance_ratio[:3] - sk_pca.explained_variance_ratio_)

# -----------------------------
# 9. 2D Visualization
# -----------------------------
plt.figure()
plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D PCA Projection (K=2)")
plt.show()
