# Re-import needed libraries after reset
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate synthetic 3D dataset
X_blob, _ = make_blobs(n_samples=300, centers=3, n_features=3, random_state=42)

# --- 3D Plot Before PCA ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot raw 3D data
ax.scatter(X_blob[:, 0], X_blob[:, 1], X_blob[:, 2], c='teal', s=40)
ax.set_title("Original Data in 3D")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
ax.grid(True)
plt.tight_layout()
plt.show()

# --- PCA Implementation ---
# Apply PCA to reduce from 3D to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_blob)

# Plot PCA result in 2D
plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='steelblue', s=40)
plt.title('PCA: Dimensionality Reduction from 3D to 2D')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

