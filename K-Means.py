from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

#Generation of sample data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

#Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
y_means = kmeans.fit_predict(X)

#Visualisation of the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_means, cmap='viridis', s=50)
plt.title("Clusters Found by K-Means")
plt.xlabel('Feature 1 (X-axis)')
plt.ylabel('Feature 2 (y-axis)')
plt.colorbar(label='Cluster Label')
plt.grid(True)
plt.show()