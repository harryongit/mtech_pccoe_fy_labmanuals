# K-Means Clustering: Working, Application, Visualization, and Evaluation

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------------------
# Step 1: Create Dataset
# -------------------------------
X, y_true = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.6,
    random_state=42
)

# -------------------------------
# Step 2: Apply K-Means
# -------------------------------
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# -------------------------------
# Step 3: Visualization
# -------------------------------
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200)
plt.title("K-Means Clustering Result")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# -------------------------------
# Step 4: Performance Evaluation
# -------------------------------
sil_score = silhouette_score(X, labels)

print("Number of clusters:", k)
print("Silhouette Score:", sil_score)

# -------------------------------
# Optional: Elbow Method
# -------------------------------
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.show()
