import numpy as np


class KMeans:
    """K-Means clustering algorithm using Euclidean distance."""

    def __init__(self, K=3, max_iter=100, tol=1e-4, seed=42):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centers = None
        self.n_iter_ = 0

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = len(X)
        k = min(self.K, n_samples)

        # Randomly choose K initial observations as centroids
        rng = np.random.default_rng(self.seed)
        chosen_indices = rng.choice(n_samples, k, replace=False)
        self.centers = X[chosen_indices].copy()

        for iteration in range(1, self.max_iter + 1):
            # Compute distance efficiently: ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c
            x_sq = np.sum(X ** 2, axis=1)[:, None]
            c_sq = np.sum(self.centers ** 2, axis=1)[None, :]
            distances = np.maximum(x_sq + c_sq - 2.0 * (X @ self.centers.T), 0)

            # Hard assignment of each observation to closest centroid
            labels = np.argmin(distances, axis=1)

            # Recompute centroids as the mean of assigned observations
            new_centers = self.centers.copy()
            for cluster_idx in range(k):
                mask = (labels == cluster_idx)
                if np.any(mask):
                    new_centers[cluster_idx] = np.mean(X[mask], axis=0)
                # Empty clusters retain their previous centroids

            # Stop when largest centroid movement is below tol
            shift = np.max(np.linalg.norm(new_centers - self.centers, axis=1))
            self.centers = new_centers
            self.n_iter_ = iteration

            if shift < self.tol:
                break

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        x_sq = np.sum(X ** 2, axis=1)[:, None]
        c_sq = np.sum(self.centers ** 2, axis=1)[None, :]
        distances = np.maximum(x_sq + c_sq - 2.0 * (X @ self.centers.T), 0)
        return np.argmin(distances, axis=1)

    def distortion(self, X):
        """Compute the total distortion (sum of squared distances to closest centroid)."""
        X = np.asarray(X, dtype=float)
        x_sq = np.sum(X ** 2, axis=1)[:, None]
        c_sq = np.sum(self.centers ** 2, axis=1)[None, :]
        distances = np.maximum(x_sq + c_sq - 2.0 * (X @ self.centers.T), 0)
        min_distances = np.min(distances, axis=1)
        return float(np.sum(min_distances))
