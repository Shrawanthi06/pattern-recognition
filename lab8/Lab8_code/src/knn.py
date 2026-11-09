import numpy as np


class KNN:
    """K-Nearest Neighbors classifier using Euclidean distance."""

    def __init__(self, k=5, chunk=128):
        self.k = k
        self.chunk = chunk

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(y)
        self.train_sq_norms = np.sum(self.X_train ** 2, axis=1)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        predictions = []
        k = min(self.k, len(self.X_train))

        # Process in chunks to maintain low memory usage
        for i in range(0, len(X), self.chunk):
            batch = X[i : i + self.chunk]
            batch_sq_norms = np.sum(batch ** 2, axis=1)[:, None]
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
            distances = np.maximum(
                batch_sq_norms + self.train_sq_norms[None, :] - 2.0 * (batch @ self.X_train.T), 0
            )

            # Find k nearest neighbors for each sample in the batch
            nearest_ids = np.argpartition(distances, k - 1, axis=1)[:, :k]

            for row in nearest_ids:
                neighbor_labels = self.y_train[row]
                votes = [np.sum(neighbor_labels == c) for c in self.classes_]
                predictions.append(self.classes_[np.argmax(votes)])

        return np.asarray(predictions)
