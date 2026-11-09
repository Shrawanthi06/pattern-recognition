import numpy as np


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero for constant features
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class PCA:
    """Principal Component Analysis using eigenvalue decomposition."""

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        centered = X - self.mean_

        # Sample covariance matrix
        cov = (centered.T @ centered) / max(len(centered) - 1, 1)

        # Eigendecomposition (eigh returns real eigenvalues in ascending order)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort descending
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[sorted_indices], 0)

        self.components_ = eigenvectors[:, sorted_indices[: self.n_components]]
        self.explained_variance_ratio_ = (
            eigenvalues[: self.n_components] / max(eigenvalues.sum(), 1e-15)
        )
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
