import numpy as np


class LDA:
    """Linear Discriminant Analysis with pooled covariance and shrinkage regularization."""

    def __init__(self, reg=0.05):
        self.reg = reg

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        within_scatter = np.zeros((n_features, n_features))
        means = []
        priors = []

        for c in self.classes_:
            X_c = X[y == c]
            mu_c = np.mean(X_c, axis=0)
            means.append(mu_c)
            priors.append(len(X_c) / n_samples)
            diff = X_c - mu_c
            within_scatter += diff.T @ diff

        cov = within_scatter / max(n_samples - len(self.classes_), 1)

        # Shrinkage regularization for numerical stability
        identity = np.eye(n_features)
        cov = (1.0 - self.reg) * cov + self.reg * (np.trace(cov) / n_features) * identity + 1e-5 * identity

        self.means_ = np.stack(means)
        self.priors_ = np.asarray(priors)

        # Weights: solve cov * w = means.T -> w = cov^-1 * means.T
        self.w_ = np.linalg.solve(cov, self.means_.T)
        self.offset_ = -0.5 * np.sum(self.means_ * self.w_.T, axis=1) + np.log(self.priors_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = X @ self.w_ + self.offset_
        return self.classes_[np.argmax(scores, axis=1)]
