import numpy as np
from .lcg import LCG


def logsumexp(a, axis=1):
    """Numerically stable log-sum-exp."""
    max_val = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(
        max_val + np.log(np.sum(np.exp(a - max_val), axis=axis, keepdims=True) + 1e-300),
        axis=axis,
    )


class GMM:
    """Gaussian Mixture Model with diagonal covariance estimated via EM."""

    def __init__(self, K=3, reg=1e-3, max_iter=100, tol=1e-4, seed=42):
        self.K = K
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.log_likelihoods = []

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        k = min(self.K, n_samples)

        rng = LCG(self.seed)
        chosen_indices = rng.choice(n_samples, k, replace=False)
        self.means_ = X[chosen_indices].copy()
        self.vars_ = np.tile(np.maximum(np.var(X, axis=0), self.reg), (k, 1))
        self.weights_ = np.full(k, 1.0 / k)

        self.log_likelihoods = []
        prev_score = -np.inf

        for _ in range(self.max_iter):
            # E-step: compute responsibilities
            log_comp = self._log_components(X)
            log_norm = logsumexp(log_comp, axis=1)
            resp = np.exp(log_comp - log_norm[:, None])
            nk = np.sum(resp, axis=0) + 1e-10

            # M-step: update parameters
            self.weights_ = nk / np.sum(nk)
            self.means_ = (resp.T @ X) / nk[:, None]
            self.vars_ = np.maximum((resp.T @ (X * X)) / nk[:, None] - self.means_ ** 2, self.reg)

            # Check convergence
            score = float(np.mean(logsumexp(self._log_components(X), axis=1)))
            self.log_likelihoods.append(score)
            if abs(score - prev_score) < self.tol:
                break
            prev_score = score

        return self

    def _log_components(self, X):
        """Compute log p(x, k) for each mixture component k."""
        diff = np.asarray(X)[:, None, :] - self.means_[None, :, :]
        log_prob = -0.5 * (
            np.sum(np.log(2.0 * np.pi * self.vars_), axis=1)[None, :]
            + np.sum((diff ** 2) / self.vars_[None, :, :], axis=2)
        )
        return np.log(self.weights_ + 1e-300)[None, :] + log_prob

    def score_samples(self, X):
        """Compute log p(x) for each sample."""
        return logsumexp(self._log_components(X), axis=1)


class GMMClassifier:
    """Class-conditional GMM classifier fitting one GMM per class."""

    def __init__(self, K=3, reg=1e-3, max_iter=100, tol=1e-4, seed=42):
        self.K = K
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_ = {}
        self.log_priors_ = {}

        for c in self.classes_:
            X_c = X[y == c]
            model = GMM(
                K=self.K, reg=self.reg, max_iter=self.max_iter, tol=self.tol, seed=self.seed
            ).fit(X_c)
            self.models_[c] = model
            self.log_priors_[c] = np.log(len(X_c) / len(y))

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.column_stack([
            self.models_[c].score_samples(X) + self.log_priors_[c]
            for c in self.classes_
        ])
        return self.classes_[np.argmax(scores, axis=1)]


# Alias for backward compatibility
ClassConditionalGMM = GMMClassifier


class FrameGMM:
    """Class-conditional GMM for variable-length frame sequences (speech)."""

    def __init__(self, K=3, reg=1e-3, max_iter=80, tol=1e-4, seed=42):
        self.K = K
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def fit(self, sequences, labels):
        labels = np.asarray(labels)
        self.classes_ = np.unique(labels)
        self.models_ = {}
        self.log_priors_ = {}

        for c in self.classes_:
            class_frames = np.concatenate([seq for seq, y in zip(sequences, labels) if y == c])
            model = GMM(
                K=self.K, reg=self.reg, max_iter=self.max_iter, tol=self.tol, seed=self.seed
            ).fit(class_frames)
            self.models_[c] = model
            self.log_priors_[c] = np.log(np.mean(labels == c))

        return self

    def predict(self, sequences):
        scores = np.asarray([
            [self.models_[c].score_samples(seq).mean() + self.log_priors_[c] for c in self.classes_]
            for seq in sequences
        ])
        return self.classes_[np.argmax(scores, axis=1)]
