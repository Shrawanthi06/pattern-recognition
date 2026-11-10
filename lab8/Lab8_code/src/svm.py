import numpy as np
from .lcg import LCG


class BinarySVM:
    """Binary Support Vector Machine using simplified Sequential Minimal Optimization (SMO)."""

    def __init__(self, kernel="rbf", C=1.0, gamma="scale", seed=42, max_passes=5, max_iter=120):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.seed = seed
        self.max_passes = max_passes
        self.max_iter = max_iter

    def _kernel(self, A, B):
        dot = A @ B.T
        if self.kernel == "linear":
            return dot
        if self.kernel != "rbf":
            raise ValueError(f"Unsupported kernel: {self.kernel}")
        sq_dist = np.maximum(
            np.sum(A ** 2, axis=1)[:, None] + np.sum(B ** 2, axis=1)[None, :] - 2.0 * dot, 0
        )
        return np.exp(-self.gamma_ * sq_dist)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        if self.gamma == "scale":
            self.gamma_ = 1.0 / (X.shape[1] * max(float(X.var()), 1e-12))
        else:
            self.gamma_ = float(self.gamma)

        K = self._kernel(X, X)
        alphas = np.zeros(n)
        b = 0.0
        rng = LCG(self.seed)

        passes = 0
        iteration = 0
        f = np.zeros(n)  # Decision values for training points

        while passes < self.max_passes and iteration < self.max_iter:
            num_changed = 0
            for i in rng.permutation(n):
                Ei = f[i] - y[i]
                # Check KKT conditions violation
                if not ((y[i] * Ei < -1e-3 and alphas[i] < self.C - 1e-7) or 
                        (y[i] * Ei > 1e-3 and alphas[i] > 1e-7)):
                    continue

                # Select random j != i
                j = int(rng.integers(n - 1))
                if j >= i:
                    j += 1
                Ej = f[j] - y[j]

                ai_old, aj_old = alphas[i], alphas[j]

                # Compute clipping bounds L and H
                if y[i] != y[j]:
                    L = max(0.0, aj_old - ai_old)
                    H = min(self.C, self.C + aj_old - ai_old)
                else:
                    L = max(0.0, ai_old + aj_old - self.C)
                    H = min(self.C, ai_old + aj_old)

                if H - L < 1e-12:
                    continue

                # Second derivative of objective function along the line
                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                if eta >= -1e-12:
                    continue

                # Compute and clip new alpha_j
                new_aj = np.clip(aj_old - y[j] * (Ei - Ej) / eta, L, H)
                if abs(new_aj - aj_old) < 1e-6:
                    continue

                # Compute new alpha_i
                new_ai = ai_old + y[i] * y[j] * (aj_old - new_aj)

                # Update bias b
                b1 = b - Ei - y[i] * (new_ai - ai_old) * K[i, i] - y[j] * (new_aj - aj_old) * K[i, j]
                b2 = b - Ej - y[i] * (new_ai - ai_old) * K[i, j] - y[j] * (new_aj - aj_old) * K[j, j]

                if 1e-7 < new_ai < self.C - 1e-7:
                    new_b = b1
                elif 1e-7 < new_aj < self.C - 1e-7:
                    new_b = b2
                else:
                    new_b = (b1 + b2) / 2.0

                # Update decision vector cache
                f += (new_ai - ai_old) * y[i] * K[:, i] + (new_aj - aj_old) * y[j] * K[:, j] + (new_b - b)
                alphas[i] = new_ai
                alphas[j] = new_aj
                b = new_b
                num_changed += 1

            passes = (passes + 1) if num_changed == 0 else 0
            iteration += 1

        # Store support vectors
        is_sv = alphas > 1e-7
        self.support_ = X[is_sv]
        self.dual_ = alphas[is_sv] * y[is_sv]
        self.b_ = b
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return self._kernel(X, self.support_) @ self.dual_ + self.b_


class SVM:
    """Multiclass One-vs-Rest Support Vector Machine."""

    def __init__(self, kernel="rbf", C=1.0, gamma="scale", seed=42, max_train=2500):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.seed = seed
        self.max_train = max_train

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        # Subsample if dataset exceeds computational cap for SMO
        if len(X) > self.max_train:
            rng = LCG(self.seed)
            sampled_ids = []
            for c in self.classes_:
                c_ids = np.flatnonzero(y == c)
                n_take = max(1, int(round(self.max_train * len(c_ids) / len(y))))
                sampled_ids.extend(rng.choice(c_ids, min(n_take, len(c_ids)), replace=False))
            sampled_ids = np.asarray(sampled_ids)
            X = X[sampled_ids]
            y = y[sampled_ids]

        self.train_count_ = len(X)
        self.models_ = []

        for c in self.classes_:
            binary_y = np.where(y == c, 1.0, -1.0)
            model = BinarySVM(
                kernel=self.kernel, C=self.C, gamma=self.gamma, seed=self.seed
            ).fit(X, binary_y)
            self.models_.append(model)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.column_stack([model.decision_function(X) for model in self.models_])
        return self.classes_[np.argmax(scores, axis=1)]
