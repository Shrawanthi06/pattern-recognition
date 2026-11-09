import numpy as np


def dtw_cost(a, b, window=25):
    """Compute normalized DTW cost between sequences a and b with Sakoe-Chiba band."""
    n, m = len(a), len(b)
    radius = max(window, abs(n - m)) if window is not None else max(n, m)

    # Frame-to-frame Euclidean distance matrix
    a_sq = np.sum(a ** 2, axis=1)[:, None]
    b_sq = np.sum(b ** 2, axis=1)[None, :]
    dist = np.sqrt(np.maximum(a_sq + b_sq - 2.0 * (a @ b.T), 0))

    # DP cost and path length tables
    costs = np.full((n + 1, m + 1), np.inf)
    lengths = np.zeros((n + 1, m + 1), dtype=int)
    costs[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - radius)
        j_end = min(m, i + radius) + 1
        for j in range(j_start, j_end):
            c_diag = costs[i - 1, j - 1]
            c_up = costs[i - 1, j]
            c_left = costs[i, j - 1]

            if c_diag <= c_up and c_diag <= c_left:
                costs[i, j] = dist[i - 1, j - 1] + c_diag
                lengths[i, j] = lengths[i - 1, j - 1] + 1
            elif c_up <= c_left:
                costs[i, j] = dist[i - 1, j - 1] + c_up
                lengths[i, j] = lengths[i - 1, j] + 1
            else:
                costs[i, j] = dist[i - 1, j - 1] + c_left
                lengths[i, j] = lengths[i, j - 1] + 1

    return costs[n, m] / max(1, lengths[n, m])


class DTWClassifier:
    """1-Nearest-Neighbor DTW classifier using selected reference templates."""

    def __init__(self, references_per_class=10, window=25):
        self.references_per_class = references_per_class
        self.window = window

    def fit(self, X, y):
        self.references_ = []
        y = np.asarray(y)
        for cls in np.unique(y):
            indices = np.flatnonzero(y == cls)
            n_refs = min(len(indices), self.references_per_class)
            # Pick evenly spaced template references
            chosen = indices[np.linspace(0, len(indices) - 1, n_refs, dtype=int)]
            for idx in chosen:
                self.references_.append((X[idx], cls))
        return self

    def predict(self, X):
        predictions = []
        for seq in X:
            best_cost = float("inf")
            best_class = None
            for ref_seq, ref_class in self.references_:
                cost = dtw_cost(seq, ref_seq, window=self.window)
                if cost < best_cost:
                    best_cost = cost
                    best_class = ref_class
            predictions.append(best_class)
        return np.asarray(predictions)
