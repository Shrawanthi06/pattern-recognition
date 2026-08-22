import math
import random

def train_val_test_split(x, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    N = len(x)
    if N != len(y):
        raise ValueError(f"Length mismatch: x has {N} items, y has {len(y)} items.")

    # Create an index list [0, 1, 2, ..., N-1]
    indices = list(range(N))

    # Shuffle indices reproducibly using the specified random_seed
    rng = random.Random(random_seed)
    rng.shuffle(indices)

    # Compute split cutoffs
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Gather data points for each split
    x_train = [x[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]

    x_val = [x[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    x_test = [x[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


class StandardScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        self.is_fitted = False

    def fit(self, x):
        """Computes mean and standard deviation from training data only."""
        N = len(x)
        self.mean = sum(x) / N
        variance = sum((val - self.mean) ** 2 for val in x) / N
        self.std = math.sqrt(variance) if variance > 1e-12 else 1.0
        self.is_fitted = True
        return self

    def transform(self, x):
        """Transforms data using the learned mean and standard deviation."""
        if not self.is_fitted:
            raise RuntimeError("StandardScaler must be fitted before calling transform().")
        return [(val - self.mean) / self.std for val in x]

    def fit_transform(self, x):
        """Fits to data and then returns the transformed version."""
        return self.fit(x).transform(x)


def create_polynomial_features(x, degree):
    N = len(x)
    X = []
    for i in range(N):
        val = x[i]
        row = [1.0]  # Column 0: x^0 = 1 (bias/intercept term)
        curr = 1.0
        for _ in range(1, degree + 1):
            curr *= val
            row.append(curr)
        X.append(row)
    return X


if __name__ == "__main__":
    print("Testing Preprocessing Module...")

    # Generate dummy data
    dummy_x = list(range(100))
    dummy_y = [2.0 * v + 1.0 for v in dummy_x]

    # Test split with seed 42
    (x_tr1, y_tr1), (x_v1, y_v1), (x_te1, y_te1) = train_val_test_split(dummy_x, dummy_y, 0.70, 0.15, 0.15, random_seed=42)
    print(f"Seed 42 - Train size: {len(x_tr1)}, Val size: {len(x_v1)}, Test size: {len(x_te1)}")
    print(f"Seed 42 - First 3 Train x values: {x_tr1[:3]}")

    # Test split with a different seed (e.g., 100) to see the difference
    (x_tr2, y_tr2), (x_v2, y_v2), (x_te2, y_te2) = train_val_test_split(dummy_x, dummy_y, 0.70, 0.15, 0.15, random_seed=100)
    print(f"Seed 100 - First 3 Train x values: {x_tr2[:3]}")

    # Test StandardScaler
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_tr1)
    print(f"Scaler Mean: {scaler.mean:.2f}, Std: {scaler.std:.2f}")

    # Test Polynomial Features (x=2.0, degree=3 -> [1.0, 2.0, 4.0, 8.0])
    X_poly = create_polynomial_features([2.0], degree=3)
    print(f"Polynomial design matrix for x=2.0 (deg=3): {X_poly[0]} (Expected: [1.0, 2.0, 4.0, 8.0])")

    print("All Preprocessing tests passed successfully!")