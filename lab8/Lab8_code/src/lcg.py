import numpy as np


class LCG:
    """Linear Congruential Generator (LCG) for reproducible pseudo-random numbers.
    """

    def __init__(self, seed=42, a=1664525, c=1013904223, m=2**32):
        self.state = int(seed)
        self.a = a
        self.c = c
        self.m = m

    def random(self):
        """Generate a pseudo-random float in [0.0, 1.0)."""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

    def integers(self, low, high=None):
        """Generate a pseudo-random integer in [low, high)."""
        if high is None:
            low, high = 0, low
        return low + int(self.random() * (high - low))

    def permutation(self, n):
        """Generate a random permutation of integers 0..n-1 or an array using Fisher-Yates shuffle."""
        if isinstance(n, (int, np.integer)):
            arr = np.arange(n)
        else:
            arr = np.array(n, copy=True)
        for i in range(len(arr) - 1, 0, -1):
            j = int(self.random() * (i + 1))
            arr[i], arr[j] = arr[j], arr[i]
        return arr

    def choice(self, a, size=None, replace=True):
        """Randomly sample from an array or integer range."""
        if isinstance(a, (int, np.integer)):
            pool = np.arange(a)
        else:
            pool = np.asarray(a)
        if size is None:
            return pool[self.integers(len(pool))]
        if replace:
            indices = [self.integers(len(pool)) for _ in range(size)]
            return pool[indices]
        perm = self.permutation(len(pool))
        return pool[perm[:size]]
