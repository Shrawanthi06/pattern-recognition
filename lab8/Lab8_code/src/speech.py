import numpy as np


class SpeechSummary:
    """Extract fixed-length summary vector (mean + std) from frame sequences."""

    @staticmethod
    def transform(sequences):
        summaries = []
        for seq in sequences:
            seq_mean = np.mean(seq, axis=0)
            seq_std = np.std(seq, axis=0)
            summaries.append(np.concatenate([seq_mean, seq_std]))
        return np.asarray(summaries, dtype=float)


class FrameTransformer:
    """Apply fitted scaler and optional PCA to sequence frames."""

    def __init__(self, scaler=None, pca=None):
        self.scaler = scaler
        self.pca = pca

    def transform(self, sequences):
        transformed = []
        for seq in sequences:
            s = np.asarray(seq, dtype=float)
            if self.scaler is not None:
                s = self.scaler.transform(s)
            if self.pca is not None:
                s = self.pca.transform(s)
            transformed.append(s)
        return transformed
