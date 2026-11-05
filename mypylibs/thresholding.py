import numpy as np


def calculate(gray: np.ndarray, threshold: float = 240):
    """Mark pixels darker than ``threshold`` as foreground."""
    return np.asarray(gray) < threshold
