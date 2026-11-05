"""Find mask boundaries using simple horizontal and vertical differences."""

import numpy as np


def calculate(mask: np.ndarray):
    """Return a boolean image marking changes between foreground/background.

    The differences are the simple derivative filter [-1, 1], applied across
    columns and rows. A non-zero difference marks an edge.
    """
    image = np.asarray(mask, dtype=bool)
    if image.ndim != 2:
        raise ValueError("mask must be a two-dimensional image")

    horizontal_change = image[:, 1:] != image[:, :-1]
    vertical_change = image[1:, :] != image[:-1, :]
    edges = np.zeros_like(image)

    # Mark both pixels on each side of a foreground/background transition.
    edges[:, 1:] |= horizontal_change
    edges[:, :-1] |= horizontal_change
    edges[1:, :] |= vertical_change
    edges[:-1, :] |= vertical_change
    return edges
