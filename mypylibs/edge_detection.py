import numpy as np
from scipy.ndimage import convolve

LAPLACIAN = np.array([[0,  1, 0],
                      [1, -4, 1],
                      [0,  1, 0]])

def calculate(mask: np.ndarray):
    image = np.asarray(mask, dtype=bool)
    if image.ndim != 2:
        raise ValueError("mask must be a two-dimensional image")

    response = convolve(image.astype(int), LAPLACIAN, mode="constant", cval=0)
    edges = image & (response < 0)
    return edges

