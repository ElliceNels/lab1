
import numpy as np
from scipy.ndimage import convolve

def gaussian_kernel(sigma):

    radius = int(np.ceil(3 * sigma))  # Covers ~99.7% of the Gaussian
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    return k / k.sum()

def gaussian_blur(gray, sigma=2.0):

    k = gaussian_kernel(sigma)

    return convolve(gray.astype(np.float64), k, mode="reflect")
