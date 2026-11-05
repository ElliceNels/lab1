
import yaml

from PIL import Image
import numpy as np
from os import path as opath
import matplotlib

# Use a non-interactive backend so the script works in CI and headless shells.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mypylibs import trajectory
from mypylibs import grayscale
from mypylibs import filters
from mypylibs import thresholding
from mypylibs import edge_detection
from mypylibs import fourier

ROOT_IN = 'in'
ROOT_OUT = 'out'
EXT = 'png'

def main(case: int):

    img = Image.open(opath.join(ROOT_IN, f'{case:02}.{EXT}'))
    with open(opath.join(ROOT_IN, f'{case:02}.yaml')) as f:
        params = yaml.safe_load(f) or {}
    
    # Convert image to RGB, then to a NumPy array
    rgb = np.array(img.convert("RGB"))

    # 1. Convert RGB image to grayscale
    gray = grayscale.grayscale(rgb)

    # 2. Smooth with a Gaussian to suppress texture and small artifacts. The
    # per-image YAML can override the default sigma of 2.
    sigma = params.get("sigma", 2.0)
    blurred = filters.gaussian_blur(gray, sigma)

    # 3. Threshold the blurred grayscale image. Pixels darker than the chosen
    # value become foreground; the per-image YAML can override 240.
    threshold_value = params.get("threshold", 240)
    mask = thresholding.calculate(blurred, threshold=threshold_value)

    # 4. Extract the boundary of the thresholded silhouette
    edges = edge_detection.calculate(mask)

    # Save the original plus the four stages: (title, image, display maximum)
    panels = [
        ("Original", rgb, 255),
        ("1. Grayscale", gray, 255),
        (f"2. Gaussian blur (sigma={sigma:.1f})", blurred, 255),
        (f"3. Threshold mask (T={threshold_value:.1f})", mask, 1),
        ("4. Edge detection", edges, 1),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, panel, vmax) in zip(axes.flat, panels):
        ax.imshow(panel, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(opath.join(ROOT_OUT, f'{case:02}_transforms.{EXT}'), bbox_inches="tight")
    plt.close(fig)

    # 5. Calculate the ordered and resampled trajectory
    x_resampled, y_resampled = trajectory.calculate(edges)
    fig = trajectory.plot_trajectory(x_resampled, y_resampled)

    # Save the trajectory mosaic
    fig.savefig(opath.join(ROOT_OUT, f'{case:02}_trajectory.{EXT}'))
    plt.close(fig)

    # 6. Fourier reconstruction
    fourier.save_reconstruction_mosaic(
        x_resampled,
        y_resampled,
        opath.join(ROOT_OUT, f'{case:02}_final.{EXT}'),
    )

    # Bonus: animate the dominant rotating Fourier vectors.
    fourier.save_epicycle_animation(
        x_resampled,
        y_resampled,
        opath.join(ROOT_OUT, f'{case:02}.mp4'),
    )

if __name__ == '__main__':

    case = 0
    while True:
        try:
            main(case)
        except FileNotFoundError:
            print(f'No more cases found after case {case-1}. Exiting.')
            break
        case += 1
