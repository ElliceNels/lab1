
import yaml

from PIL import Image
import numpy as np
from os import path as opath
from mypylibs import trajectory
from mypylibs import grayscale
from mypylibs import thresholding
import matplotlib.pyplot as plt

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

    # 2. Blur

    # 3. Threshold the (later: blurred) grayscale image. Pixels darker than
    # the chosen value become foreground; the per-image YAML can override 240.
    threshold_value = params.get("threshold", 240)
    mask = thresholding.calculate(gray, threshold=threshold_value)

    # Save the grayscale and threshold stages for visual inspection.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Grayscale")
    axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Threshold mask (T={threshold_value:.1f})")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(opath.join(ROOT_OUT, f'{case:02}_threshold_preview.png'))
    plt.close(fig)
    # Expected outputs per case
    # plt.savefig(opath.join(ROOT_OUT, f'{case:02}_transforms.{EXT}'))
    # plt.savefig(opath.join(ROOT_OUT, f'{case:02}_trajectory.{EXT}'))
    # plt.savefig(opath.join(ROOT_OUT, f'{case:02}_final.{EXT}'))

if __name__ == '__main__':

    case = 0
    while True:
        try:
            main(case)
        except FileNotFoundError:
            print(f'No more cases found after case {case-1}. Exiting.')
            break
        case += 1
