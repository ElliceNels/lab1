
import yaml

from PIL import Image
import numpy as np
from os import path as opath
from mypylibs import trajectory
import matplotlib.pyplot as plt

ROOT_IN = 'in'
ROOT_OUT = 'out'
EXT = 'png'

def main(case: int):

    img = Image.open(opath.join(ROOT_IN, f'{case:02}.{EXT}'))
    with open(opath.join(ROOT_IN, f'{case:02}.yaml')) as f:
        params = yaml.safe_load(f)
    
    # 1. Convert image to RGB, then to a NumPy array
    rgb = np.array(img.convert("RGB"))

    # 2. Convert RGB image to grayscale
    gray = (
        0.299 * rgb[:, :, 0] +
        0.587 * rgb[:, :, 1] +
        0.114 * rgb[:, :, 2]
    )
    # Display the grayscale image
    plt.figure(figsize=(6, 6))

    plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(opath.join(ROOT_OUT, f'{case:02}_grayscale_preview.png'))
    plt.close()
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
