"""
Process iMaterialist Fashion 2019 https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from iglovikov_helper_functions.utils.mask_utils import rle2mask


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder", type=Path, help="Path to folder with images")
    parser.add_argument("-l", "--label_path", type=Path, help="Path to csv with labels")
    parser.add_argument("-o", "--output_folder", type=Path, help="Path to the output folder")
    return parser.parse_args()


def main():
    args = get_args()

    output_image_folder = args.output_folder / "images"
    output_image_folder.mkdir(exist_ok=True, parents=True)

    output_label_folder = args.output_folder / "labels"
    output_label_folder.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(args.label_path)

    for file_name, dft in tqdm(df.groupby("ImageId")):

        if not (args.image_folder / file_name).exists():
            continue

        height = dft.iloc[0]["Height"]
        width = dft.iloc[0]["Width"]

        size = Image.open(args.image_folder / file_name).size

        if (width, height) != size:
            continue

        mask = np.zeros((height, width), dtype=np.uint8)

        for i in dft.index:
            seg = dft.loc[i, "EncodedPixels"]

            mask = mask | rle2mask(seg, (width, height))

        if mask.sum() == 0:
            continue

        shutil.copy(str(args.image_folder / file_name), str(output_image_folder / file_name))

        cv2.imwrite(str(output_label_folder / f"{Path(file_name).stem}.png"), mask * 255)


if __name__ == "__main__":
    main()
