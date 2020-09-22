"""
Process Mapillary vistas 1.2 commercial
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from iglovikov_helper_functions.utils.image_utils import load_rgb

PERSON_PIXELS = (220, 20, 60)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder", type=Path, help="Path to folder with images")
    parser.add_argument("-l", "--label_folder", type=Path, help="Path to the folder with labels")
    parser.add_argument("-o", "--output_folder", type=Path, help="Path to the output folder")
    return parser.parse_args()


def main():
    args = get_args()
    label_files = args.label_folder.rglob("*.png")

    output_image_folder = args.output_folder / "images"
    output_image_folder.mkdir(exist_ok=True, parents=True)

    output_label_folder = args.output_folder / "labels"
    output_label_folder.mkdir(exist_ok=True, parents=True)

    for label_file_name in tqdm(sorted(label_files)):
        label = load_rgb(image_path=label_file_name, lib="cv2")
        mask = (label == PERSON_PIXELS).all(axis=-1).astype(np.uint8)

        if mask.sum() == 0:
            continue

        shutil.copy(
            str(args.image_folder / f"{label_file_name.stem}.jpg"),
            str(output_image_folder / f"{label_file_name.stem}.jpg"),
        )
        cv2.imwrite(str(output_label_folder / label_file_name.name), mask * 255)


if __name__ == "__main__":
    main()
