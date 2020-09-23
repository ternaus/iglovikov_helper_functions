"""
Process Pascal VOC
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PERSON_PIXELS = (128, 128, 192)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=Path, help="Path to folder with data.")
    parser.add_argument("-o", "--output_folder", type=Path, help="Path to the output folder.")
    return parser.parse_args()


def main():
    args = get_args()

    train_file_ids = pd.read_csv(args.data_folder / "ImageSets" / "Segmentation" / "train.txt", header=None).values.T[
        0
    ]
    val_file_ids = pd.read_csv(args.data_folder / "ImageSets" / "Segmentation" / "val.txt", header=None).values.T[0]

    samples = [(train_file_ids, "training"), (val_file_ids, "validation")]

    image_path = args.data_folder / "JPEGImages"
    segmentation_path = args.data_folder / "SegmentationClass"

    for file_ids, set_type in samples:
        output_image_folder = args.output_folder / set_type / "pascal_voc" / "images"
        output_image_folder.mkdir(exist_ok=True, parents=True)

        output_label_folder = args.output_folder / set_type / "pascal_voc" / "labels"
        output_label_folder.mkdir(exist_ok=True, parents=True)

        for file_id in tqdm(file_ids):
            mask = cv2.imread(str(args.data_folder / segmentation_path / f"{file_id}.png"))

            mask = (mask == PERSON_PIXELS).all(axis=-1).astype(np.uint8)

            if mask.sum() == 0:
                continue

            shutil.copy(str(image_path / f"{file_id}.jpg"), str(output_image_folder / f"{file_id}.jpg"))
            cv2.imwrite(str(output_label_folder / f"{file_id}.png"), mask * 255)


if __name__ == "__main__":
    main()
