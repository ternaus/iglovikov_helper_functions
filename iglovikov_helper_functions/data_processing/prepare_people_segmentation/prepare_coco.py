"""
Process COCO
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mutils
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder", type=Path, help="Path to folder with images")
    parser.add_argument("-l", "--label_path", type=Path, help="Path to the json with label.")
    parser.add_argument("-o", "--output_folder", type=Path, help="Path to the output folder")
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.label_path) as f:
        labels = json.load(f)

    output_image_folder = args.output_folder / "images"
    output_image_folder.mkdir(exist_ok=True, parents=True)

    output_label_folder = args.output_folder / "labels"
    output_label_folder.mkdir(exist_ok=True, parents=True)

    df_annotations = pd.DataFrame.from_dict(labels["annotations"])
    df_annotations = df_annotations[df_annotations["category_id"] == 1]
    df_images = pd.DataFrame.from_dict(labels["images"])
    df = df_annotations.merge(df_images, left_on="image_id", right_on="id")

    for file_name, dft in tqdm(df.groupby("file_name")):
        height = dft.iloc[0]["height"]
        width = dft.iloc[0]["width"]

        mask = np.zeros((height, width), dtype=np.uint8)

        for i in dft.index:
            seg = dft.loc[i, "segmentation"]
            rles = mutils.frPyObjects(seg, height, width)
            ms = mutils.decode(rles)
            if len(ms.shape) != 3:
                mask = mask | ms
            else:
                mask = mask | ms.sum(axis=-1)

        shutil.copy(str(args.image_folder / file_name), str(output_image_folder / file_name))

        cv2.imwrite(str(output_label_folder / f"{Path(file_name).stem}.png"), mask * 255)


if __name__ == "__main__":
    main()
