"""
This script creates COCO annotations from deepfake annotations.

annotations present in format

<file_id>.json

[
    "file_path": video_file_path, # video_file_name = <video_id>.mp4
    "file_id": video_id, (could be broken)
    "bboxes": [
        {
            "frame_id": frame_id,
            "bbox": [x_min, y_min, x_max, y_max]
            "score": score  # float in [0, 1]
        }
    ]
    "landmarks": [...]
]

image_files:
<video_id>_<frame_id>.jpg

"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from iglovikov_helper_functions.utils.general_utils import group_by_key
from iglovikov_helper_functions.utils.image_utils import get_size


def fill_empty(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the gaps in the 'original' column with values from 'index'
    Args:
        df:
    Returns:
    """
    index = df["original"].isnull()
    df.loc[index, "original"] = df.loc[index, "index"]

    return df


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Map csv to COCO json")

    parser.add_argument("-id", "--deepfake_image_path", type=Path, help="Path to the jpg image files for deepfake.")
    parser.add_argument("-ld", "--deepfake_label_path", type=Path, help="Path to the json label files for deepfake.")

    parser.add_argument(
        "-io", "--openimages_image_path", type=Path, help="Path to the jpg image files for openimages."
    )
    parser.add_argument(
        "-lo", "--openimages_label_path", type=Path, help="Path to the json label files for openimages."
    )

    parser.add_argument("-m", "--label_mapper", type=Path, help="Path to the csv file that maps real to fake images.")
    parser.add_argument(
        "-o", "--output_path", type=Path, help="Path to the json file that maps real to fake images.", required=True
    )

    parser.add_argument("--exclude_folds", type=int, help="Folds that should be excluded..", nargs="*")

    parser.add_argument("-j", "--num_threads", type=int, help="The number of CPU threads", default=12)
    return parser.parse_args()


def generate_annotation_info(annotation_id, image_id, category_id, x_min, y_min, bbox_width, bbox_height):
    return {
        "segmentation": [],
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x_min, y_min, bbox_width, bbox_height],
        "iscrowd": 0,
        "area": bbox_width * bbox_height,
    }


def generate_image_info(image_id: str, file_path: str, image_width: int, image_height: int) -> dict:
    return {
        "id": image_id,
        "file_name": file_path,
        "width": image_width,
        "height": image_height,
    }


def main():
    args = parse_args()

    if not args.output_path.suffix == ".json":
        raise ValueError(f"Expected json file for output path, but got {args.output_path}.")

    coco_categories = [{"id": 1, "name": "is_face"}, {"id": 2, "name": "real"}, {"id": 3, "name": "fake"}]

    coco_images = []
    coco_annotations = []

    if args.deepfake_image_path is not None and args.deepfake_label_path is not None:
        deepfake_image_info, deepfake_annotations = process_deepfake(
            args.label_mapper, args.exclude_folds, args.deepfake_image_path, args.deepfake_label_path
        )

        coco_images += deepfake_image_info
        coco_annotations += deepfake_annotations

        print(f"Added {len(deepfake_annotations)} to deepfake.")

    if args.openimages_image_path is not None and args.openimages_label_path is not None:
        openimages_image_info, openimages_annotations = process_openimages(
            args.openimages_image_path, args.openimages_label_path
        )

        coco_images += openimages_image_info
        coco_annotations += openimages_annotations

        print(f"Added {len(openimages_annotations)} to openimages.")

    result = {
        "categories": coco_categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }

    with open(args.output_path, "w") as f:
        json.dump(result, f, sort_keys=True, indent=4)


def process_openimages(image_path: Path, label_path: Path) -> Tuple[List, List]:
    image_file_names = sorted(image_path.rglob("*.jpg"))

    print(f"Processing {len(image_file_names)} for open images.")

    coco_images: List[dict] = []
    coco_annotations: List[dict] = []

    for image_file_path in tqdm(image_file_names):
        if not image_file_path.exists():
            continue

        image_id = image_file_path.stem
        json_label_path = label_path / f"{image_id}.json"

        if not json_label_path.is_file():
            continue

        image_width, image_height = get_size(image_file_path)

        with open(json_label_path) as f:
            annotation = json.load(f)

        coco_images += [
            generate_image_info(
                image_id,
                "/".join([image_file_path.parent.parent.name, image_file_path.parent.name, image_file_path.name]),
                image_width,
                image_height,
            )
        ]
        for b, boxes in enumerate(annotation["bboxes"]):
            bbox = boxes["bbox"]

            x_min, y_min, x_max, y_max = bbox

            x_min = int(np.clip(x_min, 0, image_width))
            x_max = int(np.clip(x_max, 0, image_width))
            y_min = int(np.clip(y_min, 0, image_height))
            y_max = int(np.clip(y_max, 0, image_height))

            bbox_width = x_max - x_min
            if bbox_width <= 0:
                continue

            bbox_height = y_max - y_min
            if bbox_height <= 0:
                continue

            category_id = 1

            annotation_id = str(hash(f"{image_id}_{b}_{category_id}"))

            coco_annotations += [
                generate_annotation_info(annotation_id, image_id, category_id, x_min, y_min, bbox_width, bbox_height)
            ]

            category_id = 2

            annotation_id = str(hash(f"{image_id}_{b}_{category_id}"))

            coco_annotations += [
                generate_annotation_info(annotation_id, image_id, category_id, x_min, y_min, bbox_width, bbox_height)
            ]

    return coco_images, coco_annotations


def process_deepfake(label_mapper: Path, exclude_folds: list, image_path: Path, label_path: Path) -> Tuple[List, List]:
    if not label_mapper.suffix == ".csv":
        raise ValueError(f"Label mapper should be csv file, but we got {label_mapper}.")

    image_file_names = [str(x) for x in sorted(image_path.rglob("*.jpg"))]

    print(f"Processing {len(image_file_names)} for Deepfakes.")

    image_df = pd.DataFrame({"image_file_path": image_file_names})

    image_df["video_id"] = image_df["image_file_path"].str.extract(r"([a-z]+)_\d+\.jpg")
    image_df["frame_id"] = image_df["image_file_path"].str.extract(r"[a-z]+_(\d+)\.jpg").astype(int)

    id2label = {x.stem: x for x in label_path.rglob("*.json")}

    label_mapping = pd.read_csv(label_mapper)
    label_mapping = label_mapping[~label_mapping["fold"].isin(exclude_folds)]
    label_mapping = fill_empty(label_mapping)

    image_df = image_df[image_df["video_id"].isin(label_mapping["index"])].reset_index(drop=True)

    index2original = dict(zip(label_mapping["index"].values, label_mapping["original"].values))

    index2target = dict(zip(label_mapping["index"].values, label_mapping["target"].values))

    g = image_df.groupby("video_id")

    coco_images: List[dict] = []
    coco_annotations: List[dict] = []

    for video_id, df in tqdm(g):
        if video_id not in index2original:
            continue

        original = index2original[video_id]

        if original not in id2label:
            continue

        label_file_path = id2label[original]

        with open(label_file_path) as f:
            label = json.load(f)

        grouped_label = group_by_key(label["bboxes"], "frame_id")

        for i in df.index:
            image_file_path = Path(df.loc[i, "image_file_path"])

            if not image_file_path.exists():
                continue
            frame_id = df.loc[i, "frame_id"]
            image_width, image_height = get_size(image_file_path)

            image_id = f"{video_id}_{frame_id}"

            image_info = generate_image_info(
                image_id,
                "/".join([image_file_path.parent.parent.name, image_file_path.parent.name, image_file_path.name]),
                image_width,
                image_height,
            )

            for b, bbox in enumerate(grouped_label[frame_id]):
                x_min, y_min, x_max, y_max = bbox["bbox"]

                x_min = int(np.clip(x_min, 0, image_width))
                x_max = int(np.clip(x_max, 0, image_width))
                y_min = int(np.clip(y_min, 0, image_height))
                y_max = int(np.clip(y_max, 0, image_height))

                bbox_width = x_max - x_min
                if bbox_width <= 0:
                    continue

                bbox_height = y_max - y_min
                if bbox_height <= 0:
                    continue

                category_id = 1

                annotation_id = str(hash(f"{image_id}_{b}_{category_id}"))

                coco_annotations += [
                    generate_annotation_info(
                        annotation_id, image_id, category_id, x_min, y_min, bbox_width, bbox_height
                    )
                ]

                target = index2target[video_id]

                if target == 0:
                    category_id = 2
                elif target == 1:
                    category_id = 3
                else:
                    raise ValueError(f"Target should be 0 or 1, but we got {target}.")

                annotation_id = str(hash(f"{image_id}_{b}_{category_id}"))

                coco_annotations += [
                    generate_annotation_info(
                        annotation_id, image_id, category_id, x_min, y_min, bbox_width, bbox_height
                    )
                ]

            coco_images += [image_info]
    return coco_images, coco_annotations


if __name__ == "__main__":
    main()
