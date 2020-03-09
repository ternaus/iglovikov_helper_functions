"""
This script creates object detection annotations

annotations present in format

<file_id>_faces.json

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
]

image_files:
<video_id>_<frame_id>.jpg
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from iglovikov_helper_functions.utils.general_utils import group_by_key

from iglovikov_helper_functions.utils.image_utils import get_size
from joblib import Parallel, delayed


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

    parser.add_argument("-i", "--image_path", type=Path, help="Path to the jpg image files.", required=True)
    parser.add_argument("-l", "--label_path", type=Path, help="Path to the json label files.", required=True)
    parser.add_argument(
        "-m", "--label_mapper", type=Path, help="Path to the csv file that maps real to fake images.", required=True
    )
    parser.add_argument(
        "-o", "--output_path", type=Path, help="Path to the json file that maps real to fake images.", required=True
    )

    parser.add_argument("-j", "--num_threads", type=int, help="The number of CPU threads", default=12)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.label_mapper.suffix == ".csv":
        raise ValueError(f"Label mapper should be csv file, but we got {args.label_mapper}.")

    image_file_names = [str(x) for x in sorted(args.image_path.rglob("*.jpg"))]

    image_shape = Parallel(n_jobs=args.num_threads, prefer="threads")(
        delayed(get_size)(x) for x in tqdm(image_file_names)
    )

    image_df = pd.DataFrame({"image_file_path": image_file_names, "size": image_shape})

    image_df["video_id"] = image_df["image_file_path"].str.extract(r"([a-z]+)_\d+\.jpg")
    image_df["frame_id"] = image_df["image_file_path"].str.extract(r"[a-z]+_(\d+)\.jpg").astype(int)

    id2label = {x.stem.replace("_faces", ""): x for x in args.label_path.rglob("*.json")}

    label_mapping = pd.read_csv(args.label_mapper)

    label_mapping = fill_empty(label_mapping)

    index2original = dict(zip(label_mapping["index"].values, label_mapping["original"].values))

    g = image_df.groupby("video_id")

    coco_images = []
    coco_annotations = []

    for video_id, df in tqdm(g):
        original = index2original[video_id]

        label_file_path = id2label[original]

        with open(label_file_path) as f:
            label = json.load(f)

        grouped_label = group_by_key(label["bboxes"], "frame_id")

        for i in df.index:
            frame_id = df.loc[i, "frame_id"]
            image_width, image_height = df.loc[i, "size"]

            image_id = f"{video_id}_{frame_id}"

            image_file_path = Path(df.loc[i, "image_file_path"])

            image_info = {
                "id": image_id,
                "file_name": str(image_file_path.parent.name + "/" + image_file_path.name),
                "width": image_width,
                "height": image_height,
            }

            for b, bbox in enumerate(grouped_label[frame_id]):
                x_min, y_min, x_max, y_max = bbox["bbox"]

                bbox_width = x_max - x_min
                if bbox_width < 0:
                    continue

                bbox_height = y_max - y_min
                if bbox_height < 0:
                    continue

                annotation_id = str(hash(f"{video_id}_{frame_id}_{b}"))

                annotation_info = {
                    "segmentation": [],
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # We have only one category, faces
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "iscrowd": 0,
                    "area": bbox_width * bbox_height,
                }

                coco_annotations += [annotation_info]

            coco_images += [image_info]

    coco_categories = [{"id": 1, "name": "face"}]

    output_coco_annotations = {
        "categories": coco_categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }

    output_folder = args.output_path.parent

    output_folder.mkdir(exist_ok=True, parents=True)

    with open(args.output_path, "w") as f:
        json.dump(output_coco_annotations, f, sort_keys=True, indent=2)


if __name__ == "__main__":
    main()
