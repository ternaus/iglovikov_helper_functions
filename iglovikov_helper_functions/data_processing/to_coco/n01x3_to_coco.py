"""
Map predictions from format:

ground_truth = {image_001.jpg: [[bottle, 6, 234, 39, 128], ...], ...}
predictions = {image_001.jpg: [[bottle, 0.14981, 80, 1, 295, 500], ...], ...}

where bounding box coordinates are represented as [x_min, y_min, x_max, y_max]

to simplified COCO format that should be enough to use pycocotools.


For ground truth:


{
    "images": [
        {
        "id": image_id,
        "file_name": image_name
        }
    ],
    "categories": [
        {
        "id": category_id,
        "name": category_name,
        }
    ],
    "annotations": [
        {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x_min, y_min, width, height]
        }
    ]

}


For predictions:
[
{
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x_min, y_min, width, height]
        "score": score
        }

]

WARNING! To make ground_truth and predictions consistent make sure that they have the same number of classes!


Based on the request from Arthur Kuzin (n01z3).
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Map csv to COCO json")

    parser.add_argument("-i", "--input_file", type=str, help="Path to the input json file.", required=True)
    parser.add_argument("-o", "--output_file", type=Path, help="Path to the output json file.", required=True)
    return parser.parse_args()


def json2df(input_file):
    dfs = []

    for file_name, bboxes in input_file.items():
        if len(bboxes[0]) == 5:
            columns = ["name", "x_min", "y_min", "x_max", "y_max"]
        elif len(bboxes[0]) == 6:
            columns = ["name", "score", "x_min", "y_min", "x_max", "y_max"]
        else:
            raise NotImplementedError()

        df = pd.DataFrame(bboxes, columns=columns)

        df["image_name"] = file_name

        dfs += [df]

    df = pd.concat(dfs).reset_index(drop=True)

    df["x_min"] = df["x_min"].astype(int)
    df["y_min"] = df["y_min"].astype(int)
    df["x_max"] = df["x_max"].astype(int)
    df["y_max"] = df["y_max"].astype(int)

    df["width"] = df["x_max"] - df["x_min"]
    df["height"] = df["y_max"] - df["y_min"]

    return df


def main():
    args = parse_args()

    with open(args.input_file) as f:
        input_file = json.load(f)

    df = json2df(input_file)

    name2id = {}

    coco_categories = []
    for i, label in enumerate(sorted(df["name"].unique())):
        name2id[label] = i + 1
        coco_categories.append({"id": i + 1, "name": label})

    if df.shape[1] == 8:  # ground truth coco_annotations
        coco_images = []
        coco_annotations = []

        for image_name, dft in tqdm(df.groupby("image_name")):
            image_id = Path(image_name).stem

            image_info = {"id": image_id, "file_name": image_name}

            for i in dft.index:
                x_min = int(dft.loc[i, "x_min"])
                y_min = int(dft.loc[i, "y_min"])
                width = int(dft.loc[i, "width"])
                height = int(dft.loc[i, "height"])

                class_name = dft.loc[i, "name"]

                annotation_id = str(hash(image_id + f"_{i}"))

                annotation_info = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": name2id[class_name],
                    "bbox": [x_min, y_min, width, height],
                    "iscrowd": 0,
                    "area": width * height,
                }

                coco_annotations.append(annotation_info)

            coco_images.append(image_info)

        output_coco_annotations = {
            "categories": coco_categories,
            "images": coco_images,
            "annotations": coco_annotations,
        }

    elif df.shape[1] == 9:  # predictions
        output_coco_annotations = []
        for i in tqdm(df.index):
            x_min = int(df.loc[i, "x_min"])
            y_min = int(df.loc[i, "y_min"])
            width = int(df.loc[i, "width"])
            height = int(df.loc[i, "height"])
            class_name = df.loc[i, "name"]
            score = df.loc[i, "score"]
            image_id = Path(df.loc[i, "image_name"]).stem

            annotation_id = str(hash(f"{image_id}_{i}"))

            annotation_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": name2id[class_name],
                "bbox": [x_min, y_min, width, height],
                "score": score,
            }

            output_coco_annotations += [annotation_info]

    else:
        raise NotImplementedError()

    output_folder = args.output_file_name.parent

    output_folder.mkdir(exist_ok=True, parents=True)

    with open(args.output_file, "w") as f:
        json.dump(output_coco_annotations, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
