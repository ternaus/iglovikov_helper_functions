"""
This script annotation_ids to

annotation_id = str(hash(f"{image_id}_{label.id}"))

and adds width, height to images
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

map_category_id = {1: 1, 2: 2, 4: 3}

valid_classes = {"TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"}


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Extract labels and images for 2D detection from Waqymo tfrecords.")

    parser.add_argument("-i", "--input_json_path", type=Path, help="Path to the input json.", required=True)
    parser.add_argument("-o", "--output_json_path", type=Path, help="Path to the output json.", default=".")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input_json_path) as f:
        labels = json.load(f)

    new_categories = [
        {"id": i + 1, "name": x}
        for i, x in enumerate(x["name"] for x in labels["categories"] if x["name"] in valid_classes)
    ]

    df_annotations = pd.DataFrame.from_dict(labels["annotations"])

    result = []

    for image_id, dft in tqdm(df_annotations.groupby("image_id")):
        ids = [str(hash(f"{image_id}_{x}")) for x in range(dft.shape[0])]
        dft["id"] = ids

        result += [dft]

    df_annotations = pd.concat(result).reset_index(drop=True)

    df_annotations["category_id"] = df_annotations["category_id"].map(map_category_id)

    result = {
        "categories": new_categories,
        "annotations": df_annotations.to_dict(orient="records"),
        "images": labels["images"],
    }

    with open(args.output_json_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
