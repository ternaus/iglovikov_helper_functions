import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from iglovikov_helper_functions.data_processing.cityscapes.parse_cityscapes import (
    labels,
)
from iglovikov_helper_functions.metrics.iou import calculate_ious_global


def get_mapping_dict():
    """
    Returns: Dictionary with
        keys: mask values
        values: class_name
    """
    labels_df = pd.DataFrame(labels)
    labels_df = labels_df[~labels_df["ignoreInEval"]]
    result = dict(zip(labels_df["trainId"].values, labels_df["name"]))
    return result


def get_args():
    parser = argparse.ArgumentParser("Calculate IOU for Cityscapes dataset.")
    arg = parser.add_argument
    arg("-p", "--predictions_path", type=Path, help="Json with predictions.", required=True)
    arg("-g", "--ground_truth_path", type=Path, help="Json with ground truth.", required=True)
    arg("-n", "--num_classes", type=int, help="Number of classes to use.", required=True)
    arg("-j", "--num_workers", type=int, help="Number of workers to use.", default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    ious = calculate_ious_global(args.ground_truth_path, args.predictions_path, args.num_classes, args.num_workers)

    mapping_dict = get_mapping_dict()

    print(mapping_dict)

    print(f"Mean IOU = {np.mean(ious)}")
    print()
    for class_id, iou in enumerate(ious):
        print(f"{mapping_dict[class_id]}: IOU = {iou}")
