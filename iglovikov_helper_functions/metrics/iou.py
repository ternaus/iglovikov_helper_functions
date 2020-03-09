"""Script to calculate global IOU for segmentation tasks."""
import argparse
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from iglovikov_helper_functions.utils.image_utils import load_grayscale
from iglovikov_helper_functions.utils.metrics import (
    calculate_confusion_matrix_from_arrays,
    calculate_jaccard,
    calculate_tp_fp_fn,
)


def confusion_matrix_from_files(y_true_path: Path, y_pred_path: Path, num_classes: int) -> np.array:
    y_true = load_grayscale(y_true_path)
    y_pred = load_grayscale(y_pred_path)

    if not y_true.shape == y_pred.shape:
        raise ValueError(
            f"y_true and y_pred should have the same shape. "
            f"y_true shape = {y_true.shape} y_pred.shape = {y_pred.shape} "
            f"y_pred_path = {y_pred_path} "
            f"y_true_path = {y_true_path}"
        )

    return calculate_confusion_matrix_from_arrays(y_true, y_pred, num_classes=num_classes)


def calculate_ious_global(y_true_path: Path, y_pred_path: Path, num_classes: int, num_workers: int = 1) -> np.array:
    y_true_files = sorted(y_true_path.glob("*.png"))
    y_pred_files = sorted(y_pred_path.glob("*.png"))

    if not [x.name for x in y_pred_files] == [x.name for x in y_true_files]:
        raise AssertionError("Should have the same file names for y_true and y_pred.")

    def helper(file_name):
        return confusion_matrix_from_files(
            y_true_path.joinpath(file_name), y_pred_path.joinpath(file_name), num_classes
        )

    matrices = Parallel(n_jobs=num_workers)(delayed(helper)(file_path.name) for file_path in tqdm(y_pred_files))

    confusion_matrix = np.dstack(matrices).sum(axis=2)
    tp_fp_fn_dict = calculate_tp_fp_fn(confusion_matrix)

    return calculate_jaccard(tp_fp_fn_dict)


def get_args():
    parser = argparse.ArgumentParser("Calculate IOU for segmentation masks.")
    arg = parser.add_argument
    arg("-p", "--predictions_path", type=Path, help="Json with predictions.", required=True)
    arg("-g", "--ground_truth_path", type=Path, help="Json with ground truth.", required=True)
    arg("-n", "--num_classes", type=int, help="Number of classes to use.", required=True)
    arg("-j", "--num_workers", type=int, help="Number of workers to use.", default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    ious = calculate_ious_global(args.ground_truth_path, args.predictions_path, args.num_classes, args.num_workers)

    print(f"Mean IOU = {np.mean(ious)}")
    for class_id, iou in enumerate(ious):
        print(f"Class {class_id}: IOU = {iou}")
