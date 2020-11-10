r"""
Script to calculate mean average precision for a fixed Intersection Over Union (IOU)

Expects input data in the COCO format.

Based on https://github.com/ucbdrive/bdd-data/blob/master/bdd_data/evaluate.py
"""
import argparse
import json
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np

from iglovikov_helper_functions.utils.general_utils import group_by_key


def get_envelope(precisions: np.ndarray) -> np.array:
    """Compute the envelope of the precision curve.

    Args:
      precisions:

    Returns: enveloped precision

    """
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    return precisions


def get_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Calculate area under precision/recall curve.

    Args:
      recalls:
      precisions:

    Returns:

    """
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    precisions = get_envelope(precisions)

    # to calculate area under PR curve, look for points where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap


def get_overlaps(gt_boxes: np.ndarray, box: np.ndarray) -> np.ndarray:
    i_xmin = np.maximum(gt_boxes[:, 0], box[0])
    i_ymin = np.maximum(gt_boxes[:, 1], box[1])

    gt_xmax = gt_boxes[:, 0] + gt_boxes[:, 2]
    gt_ymax = gt_boxes[:, 1] + gt_boxes[:, 3]

    box_xmax = box[0] + box[2]
    box_ymax = box[1] + box[3]

    i_xmax = np.minimum(gt_xmax, box_xmax)
    i_ymax = np.minimum(gt_ymax, box_ymax)

    iw = np.maximum(i_xmax - i_xmin, 0.0)
    ih = np.maximum(i_ymax - i_ymin, 0.0)

    intersection = iw * ih

    union = box[2] * box[3] + gt_boxes[:, 2] * gt_boxes[:, 3] - intersection

    overlaps = intersection / (union + 1e-7)

    return overlaps


def recall_precision(
    gt: np.ndarray, predictions: np.ndarray, iou_threshold: float
) -> Tuple[np.array, np.array, np.array]:
    num_gts = len(gt)
    image_gts = group_by_key(gt, "image_id")

    image_gt_boxes = {
        img_id: np.array([[float(z) for z in b["bbox"]] for b in boxes]) for img_id, boxes in image_gts.items()
    }
    image_gt_checked = {img_id: np.zeros(len(boxes)) for img_id, boxes in image_gts.items()}

    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)

    # go down dets and mark TPs and FPs
    num_predictions = len(predictions)
    tp = np.zeros(num_predictions)
    fp = np.zeros(num_predictions)

    for prediction_index, prediction in enumerate(predictions):
        box = prediction["bbox"]

        max_overlap = -np.inf
        jmax = -1

        try:
            gt_boxes = image_gt_boxes[prediction["image_id"]]  # gt_boxes per image
            gt_checked = image_gt_checked[prediction["image_id"]]  # gt flags per image
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            overlaps = get_overlaps(gt_boxes, box)

            max_overlap = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if max_overlap >= iou_threshold:
            if gt_checked[jmax] == 0:
                tp[prediction_index] = 1.0
                gt_checked[jmax] = 1
            else:
                fp[prediction_index] = 1.0
        else:
            fp[prediction_index] = 1.0

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)

    recalls = tp / float(num_gts)

    # avoid divide by zero in case the first detection matches a difficult ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    ap = get_ap(recalls, precisions)

    return recalls, precisions, ap


def get_category2name(categories: List[Dict[str, Any]]) -> OrderedDict:
    """Creates mapping from category_id to category_name

    Args:
        categories: list of the type:
            "categories": [
                {"id": 1,"name": "person"},
                {"id": 2,"name": "bicycle"},
                {"id": 3,"name": "car"},
                {"id": 4,"name": "motorcycle"},
                {"id": 5,"name": "airplane"},
                ...
                {"id": 89,"name": "hair drier"},
                {"id": 90,"name": "toothbrush"}
            ]
    Returns: {<category_id>: <name>}

    """
    result = OrderedDict()
    for element in categories:
        result[element["id"]] = element["name"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate mAP metric for a given IOU threshold.")
    arg = parser.add_argument
    arg("-p", "--predictions", type=str, help="Json with predictions.", required=True)
    arg("-g", "--ground_truth", type=str, help="Json with ground truth.", required=True)
    arg("-t", "--iou_threshold", type=float, help="Iou threshold", default=0.5)
    args = parser.parse_args()

    with open(args.ground_truth) as f:
        gt = json.load(f)

    with open(args.predictions) as f:
        pred = json.load(f)

    gt_annotations = gt["annotations"]
    categories = gt["categories"]

    gt_by_id = group_by_key(gt_annotations, "category_id")
    pred_by_id = group_by_key(pred, "category_id")

    print(gt_by_id.keys())
    print(pred_by_id.keys())

    category2name = get_category2name(categories)

    num_class_names = len(category2name.values())

    print("Class_names in ground truth = ", sorted(category2name.values()))

    aps = np.zeros(num_class_names)

    class_names: List[str] = []

    for i, category_id in enumerate(category2name.keys()):
        if category_id in pred_by_id:  # if we predicted at leas one object for this class.
            recalls, precisions, average_precision = recall_precision(
                gt_by_id[category_id], pred_by_id[category_id], args.iou_threshold
            )
            aps[i] = average_precision

        class_names += [category2name[category_id]]

    mAP = np.mean(aps)
    print("Average per class mean average precision = ", mAP)

    for j in sorted(zip(class_names, aps.flatten().tolist())):
        print(j)
