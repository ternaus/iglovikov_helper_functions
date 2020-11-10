"""Calculate Mean Average Precision on the ground truth and predictions in the COCO format."""

import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def print_results(coco_eval):
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-p", "--pred_path", type=str, help="Path to the ground truth json file.", required=True)
    arg("-g", "--gt_path", type=str, help="Path to the json file with predictions.", required=True)

    args = parser.parse_args()

    coco = COCO(args.gt_path)

    pred_coco = coco.loadRes(args.pred_path)

    categories = coco.cats

    print("-------------------------------------------------------------------------------")
    print("CATEGORIES:")
    print(categories)

    print("-------------------------------------------------------------------------------")

    coco_eval = COCOeval(cocoGt=coco, cocoDt=pred_coco, iouType="bbox")

    print("ALL CLASSES :")

    print_results(coco_eval)

    for value in categories.values():
        category_id = value["id"]
        class_name = value["name"]
        print("-------------------------------------------------------------------------------")
        print("CLASS_NAME = ", class_name)

        coco_eval.params.catIds = category_id
        print_results(coco_eval)
