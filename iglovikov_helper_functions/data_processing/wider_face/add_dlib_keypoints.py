"""
The script adds 6 dlib keypoints to the crops that are 32x32 bigger

Takes data in the form

[
    {
        "file_name": <file_name>,
        "annotations": [
            {
                "x_min": <int>,
                "y_min": <int>,
                "width: <int>,
                "height": <int>,
                "landmarks": [l1, l2, l3, ....]
            },
        ]
    }
]

and adds the field: "dlib_landmarks"


requirement:

pip install face_alignment

"""
import argparse
import json
from pathlib import Path

import albumentations as albu
import cv2
import face_alignment
import numpy as np
import torch
from tqdm import tqdm

from iglovikov_helper_functions.utils.box_utils import resize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=Path, help="Path to the images..", required=True)
    parser.add_argument(
        "-a", "--annotation_path", type=Path, help="Path to the json with annotations..", required=True
    )
    parser.add_argument("-o", "--output_path", type=Path, help="Path to output json file.", required=True)
    parser.add_argument("-m", "--min_side", type=int, help="Minimal side of the face to be considered.", default=32)
    parser.add_argument(
        "-s", "--smallest_side", type=int, help="Minimal side of the face after upscaling.", default=256
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    vis_path = Path("viz")
    vis_path.mkdir(exist_ok=True, parents=True)

    with open(args.annotation_path) as f:
        labels = json.load(f)

    transform = albu.Compose([albu.SmallestMaxSize(max_size=args.smallest_side, interpolation=cv2.INTER_LANCZOS4)])

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, face_detector="folder")
    fa_flipped = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=True, face_detector="folder"
    )

    for label_id, label in tqdm(enumerate(labels), total=len(labels)):
        file_path = args.image_path / label["file_name"]

        image = cv2.imread(str(file_path))

        image_height, image_width = image.shape[:2]

        annotations = label["annotations"]

        for annotation_id, annotation in enumerate(annotations):

            x_min = annotation["x_min"]
            y_min = annotation["y_min"]
            width = annotation["width"]
            height = annotation["height"]

            if width < args.min_side or height < args.min_side:
                continue

            x_max = x_min + width
            y_max = y_min + height

            if not 0 <= x_min < x_max < image_width:
                continue
            if not 0 <= y_min < y_max < image_height:
                continue

            x_min, y_min, x_max, y_max = resize(
                x_min, y_min, x_max, y_max, image_width=image_width, image_height=image_height, resize_coeff=1.2
            )

            crop = image[y_min:y_max, x_min:x_max]

            resized_crop = transform(image=crop)["image"]

            preds = fa.get_landmarks(
                resized_crop, detected_faces=[(0, 0, resized_crop.shape[1], resized_crop.shape[0])]
            )
            preds_a = fa_flipped.get_landmarks(
                resized_crop, detected_faces=[(0, 0, resized_crop.shape[1], resized_crop.shape[0])]
            )

            if preds is None or preds_a is None:
                continue

            preds = (preds[0] + preds_a[0]) / 2

            preds[:, 0] = preds[:, 0] * crop.shape[1] / resized_crop.shape[1] + x_min
            preds[:, 1] = preds[:, 1] * crop.shape[0] / resized_crop.shape[0] + y_min

            preds[:, 0] = np.clip(preds[:, 0], 0, image.shape[1] - 1)
            preds[:, 1] = np.clip(preds[:, 1], 0, image.shape[0] - 1)

            labels[label_id]["annotations"][annotation_id]["dlib_landmarks"] = preds.tolist()

    with open(args.output_path, "w") as f:
        json.dump(labels, f, indent=2)


if __name__ == "__main__":
    main()
