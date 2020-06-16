"""Extracts face masks and relevant keypoints from images.

The idea and code borrowed from https://github.com/broutonlab/face-id-with-medical-masks/


You need to install:

pip install -U face_alignment

Output:
    images
        <sha256>.jpg
    metadata
        <sha256>.json

where each json looks like

{
    "points": List[List[float]]],
    "s1": float,
    "s2": float
}

"""
import argparse
import json
from pathlib import Path
from typing import Tuple

import albumentations as albu
import cv2
import face_alignment
import numpy as np
import torch
from tqdm import tqdm

from iglovikov_helper_functions.utils.image_utils import get_sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=Path, help="Path to the images..", required=True)
    parser.add_argument("-o", "--output_path", type=Path, help="Path where to save masks and metainfo.", required=True)
    parser.add_argument("-s", "--smallest_side", type=int, help="Minimal side of the image.", default=256)
    parser.add_argument("-l", "--largest_side", type=int, help="Maximal side of the image.", default=2048)
    return parser.parse_args()


def l2_measure(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(((a - b) ** 2).sum())


def extract_mask_points(points: np.ndarray) -> np.ndarray:
    target_mask_polygon_points = np.zeros((16, 2), dtype=np.int32)

    target_mask_polygon_points[0] = points[28].astype(np.int32)
    target_mask_polygon_points[1:] = points[1:16].astype(np.int32)

    return target_mask_polygon_points


def extract_polygon(image: np.ndarray, points: np.ndarray) -> tuple:
    rect = cv2.boundingRect(points)
    x_min, y_min, width, height = rect
    x_max, y_max = x_min + width, y_min + height

    crop = image[y_min:y_max, x_min:x_max]
    shifted_points = points - np.array([x_min, y_min])

    crop_mask = cv2.fillPoly(np.zeros((height, width), dtype=np.uint8), [shifted_points], (255))

    crop[crop_mask == 0] = 0

    rgba_crop = np.concatenate((crop, np.expand_dims(crop_mask, axis=2)), axis=2)

    return rgba_crop, shifted_points


def extract_target_points_and_characteristic(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    avg_left_eye_point = points[36:42].mean(axis=0)
    avg_right_eye_point = points[42:48].mean(axis=0)
    avg_mouth_point = points[48:68].mean(axis=0)

    left_face_point = points[1]
    right_face_point = points[15]

    d1 = l2_measure(left_face_point, avg_mouth_point)
    d2 = l2_measure(right_face_point, avg_mouth_point)

    x_left_eye, y_left_eye = avg_left_eye_point
    x_right_eye, y_right_eye = avg_right_eye_point
    alpha = np.arctan((y_right_eye - y_left_eye) / (x_right_eye - x_left_eye + 1e-5))

    s1 = alpha * 180 / np.pi
    s2 = d1 / (d2 + 1e-5)

    target_mask_polygon_points = extract_mask_points(points)

    return target_mask_polygon_points, s1, s2


@torch.no_grad()
def main():
    args = parse_args()

    transform_min = albu.Compose(
        [albu.SmallestMaxSize(max_size=args.smallest_side, interpolation=cv2.INTER_CUBIC, p=1)], p=1
    )
    transform_max = albu.Compose(
        [albu.LongestMaxSize(max_size=args.largest_side, interpolation=cv2.INTER_CUBIC, p=1)], p=1
    )

    output_image_path = args.output_path / "images"
    output_image_path.mkdir(exist_ok=True, parents=True)

    output_metadata_path = args.output_path / "metadata"
    output_metadata_path.mkdir(exist_ok=True, parents=True)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    fa_flipped = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)

    for file_name in tqdm(sorted(args.image_path.rglob("*.*"))):
        if not file_name.is_file():
            continue

        image = cv2.imread(str(file_name))

        if min(image.shape[:2]) < args.smallest_side:
            image = transform_min(image=image)["image"]

        if max(image.shape[:2]) < args.largest_side:
            image = transform_max(image=image)["image"]

        landmarks = fa.get_landmarks_from_image(image)
        if landmarks is None:
            continue

        landmarks_flipped = fa_flipped.get_landmarks_from_image(image)
        if landmarks_flipped is None:
            continue

        landmarks = (landmarks[0] + landmarks_flipped[0]) / 2

        target_points, s1, s2 = extract_target_points_and_characteristic(landmarks)
        mask_rgba_crop, target_points = extract_polygon(image, target_points)

        md5 = get_sha256(mask_rgba_crop)

        cv2.imwrite(str(output_image_path / f"{md5}.jpg"), cv2.cvtColor(mask_rgba_crop, cv2.COLOR_RGB2BGR))

        with open(output_metadata_path / f"{md5}.json", "w") as f:
            json.dump({"points": target_points.tolist(), "s1": s1, "s2": s2}, f)


if __name__ == "__main__":
    main()
