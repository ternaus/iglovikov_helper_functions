"""
This script extracts images and bounding box annotations.

python iglovikov_helper_functions/data_processing/waymo/to_coco.py
        -i <path to tfrecords> \
        -o <output_path> \
        -j <path to save json with labels>
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2

tf.enable_eager_execution()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Extract labels and images for 2D detection from Waqymo tfrecords.")

    parser.add_argument("-i", "--input_path", type=Path, help="Path to the folder with tfrecords.", required=True)
    parser.add_argument("-j", "--output_json_path", type=Path, help="Path save the the json with labels.")
    parser.add_argument("-o", "--output_image_path", type=Path, help="Path to save images.", default=".")
    parser.add_argument("--save_images", action="store_true", help="If we want to save images.")
    parser.add_argument("--num_workers", type=int, help="The number of workers to use.", default=16)
    return parser.parse_args()


def get_box(box: label_pb2.Label, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    center_x = box.center_x
    center_y = box.center_y

    box_width = box.length
    box_height = box.width

    x_min = center_x - 0.5 * box_width
    x_max = center_x + 0.5 * box_width

    y_min = center_y - 0.5 * box_height
    y_max = center_y + 0.5 * box_height

    x_min = np.clip(x_min, 0, image_width)
    x_max = np.clip(x_max, x_min + 1, image_width)

    y_min = np.clip(y_min, 0, image_height)
    y_max = np.clip(y_max, y_min + 1, image_height)

    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)


def get_coco_categories() -> List[Dict[str, Any]]:
    ind = 0
    result: List[Dict[str, Any]] = []
    while True:
        try:
            class_name = str(label_pb2.Label.Type.Name(ind))
            result += [{"id": ind, "name": class_name}]
            ind += 1
        except ValueError:
            return result


def main():
    args = parse_args()

    output_path = args.output_image_path

    coco_images = []
    coco_annotations = []

    for tf_record_path in tqdm(sorted(args.input_path.glob("*.tfrecord"))):
        dataset = tf.data.TFRecordDataset(
            str(tf_record_path), compression_type="", num_parallel_reads=args.num_workers
        )

        for frame_id, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            tfrecord_id = frame.context.name

            for camera_id, image in enumerate(frame.images):
                camera_type = open_dataset.CameraName.Name.Name(image.name)

                rgb_image = tf.image.decode_jpeg(image.image).numpy()
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                image_folder = output_path / tfrecord_id / str(frame_id)
                image_folder.mkdir(exist_ok=True, parents=True)
                image_path = image_folder / f"{camera_type}.jpg"
                image_id = str(Path(image_path.parent.parent.name) / Path(image_path.parent.name) / image_path.stem)

                image_height, image_width = rgb_image.shape[:2]

                image_info = {
                    "id": image_id,
                    "file_name": image_id + ".jpg",
                    "height": image_height,
                    "width": image_width,
                }

                if len(frame.camera_labels) != 0:
                    labels = frame.camera_labels[camera_id].labels

                    if frame.camera_labels[camera_id].name != image.name:
                        raise ValueError(
                            f"Labels do not correspond to the provided image. "
                            f"Image: {open_dataset.CameraName.Name.Name(image.name)} "
                            f"Camera: {open_dataset.CameraName.Name.Name(frame.camera_labels[camera_id].name)}"
                        )

                    for label in labels:
                        annotation_id = str(hash(f"{image_id}_{label.id}"))

                        box = get_box(label.box, image_width, image_height)

                        annotation_info = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(label.type),
                            "bbox": box,
                            "iscrowd": 0,
                            "area": box[2] * box[3],
                            "segmentation": [],
                        }
                        coco_annotations.append(annotation_info)

                coco_images.append(image_info)

                if args.output_image_path is not None:
                    cv2.imwrite(str(image_path), bgr_image)

    if args.save_images:
        output_coco_annotations = {
            "categories": get_coco_categories(),
            "images": coco_images,
            "annotations": coco_annotations,
        }

        with open(args.output_json_path, "w") as f:
            json.dump(output_coco_annotations, f, indent=2)


if __name__ == "__main__":
    main()
