"""
Process https://www.kaggle.com/laurentmih/aisegmentcom-matting-human-datasets
"""

import argparse
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=Path, help="Path to folder with images")
    parser.add_argument("-o", "--output_folder", type=Path, help="Path to the output folder")
    return parser.parse_args()


def main():
    args = get_args()

    output_image_folder = args.output_folder / "images"
    output_image_folder.mkdir(exist_ok=True, parents=True)

    output_label_folder = args.output_folder / "labels"
    output_label_folder.mkdir(exist_ok=True, parents=True)

    for matting_file_path in tqdm(sorted((args.data_folder / "matting").rglob("*.png"))):
        class_folder = matting_file_path.parent.parent.name
        folder = matting_file_path.parent.name

        image = cv2.imread(str(matting_file_path), cv2.IMREAD_UNCHANGED)
        mask = (image[:, :, 3] > 255 / 2) * 255

        image_output_file = f"{matting_file_path.stem}.jpg"
        mask_output_file = matting_file_path.name

        image_folder_name = folder.replace("matting", "clip")

        original_image_path = args.data_folder / "clip_img" / class_folder / image_folder_name / image_output_file

        if not original_image_path.exists():
            continue

        shutil.copy(str(original_image_path), str(output_image_folder / image_output_file))

        cv2.imwrite(str(output_label_folder / mask_output_file), mask)


if __name__ == "__main__":
    main()
