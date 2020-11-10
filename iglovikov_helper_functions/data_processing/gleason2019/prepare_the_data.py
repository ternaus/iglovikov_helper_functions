"""
This script parses the data from https://gleason2019.grand-challenge.org into standard format

The data comes in files:

Maps1_T.zip
Maps2_T.zip
Maps3_T.zip
Maps4_T.zip
Maps5_T.zip
Maps6_T.zip
Test.zip
Train Imgs.zip

Maps<N>_T.zip -> masks with the labeling from 6 different expert pathologists.


After unpacking we get folders:

Maps1_T
    slideN_coreM_classimg_nonconvex.png
Maps2_T
    slideN_coreM_classimg_nonconvex.png
Maps3_T
    slideN_coreM_classimg_nonconvex.png
Maps4_T
    slideN_coreM_classimg_nonconvex.png
Maps5_T
    slideN_coreM_classimg_nonconvex.png
Maps6_T
    slideN_core00M_classimg_nonconvex.png
Test
    Test_imgs
        slideM_coreN.jpg
Train Imgs
    slideM_coreN.jpg


We map it to:

train
    images
        slideM_coreN.jpg
    masks
        slideM_coreN.png
test
    images
        slideM_coreN.jpg


where each mask is a mode across different experts for each mask.

Labels are:

Labels 0, 1, and 6:     benign (no cancer)
Label 3:                Gleason grade 3
Label 4:                Gleason grade 4
Label 5:                Gleason grade 5

Label mapping:
Labels 0, 1, 6 => 0
Label 3 => 1
Label 4 => 2
Label 5 => 3


If there is an image without mask or mask without image => removed.
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from tqdm import tqdm


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Map gleason data to standard format.")

    parser.add_argument("-d", "--data_path", type=Path, help="Path to folder with the data.", required=True)
    parser.add_argument("-n", "--n_jobs", type=int, help="Number of jobs to run in parallel.", required=True)
    return parser.parse_args()


def prepare_folders(data_path: Path) -> Tuple[Path, Path]:
    train_path = data_path / "train"

    train_image_path = train_path / "images"
    train_mask_path = train_path / "masks"

    train_image_path.mkdir(exist_ok=True, parents=True)
    train_mask_path.mkdir(exist_ok=True, parents=True)

    test_image_path = data_path / "test" / "images"
    test_image_path.mkdir(exist_ok=True, parents=True)

    return train_image_path, test_image_path


def get_mapping() -> np.array:
    mapping = np.arange(0, 256, dtype=np.uint8)
    mapping[1] = 0
    mapping[6] = 0
    mapping[3] = 1
    mapping[4] = 2
    mapping[5] = 3

    return mapping


def merge_masks(file_path: Path) -> None:
    mask_list: List[np.ndarray] = []
    final_mask_path = file_path.parent.parent.parent / "train" / "masks" / (file_path.stem + ".png")

    for num_expert in range(1, 6 + 1):
        mask_path = file_path.parents[2] / f"Maps{num_expert}_T" / (file_path.stem + "_classimg_nonconvex.png")
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), 0)
            mask = cv2.LUT(mask, get_mapping())
            mask_list += [mask]

    if mask_list:
        mask = stats.mode(np.dstack(mask_list), axis=2).mode[:, :, 0]

        if not 0 <= mask.max() <= 3:
            raise ValueError()

        cv2.imwrite(str(final_mask_path), mask)
    else:
        print("No masks, removing img from train: ", file_path)
        file_path.unlink()


def main():
    args = parse_args()

    train_image_folder, test_image_folder = prepare_folders(args.data_path)

    old_train_image_folder = args.data_path / "Train Imgs"
    old_test_image_folder = args.data_path / "Test" / "Test_imgs"

    for old_file_name in tqdm(sorted(old_train_image_folder.glob("*.jpg"))):
        new_file_name = train_image_folder / old_file_name.name

        shutil.copy(str(old_file_name), str(new_file_name))

    for old_file_name in tqdm(sorted(old_test_image_folder.glob("*.jpg"))):
        new_file_name = test_image_folder / old_file_name.name

        shutil.copy(str(old_file_name), str(new_file_name))

    Parallel(n_jobs=args.n_jobs)(
        delayed(merge_masks)(file_path) for file_path in tqdm(sorted(train_image_folder.glob("*.jpg")))
    )


if __name__ == "__main__":
    main()
