"""
The script reads the data from openimages Challenge 2019

(https://www.kaggle.com/c/open-images-2019-instance-segmentation)

CSV files + images + masks and outputs pkl file with instance
segmentation labels.


Requires:

<X> = train or validation


challenge-2019-<X>-segmentation-masks.csv
challenge-2019-classes-description-segmentable.csv

path to masks
path to images
"""

import argparse
import pickle

import pandas as pd
from tqdm import tqdm

from pycocotools import mask as mutils
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from PIL import Image
import cv2
from src.utils.mask_tools import kaggle2coco, kaggle_rle_encode


def group2mmdetection(group, mask_path: Path, sizes: dict, categories: dict) -> dict:
    image_id, dft = group

    image_width, image_height = sizes[image_id]

    rles = []

    for i in dft.index:
        mask_file_name = dft.loc[i, 'MaskPath']

        png = (cv2.imread(str(mask_path / mask_file_name), 0) > 0).astype(np.uint8)
        png = cv2.resize(png, (image_width, image_height), cv2.INTER_NEAREST)

        rle = kaggle_rle_encode(png)

        rles += [kaggle2coco(rle, image_height, image_width)]

    rles = mutils.frPyObjects(rles, image_height, image_width)
    masks = mutils.decode(rles)
    bboxes = mutils.toBbox(mutils.encode(np.asfortranarray(masks.astype(np.uint8))))
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]

    return {
        'filename': image_id + '.jpg',
        'width': image_width,
        'height': image_height,
        'ann':
            {
                'bboxes': np.array(bboxes, dtype=np.float32),
                'original_labels': dft['LabelName'].values,
                'labels': dft['LabelName'].map(categories).values.astype(np.int) + 1,
                'masks': rles
            }
    }


def get_name2size(image_path: Path, num_jobs: int) -> dict:
    """Return image to size mapping.

    Args:
        image_path: Path where images are stored.
        num_jobs: number of CPU threads to use.

    Returns: {<file_name>}: (width, height)

    """
    def helper(x):
        image = Image.open(x)
        return x.stem, image.size

    sizes = Parallel(n_jobs=num_jobs)(
        delayed(helper)(file_name) for file_name in tqdm(sorted(image_path.glob('*.jpg'))))

    return dict(sizes)


def get_categories(categories_path: Path) -> dict:
    """Create mapping from class name to category_id. Categories start with 1.

    Args:
        categories_path: Path to the file challenge-2019-classes-description-segmentable.csv

    Returns: {class_name: category_id}

    """
    classes = pd.read_csv(str(categories_path), header=None)

    return dict(zip(classes[1].values, classes.index + 1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annotation', type=str, help='Path to the annotation file.')
    parser.add_argument('-i', '--image_path', type=Path, help='Path to images.')
    parser.add_argument('-m', '--mask_path', type=Path, help='Path to masks.')
    parser.add_argument('-c', '--classes', type=Path, help='Path to file with class mapping.')

    parser.add_argument('-o', '--output', type=str, help='Path where to store pickled data.', required=True)
    parser.add_argument('-j', '--num_jobs', type=int, default=1, help='Number of jobs to spawn.')
    return parser.parse_args()


def main():
    args = parse_args()
    annotation = pd.read_csv(args.annotation)

    sizes = get_name2size(args.image_path, args.num_jobs)

    categories = get_categories(args.classes)

    annotation['size'] = annotation['ImageID'].map(sizes)

    print(f'Masks before purge = {annotation.shape[0]}')

    annotation = annotation[annotation['size'].notnull()]

    print(f'Masks after purge = {annotation.shape[0]}')

    groups = annotation.groupby('ImageID')

    samples = Parallel(n_jobs=args.num_jobs)(
        delayed(group2mmdetection)(group, args.mask_path, sizes, categories) for group in tqdm(groups))

    with open(args.output, 'wb') as f:
        pickle.dump(samples, f)


if __name__ == '__main__':
    main()
