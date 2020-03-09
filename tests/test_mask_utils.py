import cv2
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays as h_arrays
from hypothesis.strategies import integers as h_int

from iglovikov_helper_functions.utils.mask_utils import (
    binary_mask2coco,
    coco_rle_decode,
    coco_rle_encode,
    kaggle2coco,
    kaggle_rle_decode,
    kaggle_rle_encode,
    mask2rle,
    rle2mask,
    one_hot,
    reverse_one_hot,
    remove_small_connected_binary,
)


def test_rle():
    rle = "29102 12 29346 24 29602 24 29858 24 30114 24 30370 24 30626 24 30882 24 31139 23 31395 23 31651 23 31907 23 32163 23 32419 23 32675 23 77918 27 78174 55 78429 60 78685 64 78941 68 79197 72 79452 77 79708 81 79964 85 80220 89 80475 94 80731 98 80987 102 81242 105 81498 105 81754 104 82010 104 82265 105 82521 31 82556 69 82779 27 82818 63 83038 22 83080 57 83297 17 83342 50 83555 13 83604 44 83814 8 83866 37 84073 3 84128 31 84390 25 84652 18 84918 8 85239 10 85476 29 85714 47 85960 57 86216 57 86471 58 86727 58 86983 58 87238 59 87494 59 87750 59 88005 60 88261 60 88517 60 88772 61 89028 53 89283 40 89539 32 89667 10 89795 30 89923 28 90050 29 90179 37 90306 27 90434 38 90562 14 90690 38 90817 9 90946 38 91073 3 91202 38 91458 38 91714 38 91969 39 92225 39 92481 39 92737 39 92993 39 93248 40 93504 40 93760 40 94026 30 94302 10 189792 7 190034 21 190283 28 190539 28 190795 28 191051 28 191307 28 191563 28 191819 28 192075 28 192331 28 192587 28 192843 23 193099 14 193355 5"  # noqa: E501

    width, height = 1600, 256

    mask = rle2mask(rle, (width, height))

    assert mask.shape == (height, width)

    inversed_rle = mask2rle(mask)

    assert rle == inversed_rle


@given(
    mask=h_arrays(
        dtype=np.uint8,
        shape=(np.random.randint(1, 10), np.random.randint(1, 11)),
        elements=h_int(min_value=0, max_value=255),
    )
)
def test_mask2one_hot(mask):
    num_classes = mask.max() + 1

    one_hot_mask = one_hot(mask, num_classes)

    assert np.all(mask == reverse_one_hot(one_hot_mask))


@given(
    mask=h_arrays(
        dtype=np.uint8,
        shape=(np.random.randint(1, 11), np.random.randint(1, 10)),
        elements=h_int(min_value=0, max_value=255),
    )
)
def test_mask2one_hot_with_limit(mask):
    num_classes = int(mask.max() / 2)

    one_hot_mask = one_hot(mask, num_classes)

    assert one_hot_mask.shape[-1] == num_classes


@given(
    mask=h_arrays(dtype=np.uint8, shape=(np.random.randint(1, 300), np.random.randint(1, 300)), elements=h_int(0, 1))
)
def test_kaggle_rle(mask):
    height, width = mask.shape

    kaggle_rle = kaggle_rle_encode(mask)
    coco_rle = coco_rle_encode(mask)
    assert coco_rle == kaggle2coco(kaggle_rle, height, width)
    assert np.all(mask == kaggle_rle_decode(kaggle_rle, height, width))
    assert np.all(mask == coco_rle_decode(coco_rle, height, width))


@given(
    mask=h_arrays(dtype=np.uint8, shape=(np.random.randint(1, 300), np.random.randint(1, 300)), elements=h_int(0, 1)),
    min_area=h_int(0, 100),
)
def test_remove_small_connected(mask, min_area):

    filtered_mask = remove_small_connected_binary(mask, min_area)

    assert filtered_mask.shape == mask.shape
    assert filtered_mask.dtype == mask.dtype
    assert set(np.unique(filtered_mask)) - {0, 1} == set()

    connectivity = 8
    # Perform the operation
    output = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity, cv2.CV_32S)

    areas = output[2][1:, 4]

    assert (mask - filtered_mask).sum() == areas[areas < min_area].sum()

    filtered_mask_2 = remove_small_connected_binary(filtered_mask, min_area)

    assert np.array_equal(filtered_mask, filtered_mask_2)


def test_coco2binary_mask():
    height, width = 130, 120
    mask = np.zeros((height, width))
    mask[50:70, 30:90] = 1

    result = [
        [
            89.0,
            69.5,
            88.0,
            69.5,
            87.0,
            69.5,
            86.0,
            69.5,
            85.0,
            69.5,
            84.0,
            69.5,
            83.0,
            69.5,
            82.0,
            69.5,
            81.0,
            69.5,
            80.0,
            69.5,
            79.0,
            69.5,
            78.0,
            69.5,
            77.0,
            69.5,
            76.0,
            69.5,
            75.0,
            69.5,
            74.0,
            69.5,
            73.0,
            69.5,
            72.0,
            69.5,
            71.0,
            69.5,
            70.0,
            69.5,
            69.0,
            69.5,
            68.0,
            69.5,
            67.0,
            69.5,
            66.0,
            69.5,
            65.0,
            69.5,
            64.0,
            69.5,
            63.0,
            69.5,
            62.0,
            69.5,
            61.0,
            69.5,
            60.0,
            69.5,
            59.0,
            69.5,
            58.0,
            69.5,
            57.0,
            69.5,
            56.0,
            69.5,
            55.0,
            69.5,
            54.0,
            69.5,
            53.0,
            69.5,
            52.0,
            69.5,
            51.0,
            69.5,
            50.0,
            69.5,
            49.0,
            69.5,
            48.0,
            69.5,
            47.0,
            69.5,
            46.0,
            69.5,
            45.0,
            69.5,
            44.0,
            69.5,
            43.0,
            69.5,
            42.0,
            69.5,
            41.0,
            69.5,
            40.0,
            69.5,
            39.0,
            69.5,
            38.0,
            69.5,
            37.0,
            69.5,
            36.0,
            69.5,
            35.0,
            69.5,
            34.0,
            69.5,
            33.0,
            69.5,
            32.0,
            69.5,
            31.0,
            69.5,
            30.0,
            69.5,
            29.5,
            69.0,
            29.5,
            68.0,
            29.5,
            67.0,
            29.5,
            66.0,
            29.5,
            65.0,
            29.5,
            64.0,
            29.5,
            63.0,
            29.5,
            62.0,
            29.5,
            61.0,
            29.5,
            60.0,
            29.5,
            59.0,
            29.5,
            58.0,
            29.5,
            57.0,
            29.5,
            56.0,
            29.5,
            55.0,
            29.5,
            54.0,
            29.5,
            53.0,
            29.5,
            52.0,
            29.5,
            51.0,
            29.5,
            50.0,
            30.0,
            49.5,
            31.0,
            49.5,
            32.0,
            49.5,
            33.0,
            49.5,
            34.0,
            49.5,
            35.0,
            49.5,
            36.0,
            49.5,
            37.0,
            49.5,
            38.0,
            49.5,
            39.0,
            49.5,
            40.0,
            49.5,
            41.0,
            49.5,
            42.0,
            49.5,
            43.0,
            49.5,
            44.0,
            49.5,
            45.0,
            49.5,
            46.0,
            49.5,
            47.0,
            49.5,
            48.0,
            49.5,
            49.0,
            49.5,
            50.0,
            49.5,
            51.0,
            49.5,
            52.0,
            49.5,
            53.0,
            49.5,
            54.0,
            49.5,
            55.0,
            49.5,
            56.0,
            49.5,
            57.0,
            49.5,
            58.0,
            49.5,
            59.0,
            49.5,
            60.0,
            49.5,
            61.0,
            49.5,
            62.0,
            49.5,
            63.0,
            49.5,
            64.0,
            49.5,
            65.0,
            49.5,
            66.0,
            49.5,
            67.0,
            49.5,
            68.0,
            49.5,
            69.0,
            49.5,
            70.0,
            49.5,
            71.0,
            49.5,
            72.0,
            49.5,
            73.0,
            49.5,
            74.0,
            49.5,
            75.0,
            49.5,
            76.0,
            49.5,
            77.0,
            49.5,
            78.0,
            49.5,
            79.0,
            49.5,
            80.0,
            49.5,
            81.0,
            49.5,
            82.0,
            49.5,
            83.0,
            49.5,
            84.0,
            49.5,
            85.0,
            49.5,
            86.0,
            49.5,
            87.0,
            49.5,
            88.0,
            49.5,
            89.0,
            49.5,
            89.5,
            50.0,
            89.5,
            51.0,
            89.5,
            52.0,
            89.5,
            53.0,
            89.5,
            54.0,
            89.5,
            55.0,
            89.5,
            56.0,
            89.5,
            57.0,
            89.5,
            58.0,
            89.5,
            59.0,
            89.5,
            60.0,
            89.5,
            61.0,
            89.5,
            62.0,
            89.5,
            63.0,
            89.5,
            64.0,
            89.5,
            65.0,
            89.5,
            66.0,
            89.5,
            67.0,
            89.5,
            68.0,
            89.5,
            69.0,
            89.0,
            69.5,
        ]
    ]

    assert binary_mask2coco(mask) == result
