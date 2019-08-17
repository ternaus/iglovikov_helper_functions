from src.utils.mask_tools import (rle2mask,
                                  mask2rle,
                                  kaggle_rle_decode,
                                  kaggle2coco,
                                  coco_rle_encode,
                                  coco_rle_decode,
                                  kaggle_rle_encode)

import numpy as np


def test_rle():
    rle = '29102 12 29346 24 29602 24 29858 24 30114 24 30370 24 30626 24 30882 24 31139 23 31395 23 31651 23 31907 23 32163 23 32419 23 32675 23 77918 27 78174 55 78429 60 78685 64 78941 68 79197 72 79452 77 79708 81 79964 85 80220 89 80475 94 80731 98 80987 102 81242 105 81498 105 81754 104 82010 104 82265 105 82521 31 82556 69 82779 27 82818 63 83038 22 83080 57 83297 17 83342 50 83555 13 83604 44 83814 8 83866 37 84073 3 84128 31 84390 25 84652 18 84918 8 85239 10 85476 29 85714 47 85960 57 86216 57 86471 58 86727 58 86983 58 87238 59 87494 59 87750 59 88005 60 88261 60 88517 60 88772 61 89028 53 89283 40 89539 32 89667 10 89795 30 89923 28 90050 29 90179 37 90306 27 90434 38 90562 14 90690 38 90817 9 90946 38 91073 3 91202 38 91458 38 91714 38 91969 39 92225 39 92481 39 92737 39 92993 39 93248 40 93504 40 93760 40 94026 30 94302 10 189792 7 190034 21 190283 28 190539 28 190795 28 191051 28 191307 28 191563 28 191819 28 192075 28 192331 28 192587 28 192843 23 193099 14 193355 5'  # noqa: E501

    width, height = 1600, 256

    mask = rle2mask(rle, (width, height))

    assert mask.shape == (height, width)

    inversed_rle = mask2rle(mask)

    assert rle == inversed_rle


def test_kaggle_rle():
    for _ in range(10):
        h = np.random.randint(1, 1000)
        w = np.random.randint(1, 1000)
        mask = np.random.randint(0, 2, h * w).reshape(h, w)

        kaggle_rle = kaggle_rle_encode(mask)
        coco_rle = coco_rle_encode(mask)
        assert coco_rle == kaggle2coco(kaggle_rle, h, w)
        assert np.all(mask == kaggle_rle_decode(kaggle_rle, h, w))
        assert np.all(mask == coco_rle_decode(coco_rle, h, w))
