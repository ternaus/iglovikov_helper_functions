"""Set of scripts to work with masks."""
from itertools import groupby
from pycocotools import mask as mutils
import numpy as np


def rle2mask(src_string: str, size: tuple) -> np.array:
    """Convert mask from rle to numpy array.

    Args:
        src_string: rle string
        size: (width, height)

    Returns: binary numpy array with mask

    """
    width, height = size

    mark = np.zeros(width * height).astype(np.uint8)

    array = np.asarray([int(x) for x in src_string.split()])
    starts = array[0::2]
    ends = array[1::2]

    current_position = 0
    for index, first in enumerate(starts):
        mark[int(first) : int(first + ends[index])] = 1
        current_position += ends[index]

    return np.flipud(np.rot90(mark.reshape(width, height), k=1))


def mask2rle(mask: np.array) -> str:
    """Convert binary mask to RLE

    Args:
        mask: binary mask

    Returns: binary mask in RLE.

    """
    tmp = np.rot90(np.flipud(mask), k=3)
    rle = []
    lastColor = 0
    startpos = 0

    tmp = tmp.reshape(-1, 1)
    for i in range(len(tmp)):
        if lastColor == 0 and tmp[i] > 0:
            startpos = i
            lastColor = 1
        elif (lastColor == 1) and (tmp[i] == 0):
            endpos = i - 1
            lastColor = 0
            rle.append(str(startpos) + " " + str(endpos - startpos + 1))
    return " ".join(rle)


def kaggle_rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 1
    rle[1::2] -= rle[::2]
    return rle.tolist()


def kaggle_rle_decode(rle, h, w):
    starts, lengths = map(np.asarray, (rle[::2], rle[1::2]))
    starts -= 1
    ends = starts + lengths
    img = np.zeros(h * w, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((w, h)).T


def coco_rle_encode(mask):
    rle = {"counts": [], "size": list(mask.shape)}
    counts = rle.get("counts")
    for i, (value, elements) in enumerate(groupby(mask.ravel(order="F"))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def coco_rle_decode(rle, h, w):
    return mutils.decode(mutils.frPyObjects(rle, h, w))


def kaggle2coco(kaggle_rle, height, width):
    if not len(kaggle_rle):
        return {"counts": [height * width], "size": [height, width]}
    roll2 = np.roll(kaggle_rle, 2)
    roll2[:2] = 1

    roll1 = np.roll(kaggle_rle, 1)
    roll1[:1] = 0

    if height * width != kaggle_rle[-1] + kaggle_rle[-2] - 1:
        shift = 1
        end_value = height * width - kaggle_rle[-1] - kaggle_rle[-2] + 1
    else:
        shift = 0
        end_value = 0
    coco_rle = np.full(len(kaggle_rle) + shift, end_value)
    coco_rle[: len(coco_rle) - shift] = kaggle_rle.copy()
    coco_rle[: len(coco_rle) - shift : 2] = (kaggle_rle - roll1 - roll2)[::2].copy()
    return {"counts": coco_rle.tolist(), "size": [height, width]}
