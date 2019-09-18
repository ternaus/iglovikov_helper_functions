"""Set of scripts to work with masks."""
from itertools import groupby

import numpy as np
from pycocotools import mask as mutils
from skimage import measure


def coco_seg2bbox(polygons, image_height: int, image_width: int) -> list:
    """Converts polygons in COCO format to bounding box in pixels.

    Args:
        polygons:
        image_height: Height of the target image.
        image_width: Width of the target image.

    Returns: [x_min, y_min, width, height]

    """
    rles = mutils.frPyObjects(polygons, image_height, image_width)
    mask = mutils.decode(rles)
    bbox = mutils.toBbox(mutils.encode(np.asfortranarray(mask.astype(np.uint8))))

    return bbox[0].astype(int).tolist()


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask2coco(binary_mask: np.array, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode="constant", constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def coco2binary_mask(segmentation: list, height: int, width: int) -> np.array:
    rles = mutils.frPyObjects(segmentation, height, width)
    return mutils.decode(rles)[:, :, 0]


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

    return mark.reshape(width, height).T


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
