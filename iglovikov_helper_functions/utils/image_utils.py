from hashlib import sha256
from pathlib import Path
from typing import Tuple, Union, Optional, Dict

import cv2
import jpeg4py
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


def load_rgb(image_path: Union[Path, str], lib: str = "cv2") -> np.array:
    """Load RGB image from path.

    Args:
        image_path: path to image
        lib: library used to read an image.
            currently supported `cv2` and `jpeg4py`

    Returns: 3 channel array with RGB image

    """
    if Path(image_path).is_file():
        if lib == "cv2":
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif lib == "jpeg4py":
            image = jpeg4py.JPEG(str(image_path)).decode()
        else:
            raise NotImplementedError("Only cv2 and jpeg4py are supported.")
        return image

    raise FileNotFoundError(f"File not found {image_path}")


def load_grayscale(mask_path: Union[Path, str]) -> np.array:
    """Load grayscale mask from path

    Args:
        mask_path: Path to mask

    Returns: 1 channel grayscale mask

    """
    if Path(mask_path).is_file():
        return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    raise FileNotFoundError(f"File not found {mask_path}")


def pad(image: np.array, factor: int = 32, border: int = cv2.BORDER_REFLECT_101) -> tuple:
    """Pads the image on the sides, so that it will be divisible by factor.
    Common use case: UNet type architectures.

    Args:
        image:
        factor:
        border: cv2 type border.

    Returns: padded_image

    """
    height, width = image.shape[:2]

    if height % factor == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = factor - height % factor
        y_min_pad = y_pad // 2
        y_max_pad = y_pad - y_min_pad

    if width % factor == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = factor - width % factor
        x_min_pad = x_pad // 2
        x_max_pad = x_pad - x_min_pad

    padded_image = cv2.copyMakeBorder(image, y_min_pad, y_max_pad, x_min_pad, x_max_pad, border)

    return padded_image, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


def unpad(image: np.array, pads: Tuple[int, int, int, int]) -> np.ndarray:
    """Crops patch from the center so that sides are equal to pads.

    Args:
        image:
        pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)

    Returns: cropped image

    """
    x_min_pad, y_min_pad, x_max_pad, y_max_pad = pads
    height, width = image.shape[:2]

    return image[y_min_pad : height - y_max_pad, x_min_pad : width - x_max_pad]


def pad_to_size(
    image: np.array,
    target_size: Tuple[int, int],
    bboxes: Optional[np.ndarray] = None,
    keypoints: Optional[np.ndarray] = None,
) -> Dict[str, Union[np.ndarray, Tuple[int, int, int, int]]]:
    """Pads the image on the sides to the target_size

    Args:
        image:
        target_size: (target_height, target_width)
        bboxes: np.array with shape (num_boxes, 4). Each row: [x_min, y_min, x_max, y_max]
        keypoints: np.array with shape (num_keypoints, 2), each row: [x, y]

    Returns:
        {
            "image": padded_image,
            "pads": (x_min_pad, y_min_pad, x_max_pad, y_max_pad),
            "bboxes": shifted_boxes,
            "keypoints": shifted_keypoints
        }

    """
    target_height, target_width = target_size

    image_height, image_width = image.shape[:2]

    if target_width < image_width:
        raise ValueError(f"Target width should bigger than image_width" f"We got {target_width} {image_width}")

    if target_height < image_height:
        raise ValueError(f"Target height should bigger than image_height" f"We got {target_height} {image_height}")

    if image_height == target_height:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = target_height - image_height
        y_min_pad = y_pad // 2
        y_max_pad = y_pad - y_min_pad

    if image_width == target_width:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = target_width - image_width
        x_min_pad = x_pad // 2
        x_max_pad = x_pad - x_min_pad

    result = {
        "pads": (x_min_pad, y_min_pad, x_max_pad, y_max_pad),
        "image": cv2.copyMakeBorder(image, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_CONSTANT),
    }

    if bboxes is not None:
        bboxes[:, 0] += x_min_pad
        bboxes[:, 1] += y_min_pad
        bboxes[:, 2] += x_min_pad
        bboxes[:, 3] += y_min_pad

        result["bboxes"] = bboxes

    if keypoints is not None:
        keypoints[:, 0] += x_min_pad
        keypoints[:, 1] += y_min_pad

        result["keypoints"] = keypoints

    return result


def unpad_from_size(
    image: np.array,
    pads: Tuple[int, int, int, int],
    bboxes: Optional[np.ndarray] = None,
    keypoints: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Crops patch from the center so that sides are equal to pads.

    Args:
        image:
        pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
        bboxes: np.array with shape (num_boxes, 4). Each row: [x_min, y_min, x_max, y_max]
        keypoints: np.array with shape (num_keypoints, 2), each row: [x, y]

    Returns: cropped image

    {
            "image": cropped_image,
            "bboxes": shifted_boxes,
            "keypoints": shifted_keypoints
        }

    """
    x_min_pad, y_min_pad, x_max_pad, y_max_pad = pads
    height, width = image.shape[:2]

    result = {"image": image[y_min_pad : height - y_max_pad, x_min_pad : width - x_max_pad]}

    if bboxes is not None:
        bboxes[:, 0] -= x_min_pad
        bboxes[:, 1] -= y_min_pad
        bboxes[:, 2] -= x_min_pad
        bboxes[:, 3] -= y_min_pad

        result["bboxes"] = bboxes

    if keypoints is not None:
        keypoints[:, 0] -= x_min_pad
        keypoints[:, 1] -= y_min_pad

        result["keypoints"] = keypoints

    return result


def get_size(file_path: Union[str, Path]) -> Tuple[int, int]:
    """Gets size of the image in a lazy way.

    Args:
        file_path: Path to the target image.

    Returns: (width, height)

    """
    image = Image.open(file_path)
    labeled_exif = get_labeled_exif(get_exif(image))
    if labeled_exif["Orientation"] in [6, 8]:
        cv2_height, cv2_width = cv2.imread((str(file_path))).shape[:2]
        height, width = image.size
        if cv2_height != height or cv2_width != width:
            raise ValueError(
                f"PIL and cv2 image shapes do not match. " f"PIL {width, height}. CV2 {cv2_width, cv2_height}."
            )
    else:
        width, height = image.size

    return width, height


def get_exif(image: Image) -> dict:
    image.verify()
    return image._getexif()


def get_labeled_exif(exif: dict) -> dict:
    labeled = {}
    for (key, val) in exif.items():
        labeled[TAGS.get(key)] = val
    return labeled


def bgr2rgb(image: np.array) -> np.array:
    """Convert image from bgr to rgb format

    Args:
        image:

    Returns:

    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def stretch_8bit(
    bands: np.ndarray, lower_percent: int = 1, higher_percent: int = 99, exclude_zero: bool = True
) -> np.ndarray:
    out = np.zeros_like(bands).astype(np.float32)
    for i in range(bands.shape[-1]):
        a = 0
        b = 1
        band = bands[:, :, i].flatten()
        if exclude_zero:
            filtered = band[band > 0]
        if len(filtered) == 0:
            continue
        c = np.percentile(filtered, lower_percent)
        d = np.percentile(filtered, higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out.astype(np.float32)


def get_sha256(image: np.ndarray, file_type: str = ".jpg") -> str:
    _, buffer = cv2.imencode(file_type, image)
    return sha256(buffer).hexdigest()
