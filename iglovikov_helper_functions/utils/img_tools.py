import cv2
import numpy as np
import jpeg4py


def load_rgb(image_path, lib="cv2") -> np.array:
    """Load RGB image from path.

    Args:
        image_path: path to image

    Returns: 3 channel array with RGB image

    """
    if image_path.is_file():
        if lib == "cv2":
            image = cv2.imread(str(image_path))
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif lib == "jpeg4py":
            return jpeg4py.JPEG(image_path).decode()

    raise FileNotFoundError(f"File not found {image_path}")


def load_grayscale(mask_path) -> np.array:
    """Load grayscale mask from path

    Args:
        mask_path: Path to mask

    Returns: 1 channel grayscale mask

    """
    if mask_path.is_file():
        return cv2.imread(str(mask_path), 0)
    raise FileNotFoundError(f"File not found {mask_path}")
