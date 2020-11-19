import math
import random
from typing import Tuple

import cv2
import numpy as np


def np_free_form_mask(
    max_vertex: int, max_length: int, max_brush_width: int, max_angle: int, height: int, width: int
) -> np.ndarray:
    mask = np.zeros((height, width), np.float32)

    num_vertex = random.randint(0, max_vertex)
    start_y = random.randint(0, height - 1)
    start_x = random.randint(0, width - 1)

    brush_width = 0
    for i in range(num_vertex):
        angle = random.random() * max_angle
        angle = math.radians(angle)

        if i % 2 == 0:
            angle = 2 * math.pi - angle

        length = random.randint(0, max_length)
        brush_width = random.randint(10, max_brush_width) // 2 * 2

        next_y = start_y + length * np.cos(angle)
        next_x = start_x + length * np.sin(angle)

        next_y = np.maximum(np.minimum(next_y, height - 1), 0).astype(np.int)
        next_x = np.maximum(np.minimum(next_x, width - 1), 0).astype(np.int)

        cv2.line(mask, (start_y, start_x), (next_y, next_x), 1, brush_width)
        cv2.circle(mask, (start_y, start_x), brush_width // 2, 2)
        start_y, start_x = next_y, next_x

    cv2.circle(mask, (start_y, start_x), brush_width // 2, 2)

    return mask


def generate_stroke_mask(
    image_size: Tuple[int, int],
    parts: int = 7,
    max_vertex: int = 25,
    max_length: int = 80,
    max_brush_width: int = 80,
    max_angle: int = 360,
) -> np.ndarray:
    mask = np.zeros(image_size, dtype=np.float32)
    for _ in range(parts):
        mask = mask + np_free_form_mask(
            max_vertex, max_length, max_brush_width, max_angle, image_size[0], image_size[1]
        )

    return np.minimum(mask, 1.0)
