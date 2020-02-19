from pathlib import Path
import random
import mmcv
import cv2
import numpy as np


def load_frame(video_path: (Path, str), frame_id: int = -1, rgb: bool = True) -> np.array:
    """Read frames from an mp4 video.

    Args:
        video_path: Path to the video to read
        frame_id: frame_id to read. If negative, return random frame.
        rgb: if the frame should be converted from bgr to rgb format.

    Returns: frame at a given frame id.

    """
    if not Path(video_path).suffix == ".mp4":
        raise ValueError(f"Work only with the mp4 video files, but got {video_path}")

    video = mmcv.VideoReader(str(video_path))

    num_frames = len(video)

    if frame_id < 0:
        frame_id = random.randint(0, num_frames - 1)

    frame = video.get_frame(frame_id)

    if rgb:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame
