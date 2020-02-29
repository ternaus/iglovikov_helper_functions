"""
Reads all png images in one folder and save to other as jpgs, in flatten way.

One can specify the strength of the jpeg compression
"""
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Convert png to jpg.")

    parser.add_argument("-i", "--input_path", type=Path, help="Path to the jpg image files.", required=True)
    parser.add_argument("-o", "--output_path", type=Path, help="Path to the output folder.", required=True)
    parser.add_argument("-j", "--num_threads", type=int, help="The number of CPU threads", default=64)
    parser.add_argument(
        "-c", "--compression_strength", type=int, help="The strength of the jpg compression", default=100
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not 0 <= args.compression_strength <= 100:
        raise ValueError(f"Compession srength should be in [0, 100], but we got {args.compression_strength}.")

    output_path = args.output_path
    output_path.mkdir(exist_ok=True, parents=True)

    image_file_names = sorted(args.input_path.rglob("*.png"))

    for input_file_name in tqdm(image_file_names):
        img = cv2.imread(str(input_file_name))
        output_file_name = output_path / f"{input_file_name.stem}.jpg"
        cv2.imwrite(str(output_file_name), img, [int(cv2.IMWRITE_JPEG_QUALITY), args.compression_strength])


if __name__ == "__main__":
    main()
