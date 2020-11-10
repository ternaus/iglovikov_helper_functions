import argparse
from pathlib import Path

import cv2
from imagecorruptions import corrupt, get_corruption_names
from joblib import Parallel, delayed
from tqdm import tqdm

from iglovikov_helper_functions.utils.image_utils import bgr2rgb, load_rgb


def parse_args():
    parser = argparse.ArgumentParser("Corrupting images in the original Dataset.")
    parser.add_argument("-i", "--input_image_path", type=Path, help="Path to input images.")
    parser.add_argument("-o", "--output_image_path", type=Path, help="Path to output images.")
    parser.add_argument("-j", "--num_jobs", type=int, default=12, help="Number of jobs to spawn.")
    parser.add_argument(
        "-s", "--max_severity", type=int, default=5, help="Max severity. Images will be corrupted in [1, s]"
    )
    return parser.parse_args()


def process_image(file_path: Path, max_severity: int, output_dir: Path) -> None:
    image = load_rgb(file_path)
    for corruption in get_corruption_names():
        for severity in range(max_severity):
            corrupted = corrupt(image, corruption_name=corruption, severity=severity + 1)
            corrupted = bgr2rgb(corrupted)
            cv2.imwrite(
                str(output_dir.joinpath(f"{file_path.stem}_{corruption}_{severity + 1}{file_path.suffix}")), corrupted
            )


def main():
    args = parse_args()

    output_dir = args.output_image_path
    output_dir.mkdir(exist_ok=True)

    # for file_path in tqdm(sorted(args.input_image_path.glob("*.*"))):
    Parallel(n_jobs=args.num_jobs)(
        delayed(process_image)(file_path, args.max_severity, output_dir)
        for file_path in tqdm(sorted(args.input_image_path.glob("*.*")))
    )
    #
    #
    # process_image(file_path, args.max_severity, output_dir)


if __name__ == "__main__":
    main()
