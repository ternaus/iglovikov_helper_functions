"""
This script requires access to the files
    clean_list_128Vec_WT051_P010.txt
    relabel_list_128Vec_T058.txt

from https://github.com/EB-Dodo/C-MS-Celeb


The script creates new folder for files and copies them from the old with respect to the clean and relabel files

for relabeled files, new file name is sha256 hash.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from iglovikov_helper_functions.utils.general_utils import get_sha256


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder", type=Path, help="Path to folder with images")
    parser.add_argument("-r", "--relabel_file_path", type=Path, help="Path to the file_for relabeling")
    parser.add_argument("-c", "--clean_file_path", type=Path, help="Path to the file_for cleaning")
    parser.add_argument(
        "-o", "--output_folder", type=Path, help="Path to folder with the cleaned and relabeled images"
    )
    return parser.parse_args()


def main():
    args = get_args()

    clean = pd.read_csv(args.clean_file_path, header=None, sep=" ").values

    for row in tqdm(clean):
        file_id = Path(row[1]).stem.split("_")[0]

        source_file_path = args.image_folder / row[0] / f"{file_id}.jpg"

        if not source_file_path.exists():
            continue

        output_folder = args.output_folder / Path(row[1]).parent.name
        output_folder.mkdir(exist_ok=True, parents=True)

        target_file_path = output_folder / f"{file_id}.jpg"

        output_folder.mkdir(exist_ok=True, parents=True)

        shutil.copy(str(source_file_path), str(target_file_path))

    relabel = pd.read_csv(args.relabel_file_path, header=None, sep=" ").values

    for row in tqdm(relabel):
        file_id = Path(row[1]).stem.split("_")[0]

        source_file_path = args.image_folder / Path(row[1]).parent.name / f"{file_id}.jpg"

        if not source_file_path.exists():
            continue

        output_folder = args.output_folder / row[0].strip()
        output_folder.mkdir(exist_ok=True, parents=True)

        sha256 = get_sha256(source_file_path)

        target_file_path = output_folder / f"{sha256}.jpg"

        shutil.copy(str(source_file_path), str(target_file_path))


if __name__ == "__main__":
    main()
