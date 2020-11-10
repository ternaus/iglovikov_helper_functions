"""The dataset comes in a form of tsv file with the serialized images."""

import argparse
import base64
import csv
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, help="Path to input tsv file")
    parser.add_argument("-o", "--output_dir", type=Path, help="Path to output file")
    parser.add_argument("-j", "--num_workers", type=int, default=1, help="The number of CPU threads.")
    return parser.parse_args()


def helper(row, args):
    person_id, image_search_rank, face_id, data = row[0], row[1], row[4], base64.b64decode(row[-1])

    save_dir = args.output_dir / person_id
    save_dir.mkdir(exist_ok=True, parents=True)

    savePath = save_dir / f"{image_search_rank}-{face_id}.jpg"

    with open(savePath, "wb") as f:
        f.write(data)


def main():
    args = get_args()

    with open(args.input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")

        Parallel(n_jobs=args.num_workers)(delayed(helper)(row, args) for row in tqdm(reader))


if __name__ == "__main__":
    main()
