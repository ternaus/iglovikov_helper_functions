"""
Script creates additional train_0.csv which is like train.csv, but has fold ids.
"""
import pandas as pd
from iglovikov_helper_functions.utils.generate_splits import stratified_kfold_sampling
import numpy as np
from pathlib import Path
import argparse


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Map gleason data to standard format.")

    parser.add_argument("-d", "--data_path", type=Path, help="Path to folder with the data.", required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    NUM_SPLITS = 5
    RANDOM_STATE = 2016

    train_df = pd.read_csv(args.data_path / "train.csv")
    train_df["exists"] = train_df["EncodedPixels"].notnull().astype(int).apply(lambda x: [x], 1)

    train_df["image_name"] = train_df["ImageId_ClassId"].str.split("_").str.get(0)

    train_df["class_id"] = train_df["ImageId_ClassId"].str.split("_").str.get(-1)

    dft = train_df.groupby("image_name")["exists"].sum()

    image_names = dft.index.values
    labels = np.stack(dft.values)

    is_empty = (1 - np.vstack(dft.values).sum(axis=1)).astype(bool).astype(int)

    image_name2is_empty = dict(zip(image_names, is_empty))

    train_folds, val_folds = stratified_kfold_sampling(Y=labels, n_splits=NUM_SPLITS, random_state=RANDOM_STATE)

    image_name2fold_id = {}

    for fold_id, val_index in enumerate(val_folds):
        for file_name in image_names[val_index]:
            image_name2fold_id[file_name] = fold_id

    train_df["fold_id"] = train_df["image_name"].map(image_name2fold_id)

    train_df["is_empty"] = train_df["image_name"].map(image_name2is_empty)

    del train_df["exists"]

    train_df.to_csv(args.data_path / "train_0.csv", index=False)
