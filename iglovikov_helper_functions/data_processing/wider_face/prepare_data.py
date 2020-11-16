"""
Maps annotations (face bounding boxes & five facial landmarks) from txt to json format.

Download annotations at https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0

from the format:

# 0--Parade/0_Parade_marchingband_1_849.jpg
449 330 122 149 488.906 373.643 0.0 542.089 376.442 0.0 515.031 412.83 0.0 485.174 425.893 0.0 538.357 431.491 0.0 0.82
# 0--Parade/0_Parade_Parade_0_904.jpg
361 98 263 339 424.143 251.656 0.0 547.134 232.571 0.0 494.121 325.875 0.0 453.83 368.286 0.0 561.978 342.839 0.0 0.89
# 0--Parade/0_Parade_marchingband_1_799.jpg
78 221 7 8 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 0.2
78 238 14 17 84.188 244.607 1.0 89.527 244.491 1.0 86.973 247.857 1.0 85.116 250.643 1.0 88.482 250.643 1.0 0.36
113 212 11 15 117.0 220.0 0.0 122.0 220.0 0.0 119.0 222.0 0.0 118.0 225.0 0.0 122.0 225.0 0.0 0.3
134 260 15 15 142.0 265.0 0.0 146.0 265.0 0.0 145.0 267.0 0.0 142.0 272.0 0.0 146.0 271.0 0.0 0.24
163 250 14 17 169.357 256.5 1.0 175.25 257.143 1.0 172.357 260.786 1.0 170.214 262.929 1.0 174.071 262.821 1.0 0.41

to

[
    {
        "file_name": <file_name>,
        "annotations": [
            {
                "bbox": [x_min, x_max, y_min, y_max],
                "landmarks": [[l1_x, l1_y], [l2_x, l2_y], ....]
            },
        ]
    }
]

"""
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=Path, help="Path to the input txt file..", required=True)
    parser.add_argument("-o", "--output_path", type=Path, help="Path to output json file.", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    result = []
    temp = {}

    valid_annotation_indices = np.array([0, 1, 3, 4, 6, 7, 9, 10, 12, 13])

    with open(args.input_path) as f:
        for line_id, line in enumerate(f.readlines()):
            if line[0] == "#":
                if line_id != 0:
                    result += [temp]

                temp = {"file_name": line.replace("#", "").strip(), "annotations": []}
            else:
                points = line.strip().split()

                x_min = int(points[0])
                y_min = int(points[1])
                x_max = int(points[2]) + x_min
                y_max = int(points[3]) + y_min

                x_min = max(x_min, 0)
                y_min = max(y_min, 0)

                x_max = max(x_min + 1, x_max)
                y_max = max(y_min + 1, y_max)

                landmarks = np.array([float(x) for x in points[4:]])

                if landmarks.size > 0:
                    landmarks = landmarks[valid_annotation_indices].reshape(-1, 2).tolist()
                else:
                    landmarks = []

                temp["annotations"] += [{"bbox": [x_min, y_min, x_max, y_max], "landmarks": landmarks}]

        result += [temp]

    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
