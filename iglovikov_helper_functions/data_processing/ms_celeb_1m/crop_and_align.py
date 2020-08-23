"""
Crop and align faces based on the information about keypoints.

Label_files:

{'file_name': '100-FaceId-0.jpg',
 'annotations': [{'bbox': [90, 54, 201, 193],
   'score': 0.9990234375,
   'landmarks': [[125, 99], [176, 101], [152, 137], [128, 158], [169, 159]]}],
 'file_path': 'm.0107_f/100-FaceId-0.jpg'}

"""

import argparse
from pathlib import Path
from joblib import Parallel, delayed
import cv2
import json
from iglovikov_helper_functions.utils.image_utils import align_face
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder", type=Path, help="Path to folder with images")
    parser.add_argument("-l", "--label_folder", type=Path, help="Path to the folder with labels")
    parser.add_argument("-o", "--output_folder", type=Path, help="Path to folder with the cropped images")
    parser.add_argument("-j", "--num_workers", type=int, default=1, help="The number of CPU threads.")
    parser.add_argument(
        "-c",
        "--confidence_threshold",
        type=float,
        default=0.7,
        help="If labels have field 'score' Do not use crops with lower threshold.",
    )
    parser.add_argument("--keep_one_face", action="store_true", help="If keep images that have only one face")
    return parser.parse_args()


def crop_and_save(
    output_folder: Path, image_folder: Path, label_path: Path, confidence_threshold: float, keep_one_face: bool
) -> None:
    with open(label_path) as f:
        try:
            label = json.load(f)
        except json.decoder.JSONDecodeError:
            return

    annotations = label["annotations"]

    if keep_one_face and len(annotations) > 1:
        return

    file_name = label["file_path"]
    folder_name = Path(file_name).parent.name

    if (output_folder / folder_name / f"{Path(file_name).name}_0.jpg").exists():
        return

    image = cv2.imread(str(image_folder / file_name))

    (output_folder / folder_name).mkdir(exist_ok=True, parents=True)

    for annotation_id, annotation in enumerate(annotations):
        score = annotation["score"]
        if score < confidence_threshold:
            continue

        landmarks = annotation["landmarks"]

        face_crop = align_face(image, landmarks)

        cv2.imwrite(str(output_folder / folder_name / f"{Path(file_name).name}_{annotation_id}.jpg"), face_crop)

    return


def main():
    args = get_args()

    label_paths = sorted(args.label_path.rglob("*.json"))

    Parallel(n_jobs=args.num_workers, prefer="threads")(
        delayed(crop_and_save)(
            args.output_folder, args.image_folder, label_path, args.confidence_threshold, args.keep_one_face
        )
        for label_path in tqdm(label_paths)
    )


if __name__ == "__main__":
    main()
