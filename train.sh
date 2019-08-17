#!/usr/bin/env bash

python -m src.data_processing.open_images.instance2mmdetection \
-a ~/ssd4tb/data_fast/open_images_v5/challenge-2019-validation-segmentation-masks.csv \
-i ~/ssd4tb/data_fast/open_images_v5/validation \
-m ~/ssd4tb/data_fast/open_images_v5/val_masks \
-c ~/ssd4tb/data_fast/open_images_v5/challenge-2019-classes-description-segmentable.csv \
-o valid.pkl
