#!/usr/bin/env bash

# Validation

python deepfake2coco.py \
  -io ~/workspace/data2/open_images_5/validation \
  -lo ~/workspace/evo970/data/open_images_faces/validation/labels \
  -id ~/workspace/evo970/data/kaggle_deepfake/frames_120 \
  -ld ~/workspace/evo970/data/kaggle_deepfake/real_faces/labels \
  -m ~/workspace/evo970/data/kaggle_deepfake/targets_0.csv \
  --exclude_folds 1 2 3 4 \
  -j 16 \
  --output_path validation.json

# train
