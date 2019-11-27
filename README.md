[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

[![DeepSource](https://static.deepsource.io/deepsource-badge-light.svg)](https://deepsource.io/gh/ternaus/iglovikov_helper_functions/?ref=repository-badge)
# An unstructured set of helper functions.

## Installation

`pip install -U iglovikov_helper_functions`

`pip install git+https://github.com/ternaus/iglovikov_helper_functions`


## OpenImages 2019 Challenge:

To map data from OpenImages to mmdetection:

```bash
python -m iglovikov_helper_functions.src.data_processing.open_images.instance2coco \
-a <path to annotation csv file> \
-i <path to train images> \
-m <path to train masks> \
-c <path to csv file with class description> \
-o train.pkl
```

## Cityscapes

To map the data to the standard format follow [How to map Cityscapes](data_processing/cityscapes/README.md)

## To create corrupted dataset following [Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming](https://arxiv.org/abs/1907.07484)

[Readme](iglovikov_helper_functions/data_parsing/corrupt_dataset/README.md)
