[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
# An unstructured set of helper functions.

## Installation

`pip install git+https://github.com/ternaus/helper_functions`


## OpenImages 2019 Challenge:

To map data from OpenImages to mmdetection: 

```bash
python -m helper_functions.src.data_processing.open_images.instance2mmdetection \
-a <path to annotation csv file> \
-i <path to train images> \
-m <path to train masks> \
-c <path to csv file with class description> \
-o train.pkl
```