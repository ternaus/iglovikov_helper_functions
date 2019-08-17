To map data from OpenImages 2019 challenge:



```bash
python -m src.data_processing.open_images.instance2mmdetection \
-a <path to annotation csv file> \
-i <path to train images> \
-m <path to train masks> \
-c <path to csv file with class description> \
-o train.pkl
```