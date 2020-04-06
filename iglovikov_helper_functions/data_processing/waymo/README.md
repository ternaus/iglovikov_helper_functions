# Preparation
You need to install the waymo dataset sdk

```bash
pip install upgrade --pip
pip install waymo-open-dataset-tf-2-1-0==1.2.0
```

or from [source](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md)



```bash
python iglovikov_helper_functions/data_processing/waymo/to_coco.py
        -i <path to tfrecords> \
        -o <output_path> \
        -j <path to save json with labels>
```
