Script in this folder generates corrupted version of the Dataset, following the procedure from the work [Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming](https://arxiv.org/abs/1907.07484).

In that work, authors apply 15 different augmentations to the original dataset with the severity of the transform varying in strength from 1 to 5.

Augmentations:

In the paper  | In [Albumentations](https://github.com/albumentations-team/albumentations)  |
:-----:|:---------:|
gaussian_noise | [GaussNoise()](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.GaussNoise)  |
shot_noise | - |
impulse_noise | - |
defocus_blur | - |
glass_blur | - |
motion_blur | [MotionBlur()](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.MotionBlur) |
zoom_blur   | -  |
snow   | [RandomSnow](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomSnow) |
frost | -  |
fog | [RandomFog()](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomFog)  |
brightness    | [RandomBrightnessContrast(contrast_limit=0)](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomBrightnessContrast)  |
contrast | [RandomBrightnessContrast(brightness_limit=0)](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomBrightnessContrast) |
elastic_transform | [ElasticTransform()](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ElasticTransform) |
pixelate | [Downscale(interpolation=cv2.INTER_NEAREST)](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Downscale) |
jpeg_compression | [ImageCompression(compression_type=0)](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ImageCompression) |


# To run

```bash
python iglovikov_helper_functions/data_processing/corrupt_dataset/transform.py -h                           (anaconda3)  13:51:19
usage: Corrupting images in the original Dataset. [-h] [-i INPUT_IMAGE_PATH]
                                                  [-o OUTPUT_IMAGE_PATH]
                                                  [-j NUM_JOBS]
                                                  [-s MAX_SEVERITY]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_IMAGE_PATH, --input_image_path INPUT_IMAGE_PATH
                        Path to input images.
  -o OUTPUT_IMAGE_PATH, --output_image_path OUTPUT_IMAGE_PATH
                        Path to output images.
  -j NUM_JOBS, --num_jobs NUM_JOBS
                        Number of jobs to spawn.
  -s MAX_SEVERITY, --max_severity MAX_SEVERITY
                        Max severity. Images will be corrupted in [1, s]

```

Example:
```bash
python -m iglovikov_helper_functions.data_processing.corrupt_dataset.transform \
       -i ~/workspace/data/COCO/val2017 \
       -o ~/workspace/data/COCO/val2017_corrupted
```
