Cityscapes dataset comes as two zip files:

* masks in `gtFine_trainvaltest.zip`
* images in `leftImg8bit_trainvaltest.zip`


After unpacking image files will be stored in:

```bash
leftImg8bit
    train
        <cityname>
            <img_id>_leftImg8bit.png
    val
        <cityname>
            <img_id>_leftImg8bit.png
    test
        <cityname>
            <img_id>_leftImg8bit.png
```

Labels are stored in the file gtFine_trainvaltest.zip

After unpacking, image files will be stored in:

```bash
gtFine
    train
        <cityname>
            <img_id>_gtFine_color.png
            <img_id>_instanceIds.png
            <img_id>_labelIds.png
            <img_id>_polygons.json
    val
        <cityname>
            <img_id>_gtFine_color.png
            <img_id>_instanceIds.png
            <img_id>_labelIds.png
            <img_id>_polygons.json
    test
        <cityname>
            <img_id>_gtFine_color.png
            <img_id>_instanceIds.png
            <img_id>_labelIds.png
            <img_id>_polygons.json
```


To be able to train the model we would like to have:
```bash
train
    images
        <img_id>.jpg
    masks
        <img_id>.png
val
    images
        <img_id>.jpg
    masks
        <img_id>.png
test
    images
        <img_id>.jpg
```

In addition, 33 categories are labeled at the image, but only 19 are typically used for evaluation. Hence we need to
convert masks to the format where [0, 19] values correspond to these main classes and 255 otherwise.


Official scripts to work with the CityScapes data https://github.com/mcordts/cityscapesScripts
