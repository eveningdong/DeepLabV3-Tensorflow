# DeepLabV3 Semantic Segmentation
Reimplementation of DeepLabV3 Semantic Segmentation

This is an (re-)implementation of [DeepLabv3](https://arxiv.org/abs/1706.05587) in TensorFlow for semantic image segmentation on the [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/). The implementation is based on [DrSleep's implementation on data augmentation](https://github.com/DrSleep/tensorflow-deeplab-resnet) and [CharlesShang's implementation on tfrecord](https://github.com/CharlesShang/FastMaskRCNN).
## Requirement
#### Tensorflow 1.2
```
python 3.5
tensorflow 1.2
CUDA  8.0
cuDNN 5.1
```
#### Tensorflow 1.3+ (Updated in new branch)
```
python 3.5
tensorflow 1.3+
CUDA  8.0
cuDNN 6.0
```

#### Installation
```
pip3 install -r requirements.txt
```

## Train
1. Configurate `config.py`.
2. Run `python3 convert_voc12.py`, this will generate a tfrecord file in `$DATA_DIRECTORY/records`.
3. Run `python3 train_voc12.py`

