# VFRAME: YOLOV3

## Installation

Use provided Darknet docker with NVIDIA Cuda/cudnn. To manually install:

- ``
- Edit `Makefile` then run `make`
- if `make` errors
    - 
```
# Clone AlexyAB's version of Darknet
git clone https://github.com/AlexeyAB/darknet/

# Set paths to CUDA (skip if using docker)
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`
# export PATH=/usr/local/cuda/bin:$PATH

# Edit Makfile
GPU=1
CUDNN=1
CUDNN_HALF=0
OPENCV=0
AVX=1
OPENMP=1

# Edit GPU architecture. Uncomment line for your GPU
ARCH= -gencode arch=compute_61 for GTX1080Ti

# Build
make
# make clean
```


## Training

[Explanation of parameters in `.cfg` file](https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-%5Bnet%5D-section)

- `batch=1` number of samples (images, letters, ...) which will be precossed in one batch
- `subdivisions=1` number of mini_batches in one batch, size mini_batch = batch/subdivisions, so GPU processes mini_batch samples at once, and the weights will be updated for batch samples (1 iteration processes batch images)
- `width=416` network size (width), so every image will be resized to the network size during Training and Detection
- `height=416` network size (height), so every image will be resized to the network size during Training and Detection
- `channels=3` network size (channels), so every image will be converted to this number of channels during Training and Detection
- `inputs=256` network size (inputs) is used for non-image data: letters, prices, any custom data
- `max_chart_loss=20` max value of Loss in the image chart.png


Modify the loss function to handle class imbalance. Counters per class is more effective, but can try both. Do either
```
[yolo]
focal_loss=1  # for each of the three yolo layers
```
or
```
[yolo]
counters_per_class=100, 2000, 300, ... # number of objects per class in your Training dataset
```
and use calc_anchors to calculate the counters per class:
```
./darknet detector calc_anchors data/coco.data -num_of_clusters 9 -width 416 -height 416
```

- Atart training: `bash run_train_init.sh`
- After 1.000 iterations use multi-gpu: `bash run_train_resume.sh`
- Can stop training when average loss < 0.6


### Config Settings and Tips

- If error Out of Memory error occurs then in .cfg file you should increase subdivisions=16, 32 or 64 ([source](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L4))
- To speed up training lower subdivisions from 64 to 32 to 16 until OOM error occurs
- Set `random=1` in .cfg-file to increase precision by training on different resolutions ([source](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L788))
- Increase image resolution (eg `width=608`, `height=608`) or any multiple of 32 to increaes prevision to increase accuracy of detecting smaller objects
- possible sizes: 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, etc...
- Ensure that all objects are labeled. Unlabeled objects are scored negatively
- Dataset should include objects with varying scales, resolution, lighting, angles, backgrounds and include about 2,000 different images for each class
- Use negative samples (images that do not contain any of classes) to improve results. these are includced by adding empty .txt files. Use as many negative as positive samples.
- For training for small objects set `layers = -1, 11` instead of <https://github.com/AlexeyAB/darknet/blob/6390a5a2ab61a0bdf6f1a9a6b4a739c16b36e0d7/cfg/yolov3.cfg#L720> and set `stride=4` instead of <https://github.com/AlexeyAB/darknet/blob/6390a5a2ab61a0bdf6f1a9a6b4a739c16b36e0d7/cfg/yolov3.cfg#L717>
- If you train the model to distinguish Left and Right objects as separate classes (left/right hand, left/right-turn on road signs, ...) then for disabling flip data augmentation - add `flip=0` here: https://github.com/AlexeyAB/darknet/blob/3d2d0a7c98dbc8923d9ff705b81ff4f7940ea6ff/cfg/yolov3.cfg#L17
- General rule - your training dataset should include such a set of relative sizes of objects that you want to detect: 
    - `train_network_width * train_obj_width / train_image_width ~= detection_network_width * detection_obj_width / detection_image_width`
    - `train_network_height * train_obj_height / train_image_height ~= detection_network_height * detection_obj_height / detection_image_height`
* to speedup training (with decreasing detection accuracy) do Fine-Tuning instead of Transfer-Learning, set param `stopbackward=1` here: <https://github.com/AlexeyAB/darknet/blob/6d44529cf93211c319813c90e0c1adb34426abe5/cfg/yolov3.cfg#L548>

### .cfg files

- best detection: `csresnext50-panet-spp-original-optimal.cfg`
- small objects: `yolov3_5l.cfg`
- experimental: `Gaussian_yolov3_BDD.cfg`, `resnet152_trident.cfg`, `yolov3-voc.yolov3-giou-40.cfg`, `yolov3.coco-giou-12.cfg`


```
# cfgs for classification
vgg-conv.cfg
vgg-16.cfg
strided.cfg
resnet50.cfg
resnet101.cfg
resnet152.cfg
resnext152-32x4d.cfg
efficientnet_b0.cfg
densenet201.cfg
darknet53_448_xnor.cfg
darknet53.cfg
cifar.cfg
```

### GPU Power

Change power during training to limit power consumption

- `sudo nvidia-smi -i 0 -pl 150`  # medium-low performance
- `sudo nvidia-smi -i 0 -pl 225`  # high performance (about 17% faster than 150W)
- `sudo nvidia-smi Wi 0 -pl 250`  # max performance
