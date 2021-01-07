# pytorch super resolution RCAN by BasicSR
This Rep is HW04 in NCTU 202009 Selected Topics in Visual Recognition using Deep Learning.  

super resolution trained by RCAN.

Model training by  [BasicSR](https://github.com/xinntao/BasicSR). 

## Hardware
- Ubuntu 18.04 LTS
- Intel(R) Core(TM) i7-6900K CPU @ 3.20GHz
- 31 RAM
- NVIDIA RTX 1080 8G * 2

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:

1.  [Dataset Preparation](#Dataset-Preparation)
2.  [Training](#Training)
3.  [Inference](#Inference)

## Prepare Env.
### clone BasicSR
```
$ git clone https://github.com/xinntao/BasicSR.git
```

### Project directory structured
When you are ready, your directory structured as follow:
```
BasicSR/
data/
|+ testing_lr_images/
|+ training_hr_images/
inference/
|- inference_RCAN.py
options/
|- train_RCANx3_g128b32_gt96_te384.yml
results/
pre_preocess.py
```

## Dataset Preparation

### Prepare Images
#### Download Classes Image
Download trainging Data: https://drive.google.com/file/d/1QUMTskptKjQwSaSX4b0VtZIRpldTtNKn/view?usp=sharing
Download testing Data: https://github.com/linzino7/pytorch_super_resolution_RCAN_BasicSR/blob/main/data/testing_lr_images.zip

Download and extract *training_hr_images.zip* and *testing_lr_images.zip* to *data* directory.

### Transform data
Use construct_datasets.py to make train.txt .

```
# train.txt and val.txt  
# left(x1) top(y1)  right(x2) bottom(y2) label
image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
...
```

Names file  example is in [data/SVHN.names](https://github.com/linzino7/pytorch-YOLOv4/blob/master/data/SVHN.names)
```
# names file
Label1
Label2
Label3
...
```

### Prepare Images
After downloading images and pre-process, the data directory is structured as:
```
 data/
  | +- training_hr_images/
  | +- testing_lr_images/
  | +- T_hr_images_te384/
  | +- T_lr_images_te384/
  | +- val_hr_images_te384/
  | +- val_lr_images_te384/
```

## Training
### Setting
You can setting bach size and epoch in [options/train_RCANx3_g64b32_gt96_te384.yml](https://github.com/linzino7/pytorch_super_resolution_RCAN_BasicSR/blob/main/options/train_RCANx3_g128b32_gt96_te384.yml)

### Train models
To train models, run following commands.
```
$ python3 BasicSR/basicsr/train.py -opt options/train_RCANx3_g64b32_gt96_te384.yml
```
The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time | Bach Size |
------------ | ------------- | ------------- | ------------- | ------------- | -------------|
YOLOv4 | 1x NVIDIA T4 | 608x608 | 1 | 2.5 hours | 4 |
YOLOv4 | 4x NVIDIA GTX 1080 | 608x608 | 1 | 0.6 hour | 32 |

### Muti-GPU Training
```
$ python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 BasicSR/basicsr/train.py \
  -opt options/train_RCANx3_g64b32_gt96_te384.yml --launcher pytorch
```

## Inference

### Inference images in folder
```
$ python3 inference/inference_RCAN.py \
 --model_path BasicSR/experiments/201_RCANx3_g64b32_gt96_te384/models/net_g_2000.pth \
 --folder data/testing_lr_images/ \
 --output_folder results/RCANx3_g64b32_gt96_te384_2000/ 
```

# Reference:
- Paper RCAN: [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/abs/1807.02758)

-----

- [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
- https://github.com/eriklindernoren/PyTorch-YOLOv3
- https://github.com/marvis/pytorch-caffe-darknet-convert
- https://github.com/marvis/pytorch-yolo3
- Paper Yolo v4: https://arxiv.org/abs/2004.10934
- Source code:https://github.com/AlexeyAB/darknet
- More details: http://pjreddie.com/darknet/yolo/
