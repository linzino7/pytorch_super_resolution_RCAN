# pytorch 3X super resolution RCAN by BasicSR
This Rep is HW04 in NCTU 202009 Selected Topics in Visual Recognition using Deep Learning.  

3x super resolution trained by RCAN.

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
- Download trainging Data: https://drive.google.com/file/d/1QUMTskptKjQwSaSX4b0VtZIRpldTtNKn/view?usp=sharing
- Download testing Data: https://github.com/linzino7/pytorch_super_resolution_RCAN_BasicSR/blob/main/data/testing_lr_images.zip

Download and extract *training_hr_images.zip* and *testing_lr_images.zip* to *data* directory.

### Transform data
```
$ python 
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

- RCAN
```
$ python3 BasicSR/basicsr/train.py -opt options/train_RCANx3_g64b32_gt96_te384.yml
```
- EDSR
```
$  python3 BasicSR/basicsr/train.py -opt options/train_EDSR_Mx3_te384.yml
```
- MSRResNet
```
$ python3 BasicSR/basicsr/train.py -opt options/train_MSRResNet_x3_z.yml
```

The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time | Bach Size |
------------ | ------------- | ------------- | ------------- | ------------- | -------------|
RCAN | 1x NVIDIA GTX 1080 | 384x384 | 1 | s mins | 32 |
EDSR | 1x NVIDIA GTX 1080 | 384x384 | 1 | 1 mins | 64 |
MSRResNet | 1x NVIDIA GTX 1080 |384x384 | 1 | 4 mins | 8 |
### Muti-GPU Training
```
$ python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 BasicSR/basicsr/train.py \
  -opt options/train_RCANx3_g64b32_gt96_te384.yml --launcher pytorch
```

## Inference

### Inference images in folder
- RCAN
```
$ python3 inference/inference_RCAN.py \
 --model_path BasicSR/experiments/201_RCANx3_g64b32_gt96_te384/models/net_g_2000.pth \
 --folder data/testing_lr_images/ \
 --output_folder results/RCANx3_g64b32_gt96_te384_2000/ 
```
- EDSR
```
$ python3 inference/inference_EDSR_Mx3.py \
 --model_path BasicSR/experiments/202_EDSR_Mx3_f64b16_te384/models/net_g_6000.pth \
 --folder data/testing_lr_images/ \
 --output_folder results/EDSR_Mx3_f64b16_te384_6000/ 
```
- MSRResNet
```
$ python3 BasicSR/inference/inference_MSRResNet.py \
 --model_path BasicSR/experiments/003_MSRResNet_x3_f64b16_hw04_192/models/net_g_20000.pth \
 --folder data/testing_lr_images/
```

## result by PSNR 25.07
ground trouth 04.png：

![](https://github.com/linzino7/pytorch_super_resolution_RCAN_BasicSR/blob/main/data/testing_lr_images/04.png)

result 04.png：

![](https://github.com/linzino7/pytorch_super_resolution_RCAN_BasicSR/blob/main/result/RCANx3_g64b32_gt96_te384_3000_4_psnr_25.078_______/04.png)

ground trouth 06.png：

![](https://github.com/linzino7/pytorch_super_resolution_RCAN_BasicSR/blob/main/data/testing_lr_images/06.png)

result 06.png：

![](https://github.com/linzino7/pytorch_super_resolution_RCAN_BasicSR/blob/main/result/RCANx3_g64b32_gt96_te384_3000_4_psnr_25.078_______/06.png)

ground trouth 09.png：

![](https://github.com/linzino7/pytorch_super_resolution_RCAN_BasicSR/blob/main/data/testing_lr_images/09.png)

result 09.png：

![](https://github.com/linzino7/pytorch_super_resolution_RCAN_BasicSR/blob/main/result/RCANx3_g64b32_gt96_te384_3000_4_psnr_25.078_______/09.png)

ground trouth 11.png：

![](https://github.com/linzino7/pytorch_super_resolution_RCAN_BasicSR/blob/main/data/testing_lr_images/11.png)

result 11.png：

![](https://github.com/linzino7/pytorch_super_resolution_RCAN_BasicSR/blob/main/result/RCANx3_g64b32_gt96_te384_3000_4_psnr_25.078_______/11.png)

# Reference:
- Paper RCAN: [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/abs/1807.02758)
- BasicSR: https://github.com/xinntao/BasicSR
