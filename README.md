# Real-time Image Demoireing on Mobile Devices

Pytorch implementation of our paper accepted by ICLR 2023 -- "Real-time Image Demoireing on Mobile Devices".

## Requirements

- python 3.7
- pytorch 1.9.0
- torchvision 0.11.3
- opencv-python-headless 4.6
- colour 0.1.5
- scikit-image 0.19.3

## Taining

First, split and divide the training dataset by the following command:

```shell
sh run/split_dataset.sh
```

Note that the data path of demoireing benchmarks should be modified in `/data_script/aim/slit_patches_train.py`

Second, run the command scripts in `run/` to accelerate models on different benchmarks. For example, to reproduce the results of DDA for accelerating MBCNN on FHDMI, run:

```shell
sh run/mbcnn_fhdmi.sh
```

## Evaluating

The checkpoint file of the accelerated models are provided in the following anonymous link. To evaluate them, download the model file and place it into  `/ckpt`  and then run the command script in `run/`. For example, to evaluate the accelerated model of DDA for accelerating MBCNN on FHDMI, run:

```shell
sh run/test_mbcnn_fhdmi.sh
```

| Model     | Dataset  | PSNR  | FLOPs reduction | Link                                                         |
| --------- | -------- | ----- | --------------- | ------------------------------------------------------------ |
| DMCNN     | LCDMoire | 34.19 | 0%              | [Link](https://drive.google.com/file/d/1bSyRNEBV1vW1kp7VXE-q1ELsBjcFBHMH/view?usp=sharing) |
| DMCNN-DDA | LCDMoire | 34.58 | 55.1%           | [Link](https://drive.google.com/file/d/1lHZyfGcds9QaFtJFkeWGMk5sv8H6Q2bJ/view?usp=sharing) |
| DMCNN     | FHDMI    | 21.69 | 0%              | [Link](https://drive.google.com/file/d/12z690vkzr___LKdTrCchDXHz3leP7e5T/view?usp=sharing) |
| DMCNN-DDA | FHDMI    | 21.86 | 52.3%           | [Link](https://drive.google.com/file/d/1lHZyfGcds9QaFtJFkeWGMk5sv8H6Q2bJ/view?usp=sharing) |
| MBCNN     | LCDMoire | 43.95 | 0%              | [Link](https://drive.google.com/file/d/12z690vkzr___LKdTrCchDXHz3leP7e5T/view?usp=sharing) |
| MBCNN-DDA | LCDMoire | 41.68 | 46.9%           | [Link](https://drive.google.com/file/d/1lHZyfGcds9QaFtJFkeWGMk5sv8H6Q2bJ/view?usp=sharing) |
| MBCNN     | FHDMI    | 23.27 | 0%              | [Link](https://drive.google.com/file/d/1olk-vq_zqfbOIeqcNHEMrmpb7HxaHHwH/view?usp=sharing) |
| MBCNN-DDA | FHDMI    | 23.62 | 45.2%           | [Link](https://drive.google.com/file/d/1lHZyfGcds9QaFtJFkeWGMk5sv8H6Q2bJ/view?usp=sharing) |

Any problem, feel free to contact [yuxinzhang@stu.xmu.edu.cn](
