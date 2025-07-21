# DCGM-IFD

This repository contains the official implementation of our paper titled *"Discriminative Condition-Guided Generative Model for Induction Motor Fault Diagnosis with Limited Data"*. 

The corresponding manuscript has been submitted to the journal for peer review.

The complete source code with detailed annotations and usage instructions will be released shortly after the manuscript review process. Please watch this repository for updates.

## Usage

### Requirements

* Python: 3.8.16
* PyTorch: 2.0.0+cu118
* Torchvision: 0.15.0+cu118
* CUDA: 11.8
* CUDNN: 8700
* NumPy: 1.23.4
* PIL: 10.0.1

### Data preparation

The data utilized in this study is proprietary; however, anyone can prepare their own dataset by adhering to the following directory structure

```
./datasets/Motor/real_images/
├── test
│   ├── 0-0.jpg
│   ├── 1-0.jpg
...
│   ├── 100-1.jpg
...
├── train
│   ├── 0-0.jpg
│   ├── 1-0.jpg
...
│   ├── 200-1.jpg
...
└── valid
    ├── 0-0.jpg
    ├── 1-0.jpg
    ...
```

### Training DCGM for data generation

To run `train_gan.py`

```bash
$ bash scripts/train.sh
```

OR：

```bash
$python train_gan.py --epochs 10000 --n_train 20 --cond_lambda 0.5 --n_critic 3
```

### Training ResNet for classification (downstream task)

To run `train_cls.py`

```bash
$python train_cls.py
```

