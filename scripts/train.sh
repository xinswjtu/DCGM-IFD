#!/bin/bash
cd "$(dirname "$0")/.." || exit 1



$torchpy train_gan.py --n_train 10
$torchpy train_gan.py --n_train 15
$torchpy train_gan.py --n_train 20
$torchpy train_gan.py --n_train 25
$torchpy train_gan.py --n_train 30














