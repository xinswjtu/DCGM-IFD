# -*- coding: utf-8 -*-            
# @Time : 2022/11/17 11:44
# @Author: Small Orange
"""加载文件下的时频图片"""
import random
import numpy as np
import os
from PIL import Image
import shutil
import torch


def read_directory(images_folder, n_images, n_class, normal=True, is_shuffle=False):
    """
    读取文件夹下的图片
    :param images_folder:文件夹下包含大量的训练图片(测试图片)
    :param normal: 是否归一化。
    :param n_img:从source_folder中每个类别选取num_img张复制到target_folder
    :param n_class:类别数
    :return:
            data -- numpy, (n, c, h, w), 范围[-1, 1]
            label -- numpy, int, (n,)
    """

    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"not found '{images_folder}'")

    if (n_images * n_class) > len(os.listdir(images_folder)):
        raise ValueError('取出的图片数量超出了')

    images_tensor, labels_tensor = [], []
    for label in range(n_class):
        images_list = [image for image in os.listdir(images_folder) if image.endswith(f'-{label}.jpg')]
        images_list.sort(key=lambda x: int(x.split('-')[0]))
        images_list = images_list[:n_images]

        for img_list in images_list:
            images_tensor.append(np.array(Image.open(images_folder + '/' + img_list)).astype(float))  # PIL to numpy
            labels_tensor.append(float(img_list.split('.')[0][-1]))  # [-1]索引的是后面的标签

    if normal:
        images_tensor = np.array(images_tensor)
        images_tensor = (images_tensor.astype(np.float32) - 127.5) / 127.5  # 归一化[-1, 1]
    else:
        images_tensor = np.array(images_tensor)  # 把列表变成numpy类型 data:(n, h, w, c), float32

    # 转换成标准的(N, C, H, W)类型
    if images_tensor.ndim == 3: # (n, h, w) --> (n, 1, h, w)
        images_tensor = np.expand_dims(images_tensor, axis=1)
    elif images_tensor.ndim == 4: # (n, h, w, c) --> (n, c, h, w)
        images_tensor = images_tensor.transpose(0, 3, 1, 2)

    labels_tensor = np.array(labels_tensor, dtype=int)

    # numpy to tensor
    images_tensor = torch.tensor(images_tensor, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_tensor, dtype=torch.long).view(-1)

    # shuffle
    if is_shuffle: # 不用打乱顺序
        index = [i for i in range(len(images_tensor))]
        random.shuffle(index)
        images_tensor = images_tensor[index]
        labels_tensor = labels_tensor[index]
    # a, b = torch.unique(labels_tensor, return_counts=True)
    # print('Dataset info:')
    # print(f'     {images_tensor.shape}, {labels_tensor.shape}, {a.tolist()}, {b.tolist()}', '\n')
    return images_tensor, labels_tensor


