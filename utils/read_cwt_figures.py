import random
import numpy as np
import os
from PIL import Image
import shutil
import torch


def read_directory(images_folder, n_images, n_class, normal=True, is_shuffle=False):
    images_tensor, labels_tensor = [], []
    for label in range(n_class):
        images_list = [image for image in os.listdir(images_folder) if image.endswith(f'-{label}.jpg')]
        images_list.sort(key=lambda x: int(x.split('-')[0]))
        images_list = images_list[:n_images]

        for img_list in images_list:
            images_tensor.append(np.array(Image.open(images_folder + '/' + img_list)).astype(float))
            labels_tensor.append(float(img_list.split('.')[0][-1]))

    if normal:
        images_tensor = np.array(images_tensor)
        images_tensor = (images_tensor.astype(np.float32) - 127.5) / 127.5
    else:
        images_tensor = np.array(images_tensor)

    if images_tensor.ndim == 3: # (n, h, w) --> (n, 1, h, w)
        images_tensor = np.expand_dims(images_tensor, axis=1)
    elif images_tensor.ndim == 4: # (n, h, w, c) --> (n, c, h, w)
        images_tensor = images_tensor.transpose(0, 3, 1, 2)

    labels_tensor = np.array(labels_tensor, dtype=int)

    images_tensor = torch.tensor(images_tensor, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_tensor, dtype=torch.long).view(-1)

    if is_shuffle:
        index = [i for i in range(len(images_tensor))]
        random.shuffle(index)
        images_tensor = images_tensor[index]
        labels_tensor = labels_tensor[index]
    return images_tensor, labels_tensor


