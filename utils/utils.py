import csv
import math
import shutil
import sys
import PIL
import torchvision
from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
import os
import random
from contextlib import contextmanager
import argparse
from utils.read_cwt_figures import read_directory
from torch.utils.data import TensorDataset
from torch.nn import init
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset(args, data_type='train'):
    '''
    Args:
        args:
        data_type: 'train', 'valid', 'test'

    Returns:
        train_x, train_y, train_ds
    '''
    train_x, train_y = read_directory(os.path.join(args.real_folder, f'{data_type}'), args.n_train, args.n_class)
    train_ds = TensorDataset(train_x, train_y)
    return train_x, train_y, train_ds

def set_bn_trainable(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.train()


def untrack_bn_statistics(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.track_running_stats = False


def track_bn_statistics(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.track_running_stats = True


def make_GAN_trainable(*models):
    for model in models:
        if model is not None:
            model.train()
            model.apply(track_bn_statistics)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def model_param(model):
    """calculate the sum of the model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@contextmanager
def module_no_grad(m: torch.nn.Module):
    requires_grad_dict = dict()
    for name, param in m.named_parameters():
        requires_grad_dict[name] = param.requires_grad
        param.requires_grad_(False)
    yield m
    for name, param in m.named_parameters():
        param.requires_grad_(requires_grad_dict[name])


def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()


def print_environ():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {} \n".format(PIL.__version__))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y


def lambda_sigmoid(t, alpha_min=0.01, alpha_max=1.0, k=0.02, t0=50):
    return alpha_min + (alpha_max - alpha_min) / (1 + np.exp(-k*(t - t0)))


def init_dict(*keys):
    '''
    example:
        loss = init_dict('dloss', 'gloss', 'c_dloss', 'c_acc', 'adv_dloss', 'adv_gloss')
    return:
        loss = {'dloss':[], 'gloss':[], 'c_dloss':[], 'c_acc':[], 'adv_dloss':[], 'adv_gloss':[]}
    '''
    return {key: [] for key in keys}


class Metric(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def tensor2img(tensor, label, images_folder, zero_one=False):
    """
    :param tensor: (b, c, h, w), tensor
    :param label: (b,), tensor, torch.long
    :param images_folder:指定图片保存的文件夹
    :return:
            images_folder/i-j.jpg，其中j是对应的label，i是序号从0 ~ b

    注意：
        zero_one = True, 表示输入的tensor范围是[0, 1], 否则输入的范围是[-1, 1]
    """
    tensor = tensor.detach().cpu()
    label = label.detach().cpu().view(-1).to(torch.long)

    if os.path.exists(images_folder):
        shutil.rmtree(images_folder)
    os.makedirs(images_folder)

    if tensor.dim() == 4 and tensor.shape[1] == 1: # 灰度图(b, 1, h, w)
        if zero_one: # [0, 1] --> [0, 255]
            for i in range(tensor.shape[0]):
                im = (tensor[i] * 255).squeeze(0).numpy().astype(np.uint8) # (1, h, w)-->(h, w), uint8, [0, 255]
                im = Image.fromarray(im, 'L')
                im.save(os.path.join(images_folder, f'{i}-{label[i].item()}.jpg'), dpi=(96, 96))
        else: # [-1, 1] --> [0, 255]
            for i in range(tensor.shape[0]):
                im = (tensor[i] * 127.5 + 127.5).squeeze(0).numpy().astype(np.uint8) # (1, h, w)-->(h, w), uint8, [0, 255]
                im = Image.fromarray(im, 'L')
                im.save(os.path.join(images_folder, f'{i}-{label[i].item()}.jpg'), dpi=(96, 96))
    elif tensor.dim() == 4 and tensor.shape[1] == 3: # RGB类型(b, 3, h, w)
        if zero_one: # [0, 1] --> [0, 255]
            for i in range(tensor.shape[0]):
                im = (tensor[i] * 255).byte().permute(1, 2, 0).numpy()          # [0, 1] --> [0, 255]
                plt.imsave(os.path.join(images_folder, f'{i}-{label[i].item()}.jpg'), im, dpi=96)
        else: # [-1, 1] --> [0, 255]
            for i in range(tensor.shape[0]):
                im = (tensor[i] * 127.5 + 127.5).byte().permute(1, 2, 0).numpy()  # [-1, 1] --> [0, 255]
                plt.imsave(os.path.join(images_folder, f'{i}-{label[i].item()}.jpg'), im, dpi=96)
    else:
        raise ValueError('请检查输入数据的维度，确保是(b, c, h, w), 且c=1或3')

