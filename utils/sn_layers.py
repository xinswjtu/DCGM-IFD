# sn_layers.py
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Conv2d_SN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_sn=False, **kwargs):
        super(Conv2d_SN, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        if use_sn:
            conv = spectral_norm(conv)
        self.conv = conv

    def forward(self, x):
        return self.conv(x)

class ConvTranspose2d_SN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, use_sn=False, **kwargs):
        super(ConvTranspose2d_SN, self).__init__()
        conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, **kwargs)
        if use_sn:
            conv_transpose = spectral_norm(conv_transpose)
        self.conv_transpose = conv_transpose

    def forward(self, x):
        return self.conv_transpose(x)

class Linear_SN(nn.Module):
    def __init__(self, in_features, out_features, use_sn=False, **kwargs):
        super(Linear_SN, self).__init__()
        linear = nn.Linear(in_features, out_features, **kwargs)
        if use_sn:
            linear = spectral_norm(linear)
        self.linear = linear

    def forward(self, x):
        return self.linear(x)

class Embedding_SN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, use_sn=False, **kwargs):
        super(Embedding_SN, self).__init__()
        embedding = nn.Embedding(num_embeddings, embedding_dim, **kwargs)
        if use_sn:
            embedding = spectral_norm(embedding)
        self.embedding = embedding

    def forward(self, x):
        return self.embedding(x)
