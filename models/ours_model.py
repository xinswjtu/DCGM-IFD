import torch
from torch import nn
from utils.sn_layers import Conv2d_SN, Linear_SN


class Generator(nn.Module):
    def __init__(self, z_dim, n_class , img_size=64, out_channels=3):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_class, z_dim)

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(z_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True), # (b, 128, 16, 16)-->(b, 128, 32, 32)

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True), # (b, 128, 32, 32)-->(b, 64, 64, 64)

            nn.Conv2d(64, out_channels, 3, stride=1, padding=1), # (b, 64, 64, 64)-->(b, 3, 64, 64)
            nn.Tanh()
        )

    def forward(self, noise, labels):  # noise 64 x 100; labels (64,)
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)  # (64, 100) ---> (64, 128 * 16 * 16)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size, in_channels=3, use_sn=True):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters):
            block = [Conv2d_SN(in_filters, out_filters, 3, 2, 1, use_sn=use_sn),
                     nn.LeakyReLU(0.25, inplace=True)]
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(in_channels, 16),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = img_size // 2 ** 4

        # adversarial output
        self.adv = Linear_SN(128 * ds_size ** 2, 1, use_sn=use_sn)

        # auxiliary discriminative classifier
        self.ac = Linear_SN(128 * ds_size ** 2, num_classes * 2, use_sn=use_sn)

    def forward(self, x, label, adc_fake=False):
        h = x
        out = self.conv_blocks(h)
        h = out.view(out.shape[0], -1)
        adv_output = self.adv(h)
        cls_output = self.ac(h)

        if adc_fake:
            label = label * 2 + 1
        else:
            label = label * 2
        return {"h": h, "adv_output": adv_output, "cls_output": cls_output, "label": label}
