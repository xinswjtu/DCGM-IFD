import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import torchvision



class ResNet18Fc(nn.Module):
    '''
    ResNet18 model
    '''
    def __init__(self, in_channel=1, out_channel=7):
        super().__init__()
        self.model_resnet18 = models.resnet18(weights=None)
        self.model_resnet18.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model_resnet18.fc = nn.Linear(self.model_resnet18.fc.in_features, out_channel)

    def forward(self, x, features=False):
        '''x: (b, c, h, w)'''
        x = self.model_resnet18.conv1(x)
        x = self.model_resnet18.bn1(x)
        x = self.model_resnet18.relu(x)
        x = self.model_resnet18.maxpool(x)

        x = self.model_resnet18.layer1(x)
        x = self.model_resnet18.layer2(x)
        x = self.model_resnet18.layer3(x)
        x = self.model_resnet18.layer4(x)

        x = self.model_resnet18.avgpool(x)
        fea = torch.flatten(x, 1)

        x = self.model_resnet18.fc(fea)
        return (x, fea) if features else x


