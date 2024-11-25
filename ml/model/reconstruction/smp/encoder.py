import math

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, linear_bias=True, bn_affine=True):
        super(BasicBlock, self).__init__()

        self.linear_bias = linear_bias
        self.bn_affine = bn_affine

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, affine=self.bn_affine)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=self.bn_affine)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes, affine=self.bn_affine),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        F.relu(out, inplace=True)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, linear_bias=True, bn_affine=True):
        super(Bottleneck, self).__init__()

        self.linear_bias = linear_bias
        self.bn_affine = bn_affine

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=self.bn_affine)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=self.bn_affine)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes, affine=self.bn_affine)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes, affine=self.bn_affine),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        F.relu(out, inplace=True)
        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block, num_blocks, num_channels=3, init_stride=1, linear_bias=True, bn_affine=True,
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.linear_bias = linear_bias
        self.bn_affine = bn_affine

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=init_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=self.bn_affine)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self._out_channels = [num_channels, 64, 64, 128, 256, 512]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.linear_bias, self.bn_affine))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        return out


class ResNet18C(ResNet):
    def __init__(
            self, num_channels=3, init_stride=1, linear_bias=True, bn_affine=True
    ):
        super(ResNet18C, self).__init__(
            BasicBlock, [2, 2, 2, 2], num_channels=num_channels, init_stride=init_stride,
            linear_bias=linear_bias, bn_affine=bn_affine
        )


class ResNet18T(ResNet):
    def __init__(
            self, num_channels=3, init_stride=2, linear_bias=True, bn_affine=True
    ):
        super(ResNet18T, self).__init__(
            BasicBlock, [2, 2, 2, 2], num_channels=num_channels, init_stride=init_stride,
            linear_bias=linear_bias, bn_affine=bn_affine
        )


if __name__ == '__main__':
    import torch

    net = ResNet18C()
    x = torch.randn(1, 3, 64, 64)
    y = net(x)

    print(y.size())
