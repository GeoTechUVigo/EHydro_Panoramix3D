import torchsparse

from torch import nn
from torchsparse import SparseTensor
from torchsparse import nn as spnn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(inplace=True),
            spnn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)
    
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.Sequential(
            spnn.Conv3d(out_channels, out_channels, kernel_size=2, stride=2),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(inplace=True)
        )

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p
    
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            spnn.Conv3d(in_channels, in_channels // 2, kernel_size=2, stride=2, transposed=True, generative=False),
            spnn.BatchNorm(in_channels // 2),
            spnn.ReLU(inplace=True)
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torchsparse.cat([x1, x2])
        
        return self.conv(x)
    
class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels, deep):
        super().__init__()
        channels = [base_channels * (2 ** i) for i in range(deep)]

        self.down_convolutions = [DownSample(in_channels, channels[0])]
        for i in range(deep - 1):
            self.down_convolutions.append(DownSample(channels[i], channels[i + 1]) for i in range(deep - 1))

        self.bottle_neck = DoubleConv(channels[-1], channels[-1] * 2)

    def forward(self, x):
        downs = []
        for down_convolution in self.down_convolutions:
            down, x = down_convolution(x)
            downs.append(down)

        downs = downs[::-1]
        b = self.bottle_neck(x)
        return b, downs