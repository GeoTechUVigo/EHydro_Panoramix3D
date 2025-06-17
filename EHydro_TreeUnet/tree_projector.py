from .modules import VoxelDecoder
from torch import nn
from torchsparse import nn as spnn
from torchsparse.backbones.resnet import SparseResNet


class TreeProjector(nn.Module):
    def __init__(self, in_channels, num_classes, max_instances, channels = [16, 32, 64, 128], latent_dim = 512):
        super().__init__()
        # self.encoder = Encoder(in_channels, channels)
        blocks = [(3, channels[0], 3, 1)]
        for channel in channels[1:]:
            blocks.append((3, channel, 3, 2))
        # blocks.append((1, channels[-1], (1, 3, 1), (1, 2, 1)))

        self.encoder = SparseResNet(
            blocks = blocks,
            in_channels=in_channels
        )

        self.voxel_decoder = VoxelDecoder(channels, latent_dim)
        self.semantic_head = spnn.Conv3d(latent_dim, num_classes, 1, bias=False)
        # self.instance_head = spnn.Conv3d(latent_dim, max_instances, 1, bias=False)

    def forward(self, x):
        feats = self.voxel_decoder(self.encoder(x))

        return self.semantic_head(feats)
    

'''
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(inplace=True),
            spnn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)
    
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            spnn.Conv3d(out_channels, out_channels, kernel_size=2, stride=2, bias=False),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class Encoder(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.down_convolutions = nn.ModuleList([DownSample(in_channels, channels[0])])
        for i in range(len(channels) - 1):
            self.down_convolutions.append(DownSample(channels[i], channels[i + 1]))

    def forward(self, x):
        downs = [x]
        for down_convolution in self.down_convolutions:
            x = down_convolution(x)
            downs.append(x)

        return downs
'''