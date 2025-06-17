import torch

from torch import nn, cat
from torchsparse import nn as spnn
from torchsparse.backbones.modules import SparseConvBlock, SparseConvTransposeBlock


class VoxelDecoder(nn.Module):
    def __init__(self, channels = [16, 32, 64, 128], latent_dim = 512):
        super().__init__()

        self.upsample = nn.ModuleList()
        for i in range(len(channels) - 1):
            out_channels = sum(channels[(i + 1):])
            conv = spnn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                transposed=True,
            )

            self._init_trilinear3d(conv)
            self.upsample.append(nn.Sequential(
                conv,
                spnn.BatchNorm(out_channels),
                spnn.ReLU(True)
            ))

        # self.smooth = DoubleConv(sum(channels), latent_dim)
        self.smooth = nn.Sequential(
            SparseConvBlock(sum(channels), latent_dim, kernel_size=3),
            SparseConvBlock(latent_dim, latent_dim, kernel_size=3),
            SparseConvBlock(latent_dim, latent_dim, kernel_size=1)
        )
    
    def forward(self, x):
        for i in range(len(x)-2, -1, -1):
            x[-1] = self.upsample[i](x[-1])
            x[-1].F = cat([x[-1].F, x[i].F], dim=1)

        return self.smooth(x[-1])
    
    def _init_trilinear3d(self, conv):
        K = conv.kernel_size[0]
        factor  = (K + 1) // 2
        center  = factor - 1 if K % 2 else factor - 0.5
        og      = torch.arange(K, dtype=torch.float32, device=conv.kernel.device)
        f1d     = 1 - torch.abs(og - center) / factor
        kernel3d = (f1d[:, None, None] * f1d[None, :, None] * f1d[None, None, :])
        kernel3d /= kernel3d.sum()
        kernel1d = kernel3d.reshape(-1)

        with torch.no_grad():
            conv.kernel.zero_()
            for oc in range(conv.out_channels):
                for ic in range(conv.in_channels):
                    if oc == ic:
                        conv.kernel[:, oc, ic].copy_(kernel1d)

'''
class VoxelDecoder(nn.Module):
    def __init__(self, channels, latent_dim):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Sequential(
            spnn.Conv3d(channel, latent_dim, 1, bias=False),
            spnn.BatchNorm(latent_dim),
            spnn.ReLU(inplace=True),
        ) for channel in channels)
        
        self.upsample = nn.ModuleList(nn.Sequential(
            spnn.Conv3d(latent_dim, latent_dim, kernel_size=2, stride=2, transposed=True, bias=False),
            spnn.BatchNorm(latent_dim),
            spnn.ReLU(inplace=True)
        ) for _ in channels)

        self.smooth = nn.ModuleList(nn.Sequential(
            spnn.Conv3d(latent_dim, latent_dim, 3, padding=1, bias=False),
            spnn.BatchNorm(latent_dim),
            spnn.ReLU(inplace=True)
        ) for _ in channels)

        self.last_upsample = nn.Sequential(
            spnn.Conv3d(latent_dim, latent_dim, kernel_size=2, stride=2, transposed=True, bias=False),
            spnn.BatchNorm(latent_dim),
            spnn.ReLU(inplace=True),

            spnn.Conv3d(latent_dim, latent_dim, 3, padding=1, bias=False),
            spnn.BatchNorm(latent_dim),
            spnn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        lat = [latconv(f) for latconv, f in zip(self.lateral, x)]
        lat[-1] = self.smooth[-1](lat[-1])

        for i in range(len(lat) - 1, 0, -1):
            up = self.upsample[i](lat[i])
            lat[i - 1] = lat[i - 1] + up
            lat[i - 1] = self.smooth[i - 1](lat[i - 1])

        return self.last_upsample(lat[0])
'''
