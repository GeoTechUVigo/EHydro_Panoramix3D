import torch

from typing import List

from torch import nn, cat
from torchsparse import nn as spnn, SparseTensor
from torchsparse.backbones.modules import SparseConvBlock


class VoxelDecoder(nn.Module):
    def __init__(self, channels: List[int] = [16, 32, 64, 128], latent_dim: int = 512):
        super().__init__()

        self.upsample = nn.ModuleList()
        self.smooth = nn.ModuleList()
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

            out_channels = sum(channels[i:])
            self.smooth.append(SparseConvBlock(out_channels, out_channels, kernel_size=3))

        self.mixer = nn.Sequential(
            SparseConvBlock(sum(channels), latent_dim - 3, kernel_size=3),
            SparseConvBlock(latent_dim - 3, latent_dim - 3, kernel_size=3),
            SparseConvBlock(latent_dim - 3, latent_dim - 3, kernel_size=1)
        )
    
    def forward(self, x: SparseTensor) -> SparseTensor:
        for i in range(len(x)-2, -1, -1):
            x[-1] = self.upsample[i](x[-1])
            x[-1].F = cat([x[-1].F, x[i].F], dim=1)
            x[-1] = self.smooth[i](x[-1])

        mix = self.mixer(x[-1])
        mix.F = cat([mix.F, mix.C[:, 1:]], dim=1)
        return mix
    
    def _init_trilinear3d(self, conv: spnn.Conv3d) -> None:
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
