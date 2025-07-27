import torch

from typing import List, Tuple, Union

from torch import nn, cat
from torchsparse import nn as spnn, SparseTensor
from torchsparse.backbones.modules import SparseConvBlock


class VoxelDecoder(nn.Module):
    def __init__(
            self,
            blocks: List[Tuple[int, int, Union[int, Tuple[int, int, int]], Union[int, Tuple[int, int, int]]]] = [
                (3, 16, 3, 1),
                (3, 32, 3, 2),
                (3, 64, 3, 2),
                (3, 128, 3, 2),
                (1, 128, (1, 3, 1), (1, 2, 1)),
            ],
            latent_dim: int = 512):
        super().__init__()

        self.upsample = nn.ModuleList()
        self.smooth = nn.ModuleList()
        for i in range(len(blocks) - 1):
            _, _, kernel_size, stride = blocks[i + 1]
            out_channels = sum(b[1] for b in blocks[i + 1:])

            conv = spnn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                transposed=True,
            )

            self._init_upsample_conv(conv)
            self.upsample.append(nn.Sequential(
                conv,
                spnn.BatchNorm(out_channels),
                spnn.ReLU(True)
            ))

            out_channels = sum(b[1] for b in blocks[i:])
            self.smooth.append(SparseConvBlock(out_channels, out_channels, kernel_size=3))

        self.mixer = nn.Sequential(
            SparseConvBlock(sum(b[1] for b in blocks), latent_dim - 3, kernel_size=3),
            SparseConvBlock(latent_dim - 3, latent_dim - 3, kernel_size=3),
            SparseConvBlock(latent_dim - 3, latent_dim - 3, kernel_size=1)
        )
    
    def forward(self, x: List[SparseTensor]) -> SparseTensor:
        for i in range(len(x)-2, -1, -1):
            x[-1] = self.upsample[i](x[-1])
            x[-1].F = cat([x[-1].F, x[i].F], dim=1)
            x[-1] = self.smooth[i](x[-1])

        mix = self.mixer(x[-1])
        mix.F = cat([mix.F, mix.C[:, 1:]], dim=1)
        return mix
    
    def _make_f(self, og):
        K = og.numel()
        factor = (K + 1) // 2
        center = factor - 1 if K % 2 else factor - 0.5

        return 1 - torch.abs(og - center) / factor
    
    def _init_upsample_conv(self, conv: spnn.Conv3d) -> None:
        kD, kH, kW = conv.kernel_size
        ogD = torch.arange(kD, dtype=torch.float32, device=conv.kernel.device)
        ogH = torch.arange(kH, dtype=torch.float32, device=conv.kernel.device)
        ogW = torch.arange(kW, dtype=torch.float32, device=conv.kernel.device)

        f_d = self._make_f(ogD)
        f_h = self._make_f(ogH)
        f_w = self._make_f(ogW)

        kernel3d = (f_d[:,None,None] * f_h[None,:,None] * f_w[None,None,:])
        kernel3d /= kernel3d.sum()
        kernel1d = kernel3d.reshape(-1)

        with torch.no_grad():
            conv.kernel.zero_()
            for oc in range(conv.out_channels):
                for ic in range(conv.in_channels):
                    if oc == ic:
                        conv.kernel[:, oc, ic].copy_(kernel1d)
    
    '''
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
    '''
