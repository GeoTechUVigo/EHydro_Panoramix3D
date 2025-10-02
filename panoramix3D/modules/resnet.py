import torch

from typing import List, Tuple, Union

from torch import nn
from torchsparse import SparseTensor
from torchsparse.backbones.modules import SparseConvBlock, SparseResBlock

__all__ = ["SparseResNet21D"]

class SparseResNet(nn.Module):
    def __init__(
        self,
        blocks: List[
            Tuple[int, int, Union[int, Tuple[int, ...]], Union[int, Tuple[int, ...]]]
        ],
        *,
        in_channels: int = 4
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.in_channels = in_channels

        self.conv_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        for num_blocks, out_channels, kernel_size, stride in blocks:
            self.conv_blocks.append(
                SparseConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                )
            )

            res_blocks = []
            for i in range(1, num_blocks):
                res_blocks.append(
                    SparseResBlock(
                        out_channels,
                        out_channels,
                        kernel_size,
                    )
                )

            self.res_blocks.append(nn.Sequential(*res_blocks))
            in_channels = out_channels

    def _mult_3d_tuple(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return tuple([a[0] * b[0], a[1] * b[1], a[2] * b[2]])

    def forward(self, x: SparseTensor) -> List[SparseTensor]:
        outputs = []
        for idx in range(len(self.blocks)):
            if x.C.size(0) == 0 or x.C[:, 3].max().item() - x.C[:, 3].min().item() < 2:
                return None
                x = SparseTensor(
                    coords=torch.zeros((0, 4), dtype=torch.int32, device=x.C.device),
                    feats=torch.zeros((0, self.blocks[idx][1]), dtype=x.F.dtype, device=x.F.device),
                    stride=self._mult_3d_tuple(x.stride, self.blocks[idx][3]),
                )
            else:
                x = self.conv_blocks[idx](x)
                x = self.res_blocks[idx](x)

            outputs.append(x)
        return outputs
