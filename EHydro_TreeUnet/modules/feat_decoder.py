import torch

from typing import List, Tuple, Union

from torch import nn, cat, Tensor
from torch_scatter import scatter_mean

from torchsparse import nn as spnn, SparseTensor
from torchsparse.nn import functional as spf
from torchsparse.backbones.modules import SparseResBlock, SparseConvTransposeBlock


class FeatDecoder(nn.Module):
    def __init__(
            self,
            blocks: List[Tuple[int, int, Union[int, Tuple[int, int, int]], Union[int, Tuple[int, int, int]]]],
            out_dim: int = 1,
            aux_dim: int = 0,
            bias: bool = True,
            init_bias: float = None
        ):
        super().__init__()

        self.upsample = nn.ModuleList()
        self.conv = nn.ModuleList()

        for i in range(len(blocks) - 1):
            _, out_channels, out_kernel_size, _ = blocks[i]
            _, in_channels, in_kernel_size, stride = blocks[i + 1]

            self.upsample.append(SparseConvTransposeBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=in_kernel_size,
                stride=stride
            ))

            self.conv.append(nn.Sequential(
                SparseResBlock(
                    in_channels=out_channels * 2,
                    out_channels=out_channels,
                    kernel_size=out_kernel_size,
                ),
                SparseResBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=out_kernel_size,
                )
            ))

        self.proj = spnn.Conv3d(
            in_channels=blocks[0][1] + aux_dim,
            out_channels=out_dim,
            kernel_size=1,
            stride=1,
            bias=bias
        )

        if bias and init_bias is not None:
            nn.init.constant_(self.proj.bias, val=init_bias)

    def _union_sparse_layers(self, A: SparseTensor, B: SparseTensor) -> SparseTensor:
        B.F = B.F.to(A.F.dtype)

        hashA = spf.sphash(A.C)
        hashB = spf.sphash(B.C)

        idxA_to_B = spf.sphashquery(hashA, hashB)
        outB = torch.zeros((A.C.size(0), B.F.size(1)), device=A.F.device, dtype=A.F.dtype)
        mask = idxA_to_B >= 0
        if mask.any():
            ai = torch.nonzero(mask, as_tuple=False).squeeze(1)
            bi = idxA_to_B[mask].to(torch.long)
            outB[ai] = B.F[bi]
        
        return SparseTensor(coords=A.C, feats=cat((A.F, outB.to(A.F.dtype)), dim=1))

    def forward(self, x: List[SparseTensor], aux: SparseTensor = None, mask: Tensor = None, new_coords: Tensor = None, reduce: Tensor = None) -> SparseTensor:
        current = x[-1]
        for i in range(len(x)-2, -1, -1):
            current = self.upsample[i](current)
            current.F = cat([current.F, x[i].F], dim=1)
            current = self.conv[i](current)
        
        if mask is not None:
            current.C = current.C[mask]
            current.F = current.F[mask]
            
        if aux is not None:
            current = self._union_sparse_layers(current, aux)

        if new_coords is not None:
            current.C = new_coords
        
        if reduce is not None:
            current.F = scatter_mean(current.F, reduce, dim=0)
            

        return self.proj(current)