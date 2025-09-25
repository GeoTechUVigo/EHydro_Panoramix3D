import torch

from typing import List, Tuple, Union

from torch import nn, cat, Tensor
from torch_scatter import scatter_mean

from torchsparse import nn as spnn, SparseTensor
from torchsparse.nn import functional as spf
from torchsparse.backbones.modules import SparseResBlock, SparseConvTransposeBlock


class FeatDecoder(nn.Module):
    """
    A multi-scale feature decoder that progressively upsamples and fuses features
    from different resolution levels. It uses transpose convolutions for upsampling
    and residual blocks for feature refinement at each level, implementing a
    U-Net-like decoder structure for sparse tensors.

    Args:
        blocks: List of tuples defining the encoder architecture to reverse.
            Each tuple contains (dim_in, dim_out, kernel_size, stride).
            The decoder processes these in reverse order, upsampling from
            the deepest (last) level to the shallowest (first) level.
        out_dim: Number of output channels for the final projection layer.
        aux_dim: Number of auxiliary feature channels to concatenate before
            final projection (e.g., from offset heads or other modalities).
        bias: Whether to use bias in the final projection layer.
        init_bias: Initial bias value for the final layer. Useful for
            setting priors (e.g., for centroid density in detection heads).

    Example:
        >>> decoder = FeatDecoder(
        ...     blocks=[(64, 32, 3, 1), (32, 16, 3, 2), (16, 8, 3, 2)],
        ...     out_dim=1,
        ...     aux_dim=3,
        ...     bias=True,
        ...     init_bias=-2.0
        ... )
        >>> output = decoder(encoder_features, aux_feats, mask)
        >>> print(f"Output shape: {output.F.shape}")
        Output shape: torch.Size([N_voxels, 1])
    """
    def __init__(
            self,
            blocks: List[Tuple[int, int, Union[int, Tuple[int, int, int]], Union[int, Tuple[int, int, int]]]],
            out_dim: int = 1,
            aux_dim: int = 0,
            bias: bool = True,
            init_bias: float = None,
            batch_norm: bool = False,
            relu: bool = False
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

        self.batch_norm = None
        self.relu = None

        if batch_norm:
            self.batch_norm = spnn.BatchNorm(out_dim)
        if relu:
            self.relu = spnn.ReLU(True)
        if bias and init_bias is not None:
            nn.init.constant_(self.proj.bias, val=init_bias)

    def _union_sparse_layers(self, A: SparseTensor, B: SparseTensor) -> SparseTensor:
        """
        Merge two SparseTensors by concatenating their features at matching coordinates.
        
        This method finds coordinates that exist in both A and B, then concatenates
        B's features to A's features at those locations. For coordinates in A that
        don't exist in B, zero features are used for B's contribution.

        Args:
            A: Primary SparseTensor whose coordinate structure is preserved.
            B: Auxiliary SparseTensor to merge. Features are aligned to A's coords.

        Returns:
            SparseTensor with A's coordinates and concatenated features [A.F, B_aligned.F].

        Notes:
            - B's features are cast to A's dtype for compatibility.
            - Missing coordinates in B contribute zero features.
            - Uses sparse hashing for efficient coordinate lookup.
        """
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

    def forward(self, x: List[SparseTensor], aux: Tensor = None, mask: Tensor = None, reduce: Tensor = None, new_coords: Tensor = None) -> SparseTensor:
        """
        Forward pass through the multi-scale decoder.

        Args:
            x: List of SparseTensors from encoder, ordered from shallow (first)
                to deep (last). The decoder processes these in reverse order,
                upsampling from x[-1] towards x[0].
            aux: Optional auxiliary SparseTensor to concatenate before final
                projection. Must have compatible coordinate structure with the
                decoded features.
            mask: Optional boolean tensor [N_voxels] to select a subset of
                coordinates from the final decoded features before projection.

        Returns:
            SparseTensor with decoded features of shape [N_output_voxels, out_dim].
            N_output_voxels equals the number of True values in mask if provided,
            otherwise equals the coordinate count at the shallowest resolution.

        Notes:
            - Upsampling uses transpose convolutions with skip connections.
            - Each decoder level applies two residual blocks for feature refinement.
            - Masking is applied before auxiliary feature fusion and final projection.
        """
        current = x[-1]
        for i in range(len(x)-2, -1, -1):
            current = self.upsample[i](current)
            current.F = cat([current.F, x[i].F], dim=1)
            current = self.conv[i](current)
        
        if mask is not None:
            current = SparseTensor(coords=current.C[mask], feats=current.F[mask])
            
        if aux is not None:
            # current = self._union_sparse_layers(current, aux)
            if isinstance(aux, list):
                current.F = cat([current.F] + aux, dim=1)
            else:
                current.F = cat([current.F, aux], dim=1)

        if reduce is not None and new_coords is not None:
            current = SparseTensor(coords=new_coords, feats=scatter_mean(current.F, reduce, dim=0))

        current = self.proj(current)
        if self.batch_norm is not None:
            current = self.batch_norm(current)
        if self.relu is not None:
            current = self.relu(current)

        return current
    