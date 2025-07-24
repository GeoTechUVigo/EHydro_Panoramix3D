import torch
import torch.nn.functional as F

from torch import nn
from torchsparse import nn as spnn, SparseTensor
from torch_cluster import radius_graph    # GPU, batch friendly
from torch_scatter import scatter_max


class CentroidHead(nn.Module):
    def __init__(self, latent_dim, instance_density):
        super().__init__()
        self.conv = spnn.Conv3d(latent_dim, 1, 1, bias=True)

        nn.init.constant_(self.conv.bias, bias_val = torch.log(instance_density / (1 - instance_density)))
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out')

    def _find_centroid_peaks(self, centroid_scores, tau=0.1, radius=1.1):
        logits  = centroid_scores.F.squeeze(-1)
        heat    = torch.sigmoid(logits)

        mask = heat > tau
        if mask.sum() == 0:
            return centroid_scores.C.new_zeros((0, 4))
        
        cand_h = heat[mask]
        cand_c = centroid_scores.C[mask]

        edge_index = radius_graph(
            cand_c[:, 1:].float(), radius,
            batch=cand_c[:, 0], loop=True, max_num_neighbors=27
        )

        maxi, _ = scatter_max(cand_h[edge_index[1]], edge_index[0], dim_size=cand_h.size(0))
        is_peak = cand_h >= maxi

        peaks = cand_c[is_peak]                    # (K,4)
        return peaks

    def forward(self, x):
        centroid_scores = self.conv(x)
        centroid_peaks = self._find_centroid_peaks(centroid_scores)

        return centroid_scores, SparseTensor(coords=centroid_peaks, feats=centroid_feats)
