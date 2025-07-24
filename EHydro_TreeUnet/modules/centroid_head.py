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

    @torch.no_grad()
    def _find_centroid_peaks(self, feats, centroid_scores, tau=0.1, radius=1.1):
        logits  = centroid_scores.F.squeeze(-1)
        heat    = torch.sigmoid(logits)

        mask = heat > tau
        if mask.sum() == 0:
            return centroid_scores.C.new_zeros((0, 4))
        
        cand_h = heat[mask]
        cand_c = feats.C[mask]
        cand_f = feats.F[mask]

        edge_index = radius_graph(
            cand_c[:, 1:].float(), radius,
            batch=cand_c[:, 0], loop=True, max_num_neighbors=27
        )

        maxi, _ = scatter_max(cand_h[edge_index[1]], edge_index[0], dim_size=cand_h.size(0))
        is_peak = cand_h >= maxi
        peaks = cand_c[is_peak]

        if peaks.shape[0] == 0:
            return peaks, feats.F.new_zeros((0, cand_f.size(1)))
        
        edge_pc = radius_graph(
            x=cand_c[:,1:].float(),        # puntos
            pos=peaks[:,1:].float(),       # picos (query set)
            r=r_cluster,
            x_batch=cand_c[:,0], pos_batch=peaks[:,0],
            max_num_neighbors=100
        )
        cand_idx = edge_pc[0]             # índices en cand_c / cand_f
        peak_idx = edge_pc[1]             # índices en peaks

        # --------- 4. media de features por pico ------------------
        feat_mean = scatter_mean(cand_f[cand_idx], peak_idx,
                                dim=0, dim_size=peaks.size(0))

        return peaks

    def forward(self, x):
        centroid_scores = self.conv(x)
        centroid_peaks, centroid_feats = self._find_centroid_peaks(centroid_scores)

        return centroid_scores, SparseTensor(coords=centroid_peaks, feats=centroid_feats)
