from torch import nn, cat
from torchsparse import SparseTensor
from torchsparse import nn as spnn


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
    
class PixelDecoder(nn.Module):
    def __init__(self, channels, latent_dim):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Sequential(
            spnn.Conv3d(channel, latent_dim, 1, bias=False),
            spnn.BatchNorm(latent_dim),
            spnn.ReLU(inplace=True),
        ) for channel in channels)
        
        self.upsample = nn.ModuleList(nn.Sequential(
            spnn.Conv3d(latent_dim, latent_dim, kernel_size=2, stride=2, transposed=True, generative=False, bias=False),
            spnn.BatchNorm(latent_dim),
            spnn.ReLU(inplace=True)
        ) for _ in channels)

        self.smooth = nn.ModuleList(nn.Sequential(
            spnn.Conv3d(latent_dim, latent_dim, 3, padding=1, bias=False),
            spnn.BatchNorm(latent_dim),
            spnn.ReLU(inplace=True)
        ) for _ in channels)

        self.last_upsample = nn.Sequential(
            spnn.Conv3d(latent_dim, latent_dim, kernel_size=2, stride=2, transposed=True, generative=False, bias=False),
            spnn.BatchNorm(latent_dim),
            spnn.ReLU(inplace=True),

            spnn.Conv3d(latent_dim, latent_dim, 3, padding=1, bias=False),
            spnn.BatchNorm(latent_dim),
            spnn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        lat = [latconv(f) for latconv, f in zip(self.lateral, x)]
        lat[-1] = self.smooth[-1](lat[-1])

        for i in range(len(lat)-1, 0, -1):
            up = self.upsample[i](lat[i])
            lat[i - 1] = lat[i - 1] + up
            lat[i - 1] = self.smooth[i - 1](lat[i - 1])

        return self.last_upsample(lat[0])
    
class PixelDecoderAlt(nn.Module):
    def __init__(self, in_channels, channels, latent_dim):
        super().__init__()
        self.upsample = nn.ModuleList(nn.Sequential(
            spnn.Conv3d(sum(channels[i:]), sum(channels[i:]), kernel_size=2, stride=2, transposed=True, generative=False, bias=False),
            spnn.BatchNorm(sum(channels[i:])),
            spnn.ReLU(inplace=True)
        ) for i in range(len(channels)))

        self.smooth = nn.Sequential(
            spnn.Conv3d(sum(channels) + in_channels, latent_dim, 3, padding=1, bias=False),
            spnn.BatchNorm(sum(channels) + in_channels),
            spnn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        result = x[-1]
        for i in range(len(x)-2, -1, -1):
            result = self.upsample[i](result)
            new_F = cat([result.F, x[i].F], dim=1)
            result = SparseTensor(coords=result.C, feats=new_F)

        return self.smooth(result)
    
class TreeProjector(nn.Module):
    def __init__(self, in_channels, num_classes, max_instances, channels = [64, 128, 256, 512], latent_dim = 512):
        super().__init__()
        self.encoder = Encoder(in_channels, channels)
        self.pixel_decoder = PixelDecoderAlt(in_channels, channels, latent_dim)
        self.semantic_head = spnn.Conv3d(latent_dim, num_classes, 1, bias=False)
        self.instance_head = spnn.Conv3d(latent_dim, max_instances, 1, bias=False)

    def forward(self, x):
        feats = self.pixel_decoder(self.encoder(x))
        return self.semantic_head(feats), self.instance_head(feats)