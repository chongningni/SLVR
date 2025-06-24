import torch
import torch.nn as nn
import torch.nn.functional as F
from litsr.utils.registry import ArchRegistry
from litsr.archs.upsampler import CSUM

import archs.ops as ops

from .common import (
    # BasicBlock,
    # Block_n,
    CARN_Block,
    # MeanShift,
    # MeanShift_n,
    # UpsampleBlock,
    UpSampler_n,
)
from .upsampler import Multiscaleupsamplev5, Multiscaleupsamplev5woSAPA, CSUM_WO_WN


@ArchRegistry.register()
class CARNNet(nn.Module):
    def __init__(self, **kwargs):
        super(CARNNet, self).__init__()
        
        self.scale = kwargs.get("scale")
        multi_scale = kwargs.get("multi_scale")
        group = kwargs.get("group", 1)

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = ops.Block(64, 64)
        self.b2 = ops.Block(64, 64)
        self.b3 = ops.Block(64, 64)
        self.c1 = ops.BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64*4, 64, 1, 1, 0)
        
        self.upsample = ops.UpsampleBlock(64, scale=self.scale, 
                                          multi_scale=multi_scale,
                                          group=group)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)
                
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=self.scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out

@ArchRegistry.register()
class CARN(nn.Module):
    def __init__(
        self, upscale_factor, in_channels, num_fea, out_channels, use_skip=False
    ):
        super(CARN, self).__init__()
        self.use_skip = use_skip
        self.upscale_factor = upscale_factor
        # self.sub_mean = MeanShift()
        # self.add_mean = MeanShift(sign=1)

        # extract features
        self.fea_in = nn.Conv2d(in_channels, num_fea, 3, 1, 1)

        # CARN body
        self.b1 = CARN_Block(num_fea)
        self.c1 = nn.Sequential(nn.Conv2d(num_fea * 2, num_fea, 1, 1, 0), nn.ReLU(True))

        self.b2 = CARN_Block(num_fea)
        self.c2 = nn.Sequential(nn.Conv2d(num_fea * 3, num_fea, 1, 1, 0), nn.ReLU(True))

        self.b3 = CARN_Block(num_fea)
        self.c3 = nn.Sequential(nn.Conv2d(num_fea * 4, num_fea, 1, 1, 0), nn.ReLU(True))

        # Reconstruct
        self.upsampler = UpSampler_n(upscale_factor, num_fea)
        self.last_conv = nn.Conv2d(num_fea, out_channels, 3, 1, 1)

    def forward(self, x):
        # x = self.sub_mean(x)

        # feature extraction
        x = self.fea_in(x)
        if self.use_skip:
            inter_res = F.interpolate(
                x, scale_factor=self.upscale_factor, mode="bicubic", align_corners=False
            )

        # body
        b1 = self.b1(x)
        c1 = torch.cat([b1, x], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        # Reconstruct
        out = self.upsampler(o3)
        if self.use_skip:
            out += inter_res
        out = self.last_conv(out)

        # out = self.add_mean(out)

        return out


@ArchRegistry.register()
class CARNMSOursNet(nn.Module):
    def __init__(
        self, upscale_factor, in_channels, num_fea, out_channels, use_skip=False
    ):
        super().__init__()
        self.use_skip = use_skip
        self.upscale_factor = upscale_factor
        # self.sub_mean = MeanShift()
        # self.add_mean = MeanShift(sign=1)

        # extract features
        self.fea_in = nn.Conv2d(in_channels, num_fea, 3, 1, 1)

        # CARN body
        self.b1 = CARN_Block(num_fea)
        self.c1 = nn.Sequential(nn.Conv2d(num_fea * 2, num_fea, 1, 1, 0), nn.ReLU(True))

        self.b2 = CARN_Block(num_fea)
        self.c2 = nn.Sequential(nn.Conv2d(num_fea * 3, num_fea, 1, 1, 0), nn.ReLU(True))

        self.b3 = CARN_Block(num_fea)
        self.c3 = nn.Sequential(nn.Conv2d(num_fea * 4, num_fea, 1, 1, 0), nn.ReLU(True))

        # Reconstruct
        # self.upsampler = LightMLPInterpolate(num_fea, radius=3)
        # self.last_conv = nn.Conv2d(num_fea, out_channels, 3, 1, 1)
        self.upsampler = Multiscaleupsamplev5(num_fea, split=4)

    def forward(self, x, outsize):
        # x = self.sub_mean(x)

        # feature extraction
        x = self.fea_in(x)
        if self.use_skip:
            inter_res = F.interpolate(
                x, scale_factor=self.upscale_factor, mode="bicubic", align_corners=False
            )

        # body
        b1 = self.b1(x)
        c1 = torch.cat([b1, x], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        # Reconstruct
        out = self.upsampler(o3, outsize)
        if self.use_skip:
            out += inter_res
        # out = self.last_conv(out)

        # out = self.add_mean(out)

        return out


@ArchRegistry.register()
class CARNCSUMNet(nn.Module):
    def __init__(
        self, upscale_factor, in_channels, num_fea, out_channels, use_skip=False
    ):
        super().__init__()
        self.use_skip = use_skip
        self.upscale_factor = upscale_factor
        # self.sub_mean = MeanShift()
        # self.add_mean = MeanShift(sign=1)

        # extract features
        self.fea_in = nn.Conv2d(in_channels, num_fea, 3, 1, 1)

        # CARN body
        self.b1 = CARN_Block(num_fea)
        self.c1 = nn.Sequential(nn.Conv2d(num_fea * 2, num_fea, 1, 1, 0), nn.ReLU(True))

        self.b2 = CARN_Block(num_fea)
        self.c2 = nn.Sequential(nn.Conv2d(num_fea * 3, num_fea, 1, 1, 0), nn.ReLU(True))

        self.b3 = CARN_Block(num_fea)
        self.c3 = nn.Sequential(nn.Conv2d(num_fea * 4, num_fea, 1, 1, 0), nn.ReLU(True))

        # Reconstruct
        self.upsampler = CSUM(
            num_fea,
            kSize=3,
            out_channels=out_channels,
            interpolate_mode="bilinear",
            levels=4,
        )

    def forward(self, x, outsize):
        # x = self.sub_mean(x)

        # feature extraction
        x = self.fea_in(x)
        if self.use_skip:
            inter_res = F.interpolate(
                x, scale_factor=self.upscale_factor, mode="bicubic", align_corners=False
            )

        # body
        b1 = self.b1(x)
        c1 = torch.cat([b1, x], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        # Reconstruct
        out = self.upsampler(o3, outsize)
        if self.use_skip:
            out += inter_res
        # out = self.last_conv(out)

        # out = self.add_mean(out)

        return out


@ArchRegistry.register()
class CARNMSOsmNet(nn.Module):
    def __init__(
        self, upscale_factor, in_channels, num_fea, out_channels, use_skip=False
    ):
        super().__init__()
        self.use_skip = use_skip
        self.upscale_factor = upscale_factor
        # self.sub_mean = MeanShift()
        # self.add_mean = MeanShift(sign=1)

        # extract features
        self.fea_in = nn.Conv2d(in_channels, num_fea, 3, 1, 1)

        # CARN body
        self.b1 = CARN_Block(num_fea)
        self.c1 = nn.Sequential(nn.Conv2d(num_fea * 2, num_fea, 1, 1, 0), nn.ReLU(True))

        self.b2 = CARN_Block(num_fea)
        self.c2 = nn.Sequential(nn.Conv2d(num_fea * 3, num_fea, 1, 1, 0), nn.ReLU(True))

        self.b3 = CARN_Block(num_fea)
        self.c3 = nn.Sequential(nn.Conv2d(num_fea * 4, num_fea, 1, 1, 0), nn.ReLU(True))

        # Reconstruct
        # self.upsampler = LightMLPInterpolate(num_fea, radius=3)
        # self.last_conv = nn.Conv2d(num_fea, out_channels, 3, 1, 1)
        # self.upsampler = Multiscaleupsamplev4(num_fea, split=4)
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.upsampler = nn.Sequential(
            wn(nn.Conv2d(num_fea, 64 * 25, 3, padding=1)),
            nn.PixelShuffle(5),
            wn(nn.Conv2d(64, 3, 3, padding=1)),
        )

    def forward(self, x, outsize):
        # x = self.sub_mean(x)

        # feature extraction
        x = self.fea_in(x)
        if self.use_skip:
            inter_res = F.interpolate(
                x, scale_factor=self.upscale_factor, mode="bicubic", align_corners=False
            )

        # body
        b1 = self.b1(x)
        c1 = torch.cat([b1, x], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        # Reconstruct
        out = F.interpolate(
            self.upsampler(o3), outsize, mode="bicubic", align_corners=False
        )
        if self.use_skip:
            out += inter_res
        # out = self.last_conv(out)

        # out = self.add_mean(out)

        return out
