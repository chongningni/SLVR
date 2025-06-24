import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from einops import rearrange
from litsr.utils.registry import Registry

UpsamplerRegistry = Registry("upsampler")


def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


################
# Upsampler
################


@UpsamplerRegistry.register()
class OSM(nn.Module):
    def __init__(self, n_feats, overscale):
        super().__init__()
        self.body = nn.Sequential(
            wn(nn.Conv2d(n_feats, 64 * (overscale ** 2), 3, padding=1)),
            nn.PixelShuffle(overscale),
            wn(nn.Conv2d(64, 64, 3, padding=1)),
        )

    def forward(self, x, out_size):
        h = self.body(x)
        return F.interpolate(h, out_size, mode="bicubic", align_corners=False)


# @UpsamplerRegistry.register()
# class LightMLPInterpolate(nn.Module):
#     def __init__(self, n_feat, radius=3):
#         super().__init__()
#         self.radius = radius

#         self.out = nn.Sequential(
#             nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(n_feat, 3, kernel_size=1, padding=0),
#         )

#         self.weight_calc = nn.Sequential(
#             nn.Linear(3, 64),
#             nn.ReLU(True),
#             nn.Linear(64, self.radius ** 2),
#             nn.Softmax(2),
#         )

#     def forward(self, x, out_size):
#         # scale-aware fusion weight calculation
#         if type(out_size) == int:
#             out_size = [out_size, out_size]

#         _device = x.device
#         _scale = torch.tensor([x.shape[2] / out_size[0]], device=_device)

#         in_shape = x.shape[-2:]
#         in_coord = (
#             make_coord(in_shape, flatten=False)
#             .to(x.device)
#             .permute(2, 0, 1)
#             .unsqueeze(0)
#             .expand(x.shape[0], 2, *in_shape)
#         )

#         out_coord = make_coord(out_size, flatten=True).to(_device)
#         out_coord = out_coord.expand(x.shape[0], *out_coord.shape)

#         q_coord = F.grid_sample(
#             in_coord,
#             out_coord.flip(-1).unsqueeze(1),
#             mode="nearest",
#             align_corners=False,
#         )[:, :, 0, :].permute(0, 2, 1)
#         rel_coord = out_coord - q_coord
#         rel_coord[:, :, 0] *= x.shape[-2]
#         rel_coord[:, :, 1] *= x.shape[-1]

#         scale_tensor = _scale.expand(rel_coord.shape[:2]).unsqueeze(2)
#         inp = torch.cat([rel_coord, scale_tensor], dim=2)

#         weights = self.weight_calc(inp)

#         # feature
#         x_unfold = F.unfold(x, self.radius, padding=self.radius // 2)
#         x_unfold = x_unfold.view(
#             x.shape[0], x.shape[1] * (self.radius ** 2), x.shape[2], x.shape[3]
#         )

#         q_feat = F.grid_sample(
#             x_unfold,
#             out_coord.flip(-1).unsqueeze(1),
#             mode="nearest",
#             align_corners=False,
#         )[:, :, 0, :].permute(0, 2, 1)

#         # fusion
#         bs, q = q_feat.shape[:2]
#         q_feat = q_feat.view(bs, q, -1, self.radius ** 2)

#         out = (weights.unsqueeze_(2) * q_feat).sum(dim=3)

#         out = rearrange(out, "bs (h w) nf -> bs nf h w", h=out_size[0])

#         out = self.out(out)
#         return out


# @UpsamplerRegistry.register()
# class LightMLPInterpolateV2(nn.Module):
#     def __init__(self, n_feat, radius=3):
#         super().__init__()
#         self.radius = radius

#         # self.first_conv = nn.Conv2d(n_feat, 3, kernel_size=3, padding=1)

#         self.weight_calc = nn.Sequential(
#             nn.Linear(3, 64),
#             nn.ReLU(True),
#             nn.Linear(64, self.radius ** 2),
#             nn.Softmax(2),
#         )

#     def forward(self, x, out_size):
#         # scale-aware fusion weight calculation
#         if type(out_size) == int:
#             out_size = [out_size, out_size]

#         _device = x.device
#         _scale = torch.tensor([x.shape[2] / out_size[0]], device=_device)

#         in_shape = x.shape[-2:]
#         in_coord = (
#             make_coord(in_shape, flatten=False)
#             .to(x.device)
#             .permute(2, 0, 1)
#             .unsqueeze(0)
#             .expand(x.shape[0], 2, *in_shape)
#         )

#         out_coord = make_coord(out_size, flatten=True).to(_device)
#         out_coord = out_coord.expand(x.shape[0], *out_coord.shape)

#         q_coord = F.grid_sample(
#             in_coord,
#             out_coord.flip(-1).unsqueeze(1),
#             mode="nearest",
#             align_corners=False,
#         )[:, :, 0, :].permute(0, 2, 1)
#         rel_coord = out_coord - q_coord
#         rel_coord[:, :, 0] *= x.shape[-2]
#         rel_coord[:, :, 1] *= x.shape[-1]

#         scale_tensor = _scale.expand(rel_coord.shape[:2]).unsqueeze(2)
#         inp = torch.cat([rel_coord, scale_tensor], dim=2)

#         weights = self.weight_calc(inp)

#         # feature
#         x = self.first_conv(x)
#         x_unfold = F.unfold(x, self.radius, padding=self.radius // 2)
#         x_unfold = x_unfold.view(
#             x.shape[0], x.shape[1] * (self.radius ** 2), x.shape[2], x.shape[3]
#         )

#         q_feat = F.grid_sample(
#             x_unfold,
#             out_coord.flip(-1).unsqueeze(1),
#             mode="nearest",
#             align_corners=False,
#         )[:, :, 0, :].permute(0, 2, 1)

#         # fusion
#         bs, q = q_feat.shape[:2]
#         q_feat = q_feat.view(bs, q, -1, self.radius ** 2)

#         out = (weights.unsqueeze(2) * q_feat).sum(dim=3)

#         out = rearrange(out, "bs (h w) nf -> bs nf h w", h=out_size[0])

#         return out


# @UpsamplerRegistry.register()
# class Multiscaleupsample(nn.Module):
#     def __init__(self, n_feat, split=4):
#         super().__init__()

#         self.distilled_channels = n_feat // split

#         self.out = nn.Sequential(
#             nn.Conv2d(n_feat, n_feat, 3, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(n_feat, 3, kernel_size=1, padding=0),
#         )

#         up = []
#         up.append(nn.Conv2d(n_feat // split * 3, (n_feat // split) * 4 * 3, 3, 1, 1))
#         up.append(nn.PixelShuffle(2))
#         self.upsample = nn.Sequential(*up)

#         up1 = []
#         up1.append(nn.Conv2d(n_feat // split * 2, (n_feat // split) * 4 * 2, 3, 1, 1))
#         up1.append(nn.PixelShuffle(2))
#         self.upsample1 = nn.Sequential(*up1)

#         up2 = []
#         up2.append(nn.Conv2d(n_feat // split, (n_feat // split) * 4, 3, 1, 1))
#         up2.append(nn.PixelShuffle(2))
#         self.upsample2 = nn.Sequential(*up2)

#     def forward(self, x, out_size):

#         out1, remaining_c1 = torch.split(
#             x, (self.distilled_channels, self.distilled_channels * 3), dim=1
#         )
#         out = self.upsample(remaining_c1)

#         out2, remaining_c2 = torch.split(
#             out, (self.distilled_channels, self.distilled_channels * 2), dim=1
#         )
#         out = self.upsample1(remaining_c2)

#         out3, remaining_c3 = torch.split(
#             out, (self.distilled_channels, self.distilled_channels), dim=1
#         )
#         out = self.upsample2(remaining_c3)

#         distilled_c1 = F.interpolate(
#             out1, out_size, mode="bicubic", align_corners=False
#         )
#         distilled_c2 = F.interpolate(
#             out2, out_size, mode="bicubic", align_corners=False
#         )
#         distilled_c3 = F.interpolate(
#             out3, out_size, mode="bicubic", align_corners=False
#         )
#         distilled_c4 = F.interpolate(out, out_size, mode="bicubic", align_corners=False)

#         out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4], dim=1)

#         out = self.out(out)

#         return out


# @UpsamplerRegistry.register()
# class Multiscaleupsamplev2(nn.Module):
#     def __init__(self, n_feat, split=4):
#         super().__init__()

#         self.distilled_channels = n_feat // split

#         self.out = nn.Sequential(
#             nn.Conv2d(12, 3, kernel_size=1, padding=0),
#         )

#         self.first = nn.Conv2d(n_feat // split, 3, 3, 1, 1)

#         up = []
#         up.append(nn.Conv2d(n_feat // split, 3 * 4, 3, 1, 1))
#         up.append(nn.PixelShuffle(2))
#         self.upsample = nn.Sequential(*up)

#         up1 = []
#         up1.append(nn.Conv2d(n_feat // split, (n_feat // split) * 4, 3, 1, 1))
#         up1.append(nn.PixelShuffle(2))
#         self.upsample1 = nn.Sequential(*up1)

#         up2 = []
#         up2.append(nn.Conv2d(n_feat // split, (n_feat // split) * 4, 3, 1, 1))
#         up2.append(nn.PixelShuffle(2))
#         self.upsample2 = nn.Sequential(*up2)

#     def forward(self, x, out_size):

#         distilled_c1, distilled_c2, distilled_c3, distilled_c4 = torch.split(
#             x, self.distilled_channels, dim=1
#         )
#         distilled_c2 = self.upsample(distilled_c2)
#         distilled_c3 = self.upsample(self.upsample1(distilled_c3))
#         distilled_c4 = self.upsample(self.upsample1(self.upsample2(distilled_c4)))

#         distilled_c1 = self.first(distilled_c1)
#         distilled_c1 = F.interpolate(
#             distilled_c1, out_size, mode="bicubic", align_corners=False
#         )
#         distilled_c2 = F.interpolate(
#             distilled_c2, out_size, mode="bicubic", align_corners=False
#         )
#         distilled_c3 = F.interpolate(
#             distilled_c3, out_size, mode="bicubic", align_corners=False
#         )
#         distilled_c4 = F.interpolate(
#             distilled_c4, out_size, mode="bicubic", align_corners=False
#         )

#         out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4], dim=1)

#         out = self.out(out)

#         return out


# @UpsamplerRegistry.register()
# class Multiscaleupsamplev3(nn.Module):
#     def __init__(self, n_feat, split=4):
#         super().__init__()

#         self.distilled_channels = n_feat // split

#         self.out = nn.Sequential(
#             nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(n_feat, 3, kernel_size=1, padding=0),
#         )

#         up = []
#         up.append(nn.Conv2d(n_feat // split, (n_feat // split) * 4, 3, 1, 1))
#         up.append(nn.PixelShuffle(2))
#         self.upsample = nn.Sequential(*up)

#         up1 = []
#         up1.append(nn.Conv2d(n_feat // split, (n_feat // split) * 4, 3, 1, 1))
#         up1.append(nn.PixelShuffle(2))
#         self.upsample1 = nn.Sequential(*up1)

#         up2 = []
#         up2.append(nn.Conv2d(n_feat // split, (n_feat // split) * 4, 3, 1, 1))
#         up2.append(nn.PixelShuffle(2))
#         self.upsample2 = nn.Sequential(*up2)

#     def forward(self, x, out_size):

#         distilled_c1, distilled_c2, distilled_c3, distilled_c4 = torch.split(
#             x, self.distilled_channels, dim=1
#         )
#         distilled_c2 = self.upsample(distilled_c2)
#         distilled_c3 = self.upsample1(self.upsample(distilled_c3))
#         distilled_c4 = self.upsample2(self.upsample1(self.upsample(distilled_c4)))

#         distilled_c1 = F.interpolate(
#             distilled_c1, out_size, mode="bicubic", align_corners=False
#         )
#         distilled_c2 = F.interpolate(
#             distilled_c2, out_size, mode="bicubic", align_corners=False
#         )
#         distilled_c3 = F.interpolate(
#             distilled_c3, out_size, mode="bicubic", align_corners=False
#         )
#         distilled_c4 = F.interpolate(
#             distilled_c4, out_size, mode="bicubic", align_corners=False
#         )

#         out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4], dim=1)

#         out = self.out(out)

#         return out


class PA(nn.Module):
    """PA is pixel attention"""

    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


# @UpsamplerRegistry.register()
# class Multiscaleupsamplev4(nn.Module):
#     # good!
#     def __init__(self, n_feat, split=4):
#         super().__init__()

#         self.distilled_channels = n_feat // split

#         self.out = nn.Sequential(
#             nn.Conv2d(n_feat, n_feat // 4, 3, padding=1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(n_feat // 4, 3, 3, padding=1),
#         )

#         up = []
#         up.append(
#             nn.Conv2d(n_feat // split * 3, (n_feat // split) * 4 * 3, 3, 1, 1, groups=3)
#         )
#         up.append(nn.PixelShuffle(2))
#         self.upsample = nn.Sequential(*up)

#         up1 = []
#         up1.append(
#             nn.Conv2d(n_feat // split * 2, (n_feat // split) * 4 * 2, 3, 1, 1, groups=2)
#         )
#         up1.append(nn.PixelShuffle(2))
#         self.upsample1 = nn.Sequential(*up1)

#         up2 = []
#         up2.append(nn.Conv2d(n_feat // split, (n_feat // split) * 4, 3, 1, 1))
#         up2.append(nn.PixelShuffle(2))
#         self.upsample2 = nn.Sequential(*up2)

#         self.pixel_attn = PA(n_feat)

#     def forward(self, x, out_size):

#         out1, remaining_c1 = torch.split(
#             x, (self.distilled_channels, self.distilled_channels * 3), dim=1
#         )
#         out = self.upsample(remaining_c1)

#         out2, remaining_c2 = torch.split(
#             out, (self.distilled_channels, self.distilled_channels * 2), dim=1
#         )
#         out = self.upsample1(remaining_c2)

#         out3, remaining_c3 = torch.split(
#             out, (self.distilled_channels, self.distilled_channels), dim=1
#         )
#         out = self.upsample2(remaining_c3)

#         distilled_c1 = F.interpolate(
#             out1, out_size, mode="bilinear", align_corners=False
#         )
#         distilled_c2 = F.interpolate(
#             out2, out_size, mode="bilinear", align_corners=False
#         )
#         distilled_c3 = F.interpolate(
#             out3, out_size, mode="bilinear", align_corners=False
#         )
#         distilled_c4 = F.interpolate(
#             out, out_size, mode="bilinear", align_corners=False
#         )

#         out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4], dim=1)
#         out = self.pixel_attn(out)
#         out = self.out(out)

#         return out


class ScaleEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        half_dim = self.dim // 2
        self.inv_freq = torch.exp(
            torch.arange(half_dim, dtype=torch.float32)
            * (-math.log(10000) / (half_dim - 1))
        )

    def forward(self, input):
        shape = input.shape
        input = input.view(-1).to(torch.float32)
        sinusoid_in = torch.ger(input, self.inv_freq.to(input.device))
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)

        return pos_emb


class SAPA(nn.Module):
    """Scale aware pixel attention"""

    def __init__(self, nf):
        super().__init__()

        self.scale_embing = ScaleEmbedding(nf)

        self.conv = nn.Sequential(
            nn.Conv2d(nf * 2, nf // 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf // 2, nf, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):

        scale_emb = self.scale_embing(scale)
        scale_emb = (
            scale_emb.unsqueeze_(2)
            .unsqueeze_(3)
            .expand([x.shape[0], scale_emb.shape[1], x.shape[2], x.shape[3]])
        )

        y = torch.cat([x, scale_emb], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


@UpsamplerRegistry.register()
class Multiscaleupsamplev5(nn.Module):
    def __init__(self, n_feat, split=4):
        # final
        super().__init__()

        self.distilled_channels = n_feat // split
        self.out = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 4, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feat // 4, 3, 3, padding=1),
        )

        up = []
        up.append(
            nn.Conv2d(n_feat // split * 3, (n_feat // split) * 4 * 3, 3, 1, 1, groups=3)
        )
        up.append(nn.PixelShuffle(2))
        self.upsample = nn.Sequential(*up)

        up1 = []
        up1.append(
            nn.Conv2d(n_feat // split * 2, (n_feat // split) * 4 * 2, 3, 1, 1, groups=2)
        )
        up1.append(nn.PixelShuffle(2))
        self.upsample1 = nn.Sequential(*up1)

        up2 = []
        up2.append(nn.Conv2d(n_feat // split, (n_feat // split) * 4, 3, 1, 1))
        up2.append(nn.PixelShuffle(2))
        self.upsample2 = nn.Sequential(*up2)

        self.SAPA = SAPA(n_feat)

    def forward(self, x, out_size):
        scale = torch.tensor([x.shape[2] / out_size[0]], device=x.device)

        out1, remaining_c1 = torch.split(
            x, (self.distilled_channels, self.distilled_channels * 3), dim=1
        )
        out = self.upsample(remaining_c1)

        out2, remaining_c2 = torch.split(
            out, (self.distilled_channels, self.distilled_channels * 2), dim=1
        )
        out = self.upsample1(remaining_c2)

        out3, remaining_c3 = torch.split(
            out, (self.distilled_channels, self.distilled_channels), dim=1
        )
        out = self.upsample2(remaining_c3)

        distilled_c1 = F.interpolate(
            out1, out_size, mode="bilinear", align_corners=False
        )
        distilled_c2 = F.interpolate(
            out2, out_size, mode="bilinear", align_corners=False
        )
        distilled_c3 = F.interpolate(
            out3, out_size, mode="bilinear", align_corners=False
        )
        distilled_c4 = F.interpolate(
            out, out_size, mode="bilinear", align_corners=False
        )

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4], dim=1)

        out = self.out(self.SAPA(out, scale))
        return out


@UpsamplerRegistry.register()
class Multiscaleupsamplev5woSAPA(nn.Module):
    # final
    def __init__(self, n_feat, split=4):
        super().__init__()

        self.distilled_channels = n_feat // split
        self.out = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 4, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feat // 4, 3, 3, padding=1),
        )

        up = []
        up.append(
            nn.Conv2d(n_feat // split * 3, (n_feat // split) * 4 * 3, 3, 1, 1, groups=3)
        )
        up.append(nn.PixelShuffle(2))
        self.upsample = nn.Sequential(*up)

        up1 = []
        up1.append(
            nn.Conv2d(n_feat // split * 2, (n_feat // split) * 4 * 2, 3, 1, 1, groups=2)
        )
        up1.append(nn.PixelShuffle(2))
        self.upsample1 = nn.Sequential(*up1)

        up2 = []
        up2.append(nn.Conv2d(n_feat // split, (n_feat // split) * 4, 3, 1, 1))
        up2.append(nn.PixelShuffle(2))
        self.upsample2 = nn.Sequential(*up2)

    def forward(self, x, out_size):

        out1, remaining_c1 = torch.split(
            x, (self.distilled_channels, self.distilled_channels * 3), dim=1
        )
        out = self.upsample(remaining_c1)

        out2, remaining_c2 = torch.split(
            out, (self.distilled_channels, self.distilled_channels * 2), dim=1
        )
        out = self.upsample1(remaining_c2)

        out3, remaining_c3 = torch.split(
            out, (self.distilled_channels, self.distilled_channels), dim=1
        )
        out = self.upsample2(remaining_c3)

        distilled_c1 = F.interpolate(
            out1, out_size, mode="bilinear", align_corners=False
        )
        distilled_c2 = F.interpolate(
            out2, out_size, mode="bilinear", align_corners=False
        )
        distilled_c3 = F.interpolate(
            out3, out_size, mode="bilinear", align_corners=False
        )
        distilled_c4 = F.interpolate(
            out, out_size, mode="bilinear", align_corners=False
        )

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4], dim=1)
        return self.out(out)


# @UpsamplerRegistry.register()
# class Multiscaleupsamplev6(nn.Module):
#     # new!
#     def __init__(self, n_feat, split=4):
#         super().__init__()

#         self.distilled_channels = n_feat // split

#         self.out = nn.Sequential(
#             nn.Conv2d(n_feat, n_feat // 4, 3, padding=1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(n_feat // 4, 3, 3, padding=1),
#         )

#         up = []
#         up.append(
#             nn.Conv2d(n_feat // split * 3, (n_feat // split) * 4 * 3, 3, 1, 1, groups=3)
#         )
#         up.append(nn.PixelShuffle(2))
#         self.upsample = nn.Sequential(*up)

#         up1 = []
#         up1.append(
#             nn.Conv2d(n_feat // split * 2, (n_feat // split) * 4 * 2, 3, 1, 1, groups=2)
#         )
#         up1.append(nn.PixelShuffle(2))
#         self.upsample1 = nn.Sequential(*up1)

#         up2 = []
#         up2.append(nn.Conv2d(n_feat // split, (n_feat // split) * 4, 3, 1, 1))
#         up2.append(nn.PixelShuffle(2))
#         self.upsample2 = nn.Sequential(*up2)

#         self.scale_encoding = nn.Sequential(
#             ScaleEmbedding(n_feat),
#             nn.Linear(n_feat, n_feat),
#             nn.LeakyReLU(0.2, True),
#             nn.Linear(n_feat, n_feat),
#             # nn.Sigmoid(),
#         )
#         # self.scale_enc_conv = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, padding=0)

#     def forward(self, x, out_size):

#         scale = torch.tensor([x.shape[2] / out_size[0]], device=x.device)

#         out1, remaining_c1 = torch.split(
#             x, (self.distilled_channels, self.distilled_channels * 3), dim=1
#         )
#         out = self.upsample(remaining_c1)

#         out2, remaining_c2 = torch.split(
#             out, (self.distilled_channels, self.distilled_channels * 2), dim=1
#         )
#         out = self.upsample1(remaining_c2)

#         out3, remaining_c3 = torch.split(
#             out, (self.distilled_channels, self.distilled_channels), dim=1
#         )
#         out = self.upsample2(remaining_c3)

#         distilled_c1 = F.interpolate(
#             out1, out_size, mode="bilinear", align_corners=False
#         )
#         distilled_c2 = F.interpolate(
#             out2, out_size, mode="bilinear", align_corners=False
#         )
#         distilled_c3 = F.interpolate(
#             out3, out_size, mode="bilinear", align_corners=False
#         )
#         distilled_c4 = F.interpolate(
#             out, out_size, mode="bilinear", align_corners=False
#         )

#         scale_enc = self.scale_encoding(scale)
#         # scale_enc = (
#         #     scale_enc.unsqueeze_(2)
#         #     .unsqueeze_(3)
#         #     .expand([out.shape[0], scale_enc.shape[1], out_size[0], out_size[1]])
#         # )

#         scale_enc = (
#             scale_enc.unsqueeze_(2)
#             .unsqueeze_(3)
#             .expand([out.shape[0], scale_enc.shape[1], 1, 1])
#         )

#         out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4], dim=1)
#         out = out * scale_enc
#         out = self.out(out)

#         return out


# @UpsamplerRegistry.register()
# class Multiscaleupsamplev5(nn.Module):
#     def __init__(self, nf, unf=24):
#         super().__init__()

#         self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
#         self.att1 = PA(unf)
#         self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

#         self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
#         self.att2 = PA(unf)
#         self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

#         self.upconv3 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
#         self.att3 = PA(unf)
#         self.HRconv3 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

#         self.compress = nn.Conv2d(unf * 3, unf, 1, 1, 0, bias=True)
#         self.conv_last = nn.Conv2d(unf, 3, 3, 1, 1, bias=True)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#         self.pixel_attn = PA(unf * 3)

#     def forward(self, x, out_size):

#         fea = self.upconv1(x)
#         fea = self.lrelu(self.att1(fea))
#         fea = self.lrelu(self.HRconv1(fea))
#         c1 = F.interpolate(fea, size=out_size, mode="bilinear")

#         fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest"))
#         fea = self.lrelu(self.att2(fea))
#         fea = self.lrelu(self.HRconv2(fea))
#         c2 = F.interpolate(fea, size=out_size, mode="bilinear")

#         fea = self.upconv3(F.interpolate(fea, scale_factor=2, mode="nearest"))
#         fea = self.lrelu(self.att3(fea))
#         fea = self.lrelu(self.HRconv3(fea))
#         c3 = F.interpolate(fea, size=out_size, mode="bilinear")

#         out = torch.cat([c1, c2, c3], dim=1)
#         out = self.pixel_attn(out)

#         out = self.lrelu(self.compress(out))
#         out = self.conv_last(out)

#         return out


@UpsamplerRegistry.register()
class PAN_upsampler(nn.Module):
    def __init__(self, nf, unf, scale, out_nc=3):
        super().__init__()
        self.scale = scale
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, fea):
        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(
                F.interpolate(fea, scale_factor=self.scale, mode="nearest")
            )
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest"))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest"))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))

        out = self.conv_last(fea)
        return out


@UpsamplerRegistry.register()
class BicUpsampler(nn.Module):
    def __init__(self, nf, out_nc=3):
        super().__init__()

        self.out = nn.Sequential(
            nn.Conv2d(nf, nf // 4, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf // 4, 3, 3, padding=1),
        )

    def forward(self, fea, out_size):
        fea = F.interpolate(fea, size=out_size, mode="bilinear", align_corners=False)

        out = self.out(fea)
        return out


class CSUM_WO_WN(nn.Module):
    # Up-sampling net
    def __init__(self, n_feats, kSize, out_channels, interpolate_mode, levels=4):
        super().__init__()
        self.interpolate_mode = interpolate_mode
        self.levels = levels

        self.UPNet_x2_list = []

        for _ in range(levels - 1):
            self.UPNet_x2_list.append(
                nn.Sequential(
                    *[
                        nn.Conv2d(
                            n_feats,
                            n_feats * 4,
                            kSize,
                            padding=(kSize - 1) // 2,
                            stride=1,
                        ),
                        nn.PixelShuffle(2),
                    ]
                )
            )

        self.scale_aware_layer = nn.Sequential(
            *[nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, levels), nn.Sigmoid()]
        )

        self.UPNet_x2_list = nn.Sequential(*self.UPNet_x2_list)

        self.fuse = nn.Sequential(
            *[
                nn.Conv2d(n_feats * levels, 256, kernel_size=1, padding=0, stride=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1),
                nn.ReLU(),
                nn.Conv2d(256, out_channels, kernel_size=1, padding=0, stride=1),
            ]
        )

    def forward(self, x, out_size):

        if type(out_size) == int:
            out_size = [out_size, out_size]

        if type(x) == list:
            return self.forward_list(x, out_size)

        r = torch.tensor([x.shape[2] / out_size[0]], device=x.device)

        scale_w = self.scale_aware_layer(r.unsqueeze(0))[0]

        # scale_in = x.new_tensor(np.ones([x.shape[0], 1, out_size[0], out_size[1]])*r)

        x_list = [x]
        for l in range(1, self.levels):
            x_list.append(self.UPNet_x2_list[l - 1](x_list[l - 1]))

        x_resize_list = []
        for l in range(self.levels):
            x_resize = F.interpolate(
                x_list[l], out_size, mode=self.interpolate_mode, align_corners=False
            )
            x_resize *= scale_w[l]
            x_resize_list.append(x_resize)

        # x_resize_list.append(scale_in)
        out = self.fuse(torch.cat(tuple(x_resize_list), 1))
        return out

    def forward_list(self, h_list, out_size):
        assert (
            len(h_list) == self.levels
        ), "The Length of input list must equal to the number of levels"
        device = h_list[0].device
        r = torch.tensor([h_list[0].shape[2] / out_size[0]], device=device)
        scale_w = self.scale_aware_layer(r.unsqueeze(0))[0]

        x_resize_list = []
        for l in range(self.levels):
            h = h_list[l]
            for i in range(l):
                h = self.UPNet_x2_list[i](h)
            x_resize = F.interpolate(
                h, out_size, mode=self.interpolate_mode, align_corners=False
            )
            x_resize *= scale_w[l]
            x_resize_list.append(x_resize)

        out = self.fuse(torch.cat(tuple(x_resize_list), 1))
        return out
