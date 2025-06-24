import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.nn.init as init
from collections import OrderedDict

from os.path import exists
import os
from matplotlib import pyplot as plt
import time

################
# Basic blocks
################


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class WeightNormedConv(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        act=nn.ReLU(True),
        res_scale=1.0,
    ):
        conv = weight_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                bias=bias,
            )
        )
        # init.constant_(conv.weight_g, res_scale)
        # init.zeros_(conv.bias)
        m = [conv]
        if act:
            m.append(act)
        super().__init__(*m)


class BasicBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False,
        bn=True,
        act=nn.ReLU(True),
    ):

        m = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=(kernel_size // 2),
                stride=stride,
                bias=bias,
            )
        ]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def mean_channels(F):
    assert F.dim() == 4
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert F.dim() == 4
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (
        F.size(2) * F.size(3)
    )
    return F_variance.pow(0.5)


def conv_layer(
    in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True
):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=padding,
        bias=bias,
        dilation=dilation,
        groups=groups,
    )


def conv_layer_wn(
    in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True
):
    padding = int((kernel_size - 1) / 2) * dilation
    return weight_norm(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            bias=bias,
            dilation=dilation,
            groups=groups,
        )
    )


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "lrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            "activation layer [{:s}] is not found".format(act_type)
        )
    return layer


# contrast-aware channel attention module
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == "reflect":
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == "replicate":
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError(
            "padding layer [{:s}] is not implemented".format(pad_type)
        )
    return layer


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == "batch":
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == "instance":
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError(
            "normalization layer [{:s}] is not found".format(norm_type)
        )
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(
    in_nc,
    out_nc,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != "zero" else None
    padding = padding if pad_type == "zero" else 0

    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def conv_block_wn(
    in_nc,
    out_nc,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != "zero" else None
    padding = padding if pad_type == "zero" else 0

    c = weight_norm(
        nn.Conv2d(
            in_nc,
            out_nc,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )
    )
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation("lrelu", neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.cca = CCALayer(self.distilled_channels * 4)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(
            out_c1, (self.distilled_channels, self.remaining_channels), dim=1
        )
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(
            out_c2, (self.distilled_channels, self.remaining_channels), dim=1
        )
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(
            out_c3, (self.distilled_channels, self.remaining_channels), dim=1
        )
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused


class IMDModulev2(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModulev2, self).__init__()

        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation("lrelu", neg_slope=0.05)
        # self.c5 = conv_layer(4, in_channels, 1)
        # self.cca = CCALayer(self.distilled_channels * 4)
        self.attention = HFCAttention(in_channels)  

    def forward(self, input, x_edge):

        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(
            out_c1, (self.distilled_channels, self.remaining_channels), dim=1
        )
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(
            out_c2, (self.distilled_channels, self.remaining_channels), dim=1
        )
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(
            out_c3, (self.distilled_channels, self.remaining_channels), dim=1
        )
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        y = self.attention(x_edge)
        out_fused = torch.mul(out, y) 

        return out_fused


def pixelshuffle_block(
    in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1
):
    conv = conv_layer(
        in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class CARN_Block(nn.Module):
    def __init__(self, num_fea):
        super(CARN_Block, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
        )
        self.c1 = nn.Sequential(nn.Conv2d(num_fea * 2, num_fea, 1, 1, 0), nn.ReLU(True))

        self.b2 = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
        )
        self.c2 = nn.Sequential(nn.Conv2d(num_fea * 3, num_fea, 1, 1, 0), nn.ReLU(True))

        self.b3 = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
        )
        self.c3 = nn.Sequential(nn.Conv2d(num_fea * 4, num_fea, 1, 1, 0), nn.ReLU(True))

        self.act = nn.ReLU(True)

    def forward(self, x):
        b1 = self.act(self.b1(x) + x)
        c1 = torch.cat([x, b1], dim=1)  # num_fea * 2
        o1 = self.c1(c1)

        b2 = self.act(self.b2(o1) + o1)
        c2 = torch.cat([c1, b2], dim=1)  # num_fea * 3
        o2 = self.c2(c2)

        b3 = self.act(self.b3(o2) + o2)
        c3 = torch.cat([c2, b3], dim=1)  # num_fea * 4
        o3 = self.c3(c3)

        return o3


class MeanShift(nn.Conv2d):
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0], sign=-1):
        super(MeanShift, self).__init__(3, 3, 1)
        std = torch.Tensor(std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * torch.Tensor(mean)
        self.bias.data.div_(std)
        for p in self.parameters():
            p.requires_grad = False


class UpSampler_n(nn.Module):
    def __init__(self, upscale_factor=2, num_fea=64):
        super(UpSampler_n, self).__init__()
        if (upscale_factor & (upscale_factor - 1)) == 0:  # upscale_factor = 2^n
            m = []
            for i in range(int(math.log(upscale_factor, 2))):
                m.append(nn.Conv2d(num_fea, num_fea * 4, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
            self.upsample = nn.Sequential(*m)

        elif upscale_factor == 3:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_fea, num_fea * 9, 3, 1, 1), nn.PixelShuffle(3)
            )
        else:
            raise NotImplementedError("Error upscale_factor in Upsampler")

    def forward(self, x):
        return self.upsample(x)


def init_weights(modules):
    pass


class Block_n(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block_n, self).__init__()

        self.b1 = ResidualBlock(64, 64)
        self.b2 = ResidualBlock(64, 64)
        self.b3 = ResidualBlock(64, 64)
        self.c1 = BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = BasicBlock(64 * 4, 64, 1, 1, 0)

    def forward(self, x):
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

        return o3


class MeanShift_n(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift_n, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [
                    nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group),
                    nn.ReLU(inplace=True),
                ]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [
                nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group),
                nn.ReLU(inplace=True),
            ]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, multi_scale, group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)



class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class AttentionBranch(nn.Module):

    def __init__(self, nf, k_size=3):

        super(AttentionBranch, self).__init__()
        self.k1 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.k2 = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution

    def forward(self, x):
        
        y = self.k1(x)
        y = self.lrelu(y)
        y = self.k2(y)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out


class AAB(nn.Module):

    def __init__(self, nf, reduction=4, K=3, t=30):
        super(AAB, self).__init__()
        self.t = t
        self.K = K

        self.conv_first = nn.Conv2d(nf, nf, kernel_size=1, bias=False)
        self.conv_last = nn.Conv2d(nf, nf, kernel_size=1, bias=False)        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Attention Dropout Module
        self.ADM = nn.Sequential(
            nn.Linear(nf, nf // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nf // reduction, self.K, bias=False),
        )
        
        # attention branch
        self.attention = AttentionBranch(nf)  
        
        # non-attention branch
        # 3x3 conv for A2N
        self.non_attention = nn.Conv2d(nf, nf, kernel_size=3, padding=(3 - 1) // 2, bias=False)         
        # 1x1 conv for A2N-M (Recommended, fewer parameters)
        # self.non_attention = nn.Conv2d(nf, nf, kernel_size=1, bias=False) 

    def forward(self, x, i):
        residual = x
        a, b, c, d = x.shape

        x = self.conv_first(x)
        x = self.lrelu(x)

        # Attention Dropout
        y = self.avg_pool(x).view(a,b)
        y = self.ADM(y)
        ax = F.softmax(y/self.t, dim = 1)

        attention = self.attention(x)

        non_attention = self.non_attention(x)
    
        x = attention * ax[:,0].view(a,1,1,1) + non_attention * ax[:,1].view(a,1,1,1)
        x = self.lrelu(x)

        out = self.conv_last(x)
        out += residual

        #==========================================

        vis = torch.mean(out, dim=1)
        # vis = (vis - vis.min()) / (vis.max() - vis.min())
        # vis = vis[..., 88:217, 32:161]
        # vis = vis + 0.2
        # vis.clamp_max_(1)
        # print(torch.min(vis), torch.max(vis))
        # print(vis.shape)

        savepath = "logs/vis"
        filename = "airplanout.png"

        savepath = os.path.join(savepath, filename.replace(".png", ""))

        if not exists(savepath):
            os.mkdir(savepath)

        savepath = os.path.join(savepath, "x{0}.png".format(i))

        plt.imsave(savepath, vis.cpu().numpy()[0], cmap="jet")

        #seismic

        #==========================================

        return out


class AABModule(nn.Module):

    def __init__(self, nf, nb):
        super(AABModule, self).__init__()

        self.AABs = nn.ModuleList()
        for i in range(nb):
            self.AABs.append(AAB(nf=nf))

    def forward(self, x):

        out = x
        i = 0
        for module in self.AABs:
            out = module(out, i)
            i = i+1
        
        return out


class SE_Block(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class AttentionBranchplus(nn.Module):

    def __init__(self, nf, k_size=3, k_sizec=1):

        super(AttentionBranchplus, self).__init__()
        self.k1 = nn.Conv2d(4, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        # self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.kedge = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        # self.sigmoid = nn.Sigmoid()
        # self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_sizec, padding=(k_sizec - 1) // 2, bias=False) # 3x3 convolution

    def forward(self, x, x_edge):
        
        y = self.k1(x_edge)
        # y = self.sigmoid(y)
        
        # out = torch.mul(x, y)
        out = self.k4(y)

        return out


class AttentionBranchplusx(nn.Module):

    def __init__(self, nf, k_size=3, k_sizec=1):

        super(AttentionBranchplusx, self).__init__()
        self.k1 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.kedge = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        # self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_sizec, padding=(k_sizec - 1) // 2, bias=False) # 3x3 convolution

    def forward(self, x, x_edge):
        
        y = self.k1(x)
        y = self.sigmoid(y)
        
        out = torch.mul(x, y)
        out = self.k4(out)

        return out


class noAttentionBranchplus(nn.Module):

    def __init__(self, nf, k_size=3, k_sizec=1):

        super(noAttentionBranchplus, self).__init__()
        self.k1 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        # self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.kedge = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        # self.sigmoid = nn.Sigmoid()
        # self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_sizec, padding=(k_sizec - 1) // 2, bias=False) # 3x3 convolution

    def forward(self, x, x_edge):
        
        out = self.k1(self.k4(x))

        return out


class AABplus(nn.Module):

    def __init__(self, nf, reduction=4, K=2, t=30):
        super(AABplus, self).__init__()
        self.t = t
        self.K = K

        self.conv_first = nn.Conv2d(nf, nf, kernel_size=1, bias=False)
        self.conv_last = nn.Conv2d(nf, nf, kernel_size=1, bias=False)        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Attention Dropout Module
        self.ADM = nn.Sequential(
            nn.Linear(nf, nf // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nf // reduction, self.K, bias=False),
        )
        
        # attention branch
        self.attention = AttentionBranchplus(nf)  
        
        # non-attention branch
        # 3x3 conv for A2N

        # self.non_attention = nn.Conv2d(nf, nf, kernel_size=3, padding=(3 - 1) // 2, bias=False)  

        self.non_attention = nn.Sequential(
            *[ 
                nn.Conv2d(nf, nf, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(nf, nf, kernel_size=1, padding=0, bias=False)
            ]
        )       
        # 1x1 conv for A2N-M (Recommended, fewer parameters)
        # self.non_attention = nn.Conv2d(nf, nf, kernel_size=1, bias=False) 

    def forward(self, x):
        x_edge = x[1]
        x = x[0]

        residual = x
        a, b, c, d = x.shape

        x = self.conv_first(x)
        x = self.lrelu(x)

        # Attention Dropout
        y = self.avg_pool(x).view(a,b)
        y = self.ADM(y)
        ax = F.softmax(y/self.t, dim = 1)

        attention = self.attention(x, x_edge)
        non_attention = self.non_attention(x)
    
        x = attention * ax[:,0].view(a,1,1,1) + non_attention * ax[:,1].view(a,1,1,1)
        x = self.lrelu(x)

        out = self.conv_last(x)
        out += residual

        return [out,x_edge]



class AABplusv2(nn.Module):

    def __init__(self, nf, reduction=4, K=2, t=50):
        super(AABplusv2, self).__init__()
        self.t = t
        self.K = K

        # self.conv_first = nn.Conv2d(nf, nf, kernel_size=1, bias=False)
        self.conv_last = nn.Conv2d(nf, nf, kernel_size=1, bias=False)        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Attention Dropout Module
        self.ADM = nn.Sequential(
            nn.Linear(nf, nf // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nf // reduction, self.K, bias=False),
        )
        
        # attention branch
        self.attention = AttentionBranchplus(nf)  
        
        # non-attention branch
        # 3x3 conv for A2N

        # self.non_attention = nn.Conv2d(nf, nf, kernel_size=3, padding=(3 - 1) // 2, bias=False)  

        self.non_attention = nn.Sequential(
            *[ 
                nn.Conv2d(nf, nf, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(nf, nf, kernel_size=3, padding=1, bias=False)
            ]
        )       
        # 1x1 conv for A2N-M (Recommended, fewer parameters)
        # self.non_attention = nn.Conv2d(nf, nf, kernel_size=1, bias=False) 

    def forward(self, x, x_edge):

        residual = x
        a, b, c, d = x.shape

        # x = self.conv_first(x)
        # x = self.lrelu(x)

        # Attention Dropout
        y = self.avg_pool(x).view(a,b)
        y = self.ADM(y)
        ax = F.softmax(y/self.t, dim = 1)

        attention = self.attention(x, x_edge)
        non_attention = self.non_attention(x)
    
        x = attention * ax[:,0].view(a,1,1,1) + non_attention * ax[:,1].view(a,1,1,1)
        x = self.lrelu(x)

        out = self.conv_last(x)
        out += residual

        return out


class AABplusv3(nn.Module):

    def __init__(self, nf, reduction=4, K=2, t=50):
        super(AABplusv3, self).__init__()
        self.t = t
        self.K = K

        # self.conv_first = nn.Conv2d(nf, nf, kernel_size=1, bias=False)
        self.conv_last = nn.Conv2d(nf, nf, kernel_size=1, bias=False)        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Attention Dropout Module
        self.ADM = nn.Sequential(
            nn.Linear(nf, nf // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nf // reduction, self.K, bias=False),
        )
        
        # attention branch
        # self.attention = AttentionBranchplusx(nf)
        # self.attention = AttentionBranchplus(nf)  
        self.attention = noAttentionBranchplus(nf)  
        
        # non-attention branch
        # 3x3 conv for A2N

        # self.non_attention = nn.Conv2d(nf, nf, kernel_size=3, padding=(3 - 1) // 2, bias=False)  

        self.non_attention = nn.Sequential(
            *[ 
                nn.Conv2d(nf, nf, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(nf, nf, kernel_size=3, padding=1, bias=False)
            ]
        )       
        # 1x1 conv for A2N-M (Recommended, fewer parameters)
        # self.non_attention = nn.Conv2d(nf, nf, kernel_size=1, bias=False) 

    def forward(self, x, x_edge):

        residual = x
        a, b, c, d = x.shape

        # x = self.conv_first(x)
        # x = self.lrelu(x)

        # Attention Dropout
        y = self.avg_pool(x).view(a,b)
        y = self.ADM(y)
        ax = F.softmax(y/self.t, dim = 1)

        attention = self.attention(x, x_edge)
        non_attention = self.non_attention(x)
    
        x = attention * ax[:,0].view(a,1,1,1) + non_attention * ax[:,1].view(a,1,1,1)
        x = self.lrelu(x)

        out = self.conv_last(x)
        out += residual

        return out


class SE_Block(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AABModuleplus(nn.Module):

    def __init__(self, nf, nb):
        super(AABModuleplus, self).__init__()

        self.AABpluss = nn.ModuleList()
        for i in range(nb):
            # if i>=nb//2:
            self.AABpluss.append(AABplusv2(nf=nf))
            # else:
            # self.AABpluss.append(AABplusv3(nf=nf))

    def forward(self, x, x_edge):

        out = x
        for module in self.AABpluss:
            out = module(out, x_edge)      

        return out



class HFCAttention(nn.Module):

    def __init__(self, nf, k_size=3):

        super(HFCAttention, self).__init__()

        self.k1 = nn.Conv2d(4, nf, kernel_size=1, padding=0, bias=False) # 3x3 convolution
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)

    def forward(self, x_edge):

        y = self.k1(x_edge)
        y = self.sigmoid(y)

        return y

class HFCPA(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch, kernel_size):
        super(HFCPA, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size, padding=0, bias=True)
        self.seblock = SE_Block(out_ch)
        self.paatten = HFCAttention(out_ch)

    def forward(self, x, x_edge):

        residual = x
        x1 = self.conv1(x)
        x2 = torch.cat([x, x1], dim=1)
        x3 = self.conv2(x2)
        x4 = torch.cat([x1, x3], dim=1)
        x = self.seblock(x4)
        y = self.paatten(x_edge)
        out = torch.mul(x, y)
        out = out + residual

        return out


class ENHFCPA(nn.Module):
    def __init__(self, inp, oup, ratio=2, dw_size=3, stride=1, relu=True):
        super(ENHFCPA, self).__init__()
        mad_channels = math.ceil(oup / ratio)

        self.HFCPA = HFCPA(
                in_ch = inp, 
                mid_ch = mad_channels, 
                out_ch = oup, 
                kernel_size = dw_size
            )
        self.relu = nn.ReLU(inplace=True)


        self.cft = nn.Sequential(
            nn.Conv2d(
                oup,
                oup,
                dw_size,
                1,
                dw_size // 2,
                groups=mad_channels,
                bias=False,
            ),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.feat_enhanced = nn.Sequential(
            nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.fuse = nn.Conv2d(oup*2, oup, kernel_size=1, padding=0, bias=False)

    def forward(self, x, x_edge):
        x1 = self.relu(self.HFCPA(x, x_edge))
        x2 = self.cft(x1)
        x_enhanced = self.feat_enhanced(x)
        out = self.fuse(torch.cat([x1, x2], dim=1))
        out = out + x_enhanced

        return out

class ElpaAttentionBranch(nn.Module):

    def __init__(self, nf, k_size=1):

        super(ElpaAttentionBranch, self).__init__()

        self.kedge = nn.Conv2d(4, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution

    def forward(self, x, x_edge):
        
        y = self.kedge(x_edge)
        y = self.sigmoid(y)
        
        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out


class ElpanoAttentionBranch(nn.Module):

    def __init__(self, nf, ratio=4):

        super(ElpanoAttentionBranch, self).__init__()

        nfratio = nf // ratio
        self.distilled_channels = nfratio
        self.k1 = nn.Conv2d(nfratio, nfratio, 3, 1, 1)
        self.k2 = nn.Conv2d(nfratio, nfratio, 1, 1, 0)
        self.k3 = nn.Conv2d(nfratio, nfratio, 3, 2, 1)
        self.k4 = nn.Conv2d(nfratio, nfratio, 1, 1, 0)
        self.k5 = nn.Conv2d(nfratio, nfratio, 3, 2, 1)
        self.k6 = nn.Conv2d(nfratio, nfratio, 1, 1, 0)
        self.k7 = nn.Conv2d(nfratio, nfratio, 3, 2, 1)
        self.k8 = nn.Conv2d(nfratio, nfratio, 1, 1, 0)

        self.k11 = nn.Conv2d(4, nfratio, 1, 1, 0)
        self.k22 = nn.Conv2d(nfratio, nfratio, 1, 1, 0)
        self.k33 = nn.Conv2d(4, nfratio, 1, 2, 0)
        self.k44 = nn.Conv2d(nfratio, nfratio, 1, 1, 0)
        self.k55 = nn.Conv2d(4, nfratio, 1, 2, 0)
        self.k555 = nn.Conv2d(nfratio, nfratio, 1, 2, 0)
        self.k66 = nn.Conv2d(nfratio, nfratio, 1, 1, 0)
        self.k77 = nn.Conv2d(4, nfratio, 1, 2, 0)
        self.k777 = nn.Conv2d(nfratio, nfratio, 1, 2, 0)
        self.k7777 = nn.Conv2d(nfratio, nfratio, 1, 2, 0)
        self.k88 = nn.Conv2d(nfratio, nfratio, 1, 1, 0)
        self.relu = nn.ReLU(True)


    def forward(self, x, x_edge):

        b, c, w, h = x.size()

        distilled_c1, remaining_c1 = torch.split(
            x, (self.distilled_channels, self.distilled_channels*3), dim=1
        )
        out_1 = self.k2(self.relu(self.k1(distilled_c1))) + self.k22(self.relu(self.k11(x_edge)))

        distilled_c2, remaining_c2 = torch.split(
            remaining_c1, (self.distilled_channels, self.distilled_channels*2), dim=1
        )
        out_2 = self.k4(self.relu(self.k3(distilled_c2))) + self.k44(self.relu(self.k33(x_edge)))

        distilled_c3, remaining_c3 = torch.split(
            remaining_c2, (self.distilled_channels, self.distilled_channels), dim=1
        )
        out_3 = self.k6(self.relu(self.k5(self.k5(distilled_c3)))) + self.k66(self.relu(self.k555(self.k55(x_edge))))

        out_4 = self.k8(self.relu(self.k7(self.k7(self.k7(remaining_c3))))) + self.k88(self.relu(self.k7777(self.k777(self.k77(x_edge)))))

        out_2 = F.interpolate(out_2, size=(w, h), mode="bilinear")
        out_3 = F.interpolate(out_3, size=(w, h), mode="bilinear")
        out_4 = F.interpolate(out_4, size=(w, h), mode="bilinear")

        out = torch.cat([out_1, out_2, out_3, out_4], dim = 1)
        
        return out


class ELPAblock(nn.Module):
    def __init__(self, nf, ratio=4, t=30, K=2):
        super(ELPAblock, self).__init__()

        self.t = t
        self.K = K

        self.conv_first = nn.Conv2d(nf, nf, kernel_size=1, bias=False)
        self.conv_last = nn.Conv2d(nf, nf, kernel_size=1, bias=False)        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Attention Dropout Module
        self.ADM = nn.Sequential(
            nn.Linear(nf, nf // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nf // ratio, self.K, bias=False),
        )
        
        # attention branch
        self.attention = ElpaAttentionBranch(nf)  
        
        # non-attention branch
        self.non_attention = ElpanoAttentionBranch(nf, ratio)  

    def forward(self, x, x_edge):
        
        residual = x
        a, b, c, d = x.shape

        x = self.conv_first(x)
        x = self.lrelu(x)

        # Attention Dropout
        y = self.avg_pool(x).view(a,b)
        y = self.ADM(y)
        ax = F.softmax(y/self.t, dim = 1)

        attention = self.attention(x, x_edge)
        non_attention = self.non_attention(x, x_edge)
    
        x = attention * ax[:,0].view(a,1,1,1) + non_attention * ax[:,1].view(a,1,1,1)
        x = self.lrelu(x)

        out = self.conv_last(x)
        out += residual

        return out


class ELPAMould(nn.Module):
    def __init__(self, nf, num_modules, ratio=4):
        super(ELPAMould, self).__init__()

        self.ELPAblock = nn.ModuleList()
        for i in range(num_modules):
            self.ELPAblock.append(ELPAblock(nf, ratio))
        

    def forward(self, x, x_edge):

        out = x
        for module in self.ELPAblock:
            out = module(out, x_edge)     

        return out