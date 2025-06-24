from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .common import pixelshuffle_block
from litsr.utils.registry import ArchRegistry
from torch import Tensor

def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value

def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), 
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.
    
    Parameters
    ----------
    args: Definition of Modules in order.
    -------
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m



# class RLFB(nn.Module):
#     """
#     Residual Local Feature Block (RLFB).
#     """

#     def __init__(self,
#                  in_channels,
#                  mid_channels=None,
#                  out_channels=None,
#                  esa_channels=16):
#         super(RLFB, self).__init__()

#         if mid_channels is None:
#             mid_channels = in_channels
#         if out_channels is None:
#             out_channels = in_channels

#         self.c1_r = conv_layer(in_channels, mid_channels, 3)
#         self.c2_r = conv_layer(mid_channels, mid_channels, 3)
#         self.c3_r = conv_layer(mid_channels, in_channels, 3)

#         self.c5 = conv_layer(in_channels, out_channels, 1)
#         self.esa = ESA(esa_channels, out_channels, nn.Conv2d)

#         self.act = activation('lrelu', neg_slope=0.05)

#     def forward(self, x):
#         out = (self.c1_r(x))
#         out = self.act(out)

#         out = (self.c2_r(out))
#         out = self.act(out)

#         out = (self.c3_r(out))
#         out = self.act(out)

#         out = out + x
#         out = self.esa(self.c5(out))

#         return out


    
# class RLFB(nn.Module):
#     """
#     Residual Local Feature Block (RLFB).
#     """

#     def __init__(self,
#                  in_channels,
#                  mid_channels=None,
#                  out_channels=None,
#                  esa_channels=16):
#         super(RLFB, self).__init__()

#         if mid_channels is None:
#             mid_channels = in_channels
#         if out_channels is None:
#             out_channels = in_channels

#         self.c1_r = conv_layer(in_channels, mid_channels, 3)
#         self.c2_r = conv_layer(mid_channels, in_channels, 3)
#         self.c3_r = conv_layer(in_channels, mid_channels, 3)
#         self.c4_r = conv_layer(mid_channels, in_channels, 3)

#         self.c5 = conv_layer(in_channels, out_channels, 1)
#         self.esa = ESA(esa_channels, out_channels, nn.Conv2d)
#         # self.pa = PA(out_channels)

#         self.act = activation('lrelu', neg_slope=0.05)

#     def forward(self, x):
#         out = (self.c1_r(x))
#         out = self.act(out)

#         out = (self.c2_r(out))
#         out = self.act(out)

#         out_ = out + x

#         out = (self.c3_r(out_))
#         out = self.act(out)

#         out = (self.c4_r(out))
#         out = self.act(out)

#         out = out + out_
#         out = self.esa(self.c5(out))

#         return out


class Pconv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x
    

class HAConv(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(HAConv, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


# class RLFB(nn.Module):
#     """
#     Residual Local Feature Block (RLFB).
#     """

#     def __init__(self,
#                  in_channels,
#                  mid_channels=None,
#                  out_channels=None,
#                  esa_channels=16):
#         super(RLFB, self).__init__()

#         if mid_channels is None:
#             mid_channels = in_channels
#         if out_channels is None:
#             out_channels = in_channels

#         self.c1_r = BSConvU(in_channels, in_channels, kernel_size=3)
#         self.c2_r = Pconv3(in_channels, 2, 'split_cat')
#         self.c3_r = BSConvU(in_channels, in_channels, kernel_size=3)
#         self.c4_r = Pconv3(in_channels, 2, 'split_cat')

#         self.c5 = conv_layer(in_channels, out_channels, 1)
#         self.esa = ESA(esa_channels, out_channels, nn.Conv2d)
#         # self.pa = CPA(out_channels)

#         self.act = activation('lrelu', neg_slope=0.05)

#     def forward(self, x):
#         out = (self.c1_r(x))
#         out = self.act(out)

#         out = (self.c2_r(out))
#         out = self.act(out)

#         out_ = out + x

#         out = (self.c3_r(out_))
#         out = self.act(out)

#         out = (self.c4_r(out))
#         out = self.act(out)

#         out = out + out_
#         out = self.esa(self.c5(out))

#         return out



class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


class SESA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d, p=0.25):
        super(SESA, self).__init__()
        f = num_feat // 4
        BSConvS_kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': p}
        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv2 = conv(f, f, 3, 2, 0)
        self.conv3 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_ = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        c1_ = (self.conv1(input))
        c1 = self.conv2(c1_)
        v_max = self.maxPooling(c1)
        v_range = self.GELU(self.conv_max(v_max))
        c3 = self.GELU(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4((c3 + cf))
        m = self.sigmoid(c4)

        return input * m
    


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
    
class PAG(nn.Module):
    """PA is pixel attention"""

    def __init__(self, nf):

        super(PAG, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1, groups=nf)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out

class CPA(nn.Module):
    """PA is pixel attention"""

    def __init__(self, nf):

        super(CPA, self).__init__()

        self.conv = nn.Conv2d(nf, nf, 1, stride=4, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)

        y = F.interpolate(y, (x.size(2), x.size(3)),
                           mode='nearest', align_corners=None)
        
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out
    

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % groups == 0)
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class ESA2(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESA2, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m
    

    
class RLFB2(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 esa_channels=16):
        super(RLFB2, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r1 = conv_layer(in_channels, 40, 1)
        self.c2_r1 = conv_layer(in_channels, 24, 3)

        self.c1_r2 = conv_layer(in_channels, 40, 1)
        self.c2_r2 = conv_layer(in_channels, 24, 3)

        self.c1_r3 = conv_layer(in_channels, 40, 1)
        self.c2_r3 = conv_layer(in_channels, 24, 3)

        self.c1_r4 = conv_layer(in_channels, 40, 1)
        self.c2_r4 = conv_layer(in_channels, 24, 3)

        self.pa = PA(out_channels)
        # self.pa2 = PA(out_channels)
        # self.esa = ESA2(esa_channels, out_channels, nn.Conv2d)

        self.c5 = conv_layer(in_channels, out_channels, 1)
        # self.esa = ESA(esa_channels, out_channels, nn.Conv2d)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):

        out11 = (self.c1_r1(x))
        out22 = (self.c2_r1(x))
        out = self.act(torch.cat([out11, out22], dim=1))

        out1 = (self.c1_r2(out))
        out2 = (self.c2_r2(out))
        out = self.act(torch.cat([out2+out22, out1+out11], dim=1))

        out_ = out + x

        out11 = (self.c1_r3(out_))
        out22 = (self.c2_r3(out_))
        out = self.act(torch.cat([out11, out22], dim=1))

        out1 = (self.c1_r4(out))
        out2 = (self.c2_r4(out))
        out = self.act(torch.cat([out2+out22, out1+out11], dim=1))
    
        out = out + out_

        out = self.c5(self.pa(out))

        return out

        # out = channel_shuffle(out, 2)
        # out = self.c5(self.pa(out)) 

        # return out + x

# class RLFB2(nn.Module):
#     """
#     Residual Local Feature Block (RLFB).
#     """

#     def __init__(self,
#                  in_channels,
#                  mid_channels=None,
#                  out_channels=None,
#                  esa_channels=16):
#         super(RLFB2, self).__init__()

#         if mid_channels is None:
#             mid_channels = in_channels
#         if out_channels is None:
#             out_channels = in_channels

#         self.c1_r = BSConvU(in_channels, mid_channels, kernel_size=3)
#         self.c2_r = BSConvU(mid_channels, mid_channels, kernel_size=3)
#         self.c3_r = BSConvU(mid_channels, mid_channels, kernel_size=3)
#         self.c4_r = BSConvU(mid_channels, in_channels, kernel_size=3)

#         self.c5 = conv_layer(in_channels, out_channels, 1)
#         self.c6 = conv_layer(in_channels + 3*mid_channels, out_channels, 1)

#         self.esa = SESA(in_channels, BSConvU)

#         self.act = activation('lrelu', neg_slope=0.05)

#     def forward(self, x):
#         out1_ = (self.c1_r(x))
#         out1 = self.act(out1_)

#         out2_ = (self.c2_r(out1))
#         out2 = self.act(out2_)

#         out3_ = (self.c3_r(out2))
#         out3 = self.act(out3_)

#         out4_ = (self.c4_r(out3))
#         out4 = self.act(out4_)

#         trunk = torch.cat([out1, out2, out3, out4], dim=1)
#         out = self.c6(trunk)

#         out = out + x
#         out = self.esa(self.c5(out))

#         return out


# class RLFB(nn.Module):
#     """
#     Residual Local Feature Block (RLFB).
#     """

#     def __init__(self,
#                  in_channels,
#                  mid_channels=None,
#                  out_channels=None,
#                  esa_channels=16):
#         super(RLFB, self).__init__()

#         if mid_channels is None:
#             mid_channels = in_channels
#         if out_channels is None:
#             out_channels = in_channels

#         self.c1_r = conv_layer(in_channels, in_channels, 3)
#         self.c2_r = conv_layer(in_channels, in_channels, 3)
#         self.c3_r = conv_layer(in_channels, in_channels, 3)

#         self.c5 = conv_layer(in_channels, out_channels, 1)
#         self.esa = ESA(esa_channels, out_channels, nn.Conv2d)

#         self.act = activation('lrelu', neg_slope=0.05)

#     def forward(self, x):
#         out1 = (self.c1_r(x))
#         out1 = self.act(out1)

#         out2 = (self.c2_r(x))
#         out2 = self.act(out2)

#         out3 = (self.c3_r(x))
#         out3 = self.act(out3)

#         out = out1 + out2 + out3 + x
#         out = self.esa(self.c5(out))

#         return out
    


# class RLFB(nn.Module):
#     """
#     Residual Local Feature Block (RLFB).
#     """

#     def __init__(self,
#                  in_channels,
#                  mid_channels=None,
#                  out_channels=None,
#                  esa_channels=16):
#         super(RLFB, self).__init__()

#         if mid_channels is None:
#             mid_channels = in_channels
#         if out_channels is None:
#             out_channels = in_channels

#         self.c1_r = conv_layer(in_channels, in_channels, 3)
#         self.c2_r = conv_layer(in_channels, in_channels, 3)
#         self.c3_r = conv_layer(in_channels, in_channels, 3)
#         self.c4_r = conv_layer(in_channels*3, in_channels, 1)

#         self.c5 = conv_layer(in_channels, out_channels, 1)
#         self.esa = ESA(esa_channels, out_channels, nn.Conv2d)

#         self.act = activation('lrelu', neg_slope=0.05)

#     def forward(self, x):
        
#         out1 = (self.c1_r(x))
#         out1 = self.act(out1)
#         out1 = (self.c1_r(out1))

#         out2 = (self.c2_r(x))
#         out2 = self.act(out2)
#         out2 = (self.c2_r(out2))

#         out3 = (self.c3_r(x))
#         out3 = self.act(out3)
#         out3 = (self.c3_r(out3))

#         out = self.c4_r(torch.cat([out1, out2, out3], dim=1)) + x
#         out = self.esa(self.c5(out))

#         return out

class ESAQ(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESAQ, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m
    

class RLFB(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 esa_channels=16):
        super(RLFB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, out_channels, 3)
        self.c2_r = conv_layer(in_channels, out_channels, 3)
        self.c3_r = conv_layer(out_channels, out_channels, 3)
        self.c4_r = conv_layer(in_channels, out_channels, 1)

        self.c5 = conv_layer(out_channels, out_channels, 1)
        self.esa = ESAQ(esa_channels, out_channels, nn.Conv2d)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        out = (self.c3_r(out))
        out = self.act(out)

        out = out + self.c4_r(x)
        out = self.esa(self.c5(out))

        return out



@ArchRegistry.register()
class RLFN(nn.Module):
    """
    Residual Local Feature Network (RLFN)
    Model definition of RLFN in `Residual Local Feature Network for 
    Efficient Super-Resolution`
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=64,
                 upscale=4):
        super(RLFN, self).__init__()

        self.conv_1 = conv_layer(in_channels,
                                       64,
                                       kernel_size=3)

        self.block_1 = RLFB(in_channels=64, out_channels=48)
        self.block_2 = RLFB(in_channels=48, out_channels=32)
        self.block_3 = RLFB(in_channels=32, out_channels=40)
        self.block_4 = RLFB(in_channels=40, out_channels=48)
        self.block_5 = RLFB(in_channels=48, out_channels=56)
        self.block_6 = RLFB(in_channels=56, out_channels=64)

        self.conv_2 = conv_layer(feature_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.upsampler = pixelshuffle_block(feature_channels,
                                                  out_channels,
                                                  upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)

        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)
        out_b6 = self.block_6(out_b5)

        out_low_resolution = self.conv_2(out_b6) + out_feature
        output = self.upsampler(out_low_resolution)

        return output
    


# class RLFN(nn.Module):
#     """
#     Residual Local Feature Network (RLFN)
#     Model definition of RLFN in `Residual Local Feature Network for 
#     Efficient Super-Resolution`
#     """

#     def __init__(self,
#                  in_channels=3,
#                  out_channels=3,
#                  feature_channels=48,
#                  upscale=4):
#         super(RLFN, self).__init__()

#         self.conv_1 = conv_layer(in_channels,
#                                        feature_channels,
#                                        kernel_size=3)

#         self.block_1 = RLFB(feature_channels, feature_channels-12)
#         self.block_2 = RLFB(feature_channels, feature_channels-12)
#         self.block_3 = RLFB(feature_channels, feature_channels-12)
#         self.block_4 = RLFB(feature_channels, feature_channels-12)
#         self.block_5 = RLFB(feature_channels, feature_channels-12)
#         self.block_6 = RLFB(feature_channels, feature_channels-12)

#         # self.block_1 = RLFB(feature_channels)
#         # self.block_2 = RLFB(feature_channels)
#         # self.block_3 = RLFB(feature_channels)
#         # self.block_4 = RLFB(feature_channels)
#         # self.block_5 = RLFB(feature_channels)
#         # self.block_6 = RLFB(feature_channels)

#         self.conv_2 = conv_layer(feature_channels,
#                                        feature_channels,
#                                        kernel_size=3)

#         self.upsampler = pixelshuffle_block(feature_channels,
#                                                   out_channels,
#                                                   upscale_factor=upscale)

#     def forward(self, x):

#         out_feature = self.conv_1(x)

#         out_b1 = self.block_1(out_feature)
#         out_b2 = self.block_2(out_b1)
#         out_b3 = self.block_3(out_b2)
#         out_b4 = self.block_4(out_b3)
#         out_b5 = self.block_5(out_b4)
#         out_b6 = self.block_6(out_b5)

#         out_low_resolution = self.conv_2(out_b6) + out_feature
#         output = self.upsampler(out_low_resolution)

#         return output
    

@ArchRegistry.register()
class RLFN2(nn.Module):
    """
    Residual Local Feature Network (RLFN)
    Model definition of RLFN in `Residual Local Feature Network for 
    Efficient Super-Resolution`
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=64,
                 upscale=4):
        super(RLFN2, self).__init__()

        self.conv_1 = conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.block_1 = RLFB2(feature_channels,feature_channels//2)
        self.block_2 = RLFB2(feature_channels,feature_channels//2)
        self.block_3 = RLFB2(feature_channels,feature_channels//2)
        self.block_4 = RLFB2(feature_channels,feature_channels//2)
        self.block_5 = RLFB2(feature_channels,feature_channels//2)
        self.block_6 = RLFB2(feature_channels,feature_channels//2)

        self.conv_2 = conv_layer(feature_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.upsampler = pixelshuffle_block(feature_channels,
                                                  out_channels,
                                                  upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)

        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)
        out_b6 = self.block_6(out_b5)

        out_low_resolution = self.conv_2(out_b6) + out_feature
        output = self.upsampler(out_low_resolution)

        return output