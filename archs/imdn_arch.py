import imp
import torch
import os
import time
from matplotlib import pyplot as plt
import torch.nn as nn
from .common import IMDModule, conv_block, conv_layer, pixelshuffle_block, IMDModulev2
from litsr.utils.registry import ArchRegistry
from .upsampler import CSUM_WO_WN, Multiscaleupsamplev5
from litsr.archs.upsampler import CSUM
from torch.nn import functional as F


@ArchRegistry.register()
class IMdnNet(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
        super(IMdnNet, self).__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        # IMDBs
        self.IMDB1 = IMDModule(in_channels=nf)
        self.IMDB2 = IMDModule(in_channels=nf)
        self.IMDB3 = IMDModule(in_channels=nf)
        self.IMDB4 = IMDModule(in_channels=nf)
        self.IMDB5 = IMDModule(in_channels=nf)
        self.IMDB6 = IMDModule(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type="lrelu")

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)

        # **************************可视化卷积核****************************
        savepath = "res/vis/learning-feat"
        # conv1 = self.IMDB6.c1
        conv1 = self.IMDB6.c1
        # savepath_e = os.path.join(savepath, "kernal3x3.png")
        point = 0
        localw = conv1.weight.cpu().clone()   
        print("total of number of filter : ", len(localw))
        plt.figure(figsize=(20, 20))
        for i in range(1, len(localw)+1):
            savepath_e = os.path.join(savepath, '64', str(i)+"kernal3x3.png")
            # savepath_e = os.path.join(savepath, "kernal3x3.png")
            localw0 = localw[i-1+point]  
            # mean of 3 channel.
            #localw0 = torch.mean(localw0,dim=0)
            # there should be 3(3 channels) 11 * 11 filter.
            if (len(localw0)) > 1:
                for idx, filer in enumerate(localw0):
                    # print(idx,filer,abs(filer[ :, :]),abs(localw0[:,0,0]).max())
                    plt.subplot(8, 8, idx+1) 
                    plt.axis('off')
                    plt.imshow(abs(filer[ :, :]).detach(),cmap='binary',vmin=0, vmax=abs(localw0[:,0,0]).max())
                    # plt.imshow(abs(filer[ :, :]).detach(),cmap='binary',vmin=0, vmax=0.15)
                plt.savefig(savepath_e)
                # break
                print(i)
                # time.sleep(10)
            else:
                plt.subplot(8, 8, i) 
                plt.axis('off')
                plt.imshow(localw0[0, :, :].detach(),cmap='gray')
        plt.savefig(savepath_e)
        # **************************可视化卷积核****************************

        out_B = self.c(
            torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1)
        )
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output



@ArchRegistry.register()
class IMdnNetv2(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
        super(IMdnNetv2, self).__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        # IMDBs
        self.IMDB1 = IMDModulev2(in_channels=nf)
        self.IMDB2 = IMDModulev2(in_channels=nf)
        self.IMDB3 = IMDModulev2(in_channels=nf)
        self.IMDB4 = IMDModulev2(in_channels=nf)
        self.IMDB5 = IMDModulev2(in_channels=nf)
        self.IMDB6 = IMDModulev2(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type="lrelu")

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

    def forward(self, input):
        xlr = input[:,0:3,:,:]
        x_edge = input[:,3:7,:,:]
        out_fea = self.fea_conv(xlr)
        out_B1 = self.IMDB1(out_fea, x_edge)
        out_B2 = self.IMDB2(out_B1, x_edge)
        out_B3 = self.IMDB3(out_B2, x_edge)
        out_B4 = self.IMDB4(out_B3, x_edge)
        out_B5 = self.IMDB5(out_B4, x_edge)
        out_B6 = self.IMDB6(out_B5, x_edge)

        out_B = self.c(
            torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1)
        )
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output



@ArchRegistry.register()
class IMdnOusNet(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
        super().__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        # IMDBs
        self.IMDB1 = IMDModule(in_channels=nf)
        self.IMDB2 = IMDModule(in_channels=nf)
        self.IMDB3 = IMDModule(in_channels=nf)
        self.IMDB4 = IMDModule(in_channels=nf)
        self.IMDB5 = IMDModule(in_channels=nf)
        self.IMDB6 = IMDModule(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type="lrelu")

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        # upsample_block = pixelshuffle_block
        # self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        # self.upsampler = LightMLPInterpolate(nf, radius=3)
        self.upsampler = Multiscaleupsamplev5(nf, split=4)

    def forward(self, input, outsize):
        out_fea = self.fea_conv(input)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)

        out_B = self.c(
            torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1)
        )
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr, outsize)
        return output


@ArchRegistry.register()
class IMdnCSUMNet(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
        super().__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        # IMDBs
        self.IMDB1 = IMDModule(in_channels=nf)
        self.IMDB2 = IMDModule(in_channels=nf)
        self.IMDB3 = IMDModule(in_channels=nf)
        self.IMDB4 = IMDModule(in_channels=nf)
        self.IMDB5 = IMDModule(in_channels=nf)
        self.IMDB6 = IMDModule(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type="lrelu")

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        self.upsampler = CSUM(
            nf, kSize=3, out_channels=out_nc, interpolate_mode="bilinear", levels=4
        )

    def forward(self, input, outsize):
        out_fea = self.fea_conv(input)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)

        out_B = self.c(
            torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1)
        )
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr, outsize)
        return output


@ArchRegistry.register()
class IMdnOsmNet(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
        super().__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        # IMDBs
        self.IMDB1 = IMDModule(in_channels=nf)
        self.IMDB2 = IMDModule(in_channels=nf)
        self.IMDB3 = IMDModule(in_channels=nf)
        self.IMDB4 = IMDModule(in_channels=nf)
        self.IMDB5 = IMDModule(in_channels=nf)
        self.IMDB6 = IMDModule(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type="lrelu")

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.upsampler = nn.Sequential(
            wn(nn.Conv2d(nf, 64 * 25, 3, padding=1)),
            nn.PixelShuffle(5),
            wn(nn.Conv2d(64, 3, 3, padding=1)),
        )

    def forward(self, input, outsize):
        out_fea = self.fea_conv(input)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)

        out_B = self.c(
            torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1)
        )
        out_lr = self.LR_conv(out_B) + out_fea
        output = F.interpolate(
            self.upsampler(out_lr), outsize, mode="bicubic", align_corners=False
        )
        return output
