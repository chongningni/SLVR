import  torch
import  torch.nn as nn
import  torch.nn.functional as F

from litsr.utils.registry import ArchRegistry


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3,
                       bias=False):
    """
    Upsample features according to `upscale_factor`.
    """
    padding = kernel_size // 2
    conv = nn.Conv2d(in_channels,
                     out_channels * (upscale_factor ** 2),
                     kernel_size,
                     padding=padding,
                     bias=bias)  
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])



class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d):
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

class OSAG(nn.Module):
    def __init__(self, channel_num=64, bias = True, block_num=4,**kwargs):
        super(OSAG, self).__init__()

        ffn_bias    = kwargs.get("ffn_bias", False)
        window_size = kwargs.get("window_size", 0)
        pe          = kwargs.get("pe", False)

        print("window_size: %d"%(window_size))
        print('with_pe', pe)
        print("ffn_bias: %d"%(ffn_bias))

        block_script_name   = kwargs["block_script_name"]
        block_class_name    = kwargs["block_class_name"]

        script_name     = "archs." + block_script_name
        package         = __import__(script_name, fromlist=True)
        block_class     = getattr(package, block_class_name)
        group_list = []
        for _ in range(block_num):
            temp_res = block_class(channel_num,bias,ffn_bias=ffn_bias,window_size=window_size,with_pe=pe)
            group_list.append(temp_res)
        group_list.append(nn.Conv2d(channel_num,channel_num,1,1,0,bias=bias))
        self.residual_layer = nn.Sequential(*group_list)
        esa_channel     = max(channel_num // 4, 16)
        self.esa        = ESA(esa_channel, channel_num)
        
    def forward(self, x):
        out = self.residual_layer(x)
        out = out + x
        return self.esa(out)


@ArchRegistry.register()
class OmniSR(nn.Module):
    def __init__(self,num_in_ch=3,num_out_ch=3,num_feat=32,**kwargs):
        super(OmniSR, self).__init__()

        res_num     = kwargs["res_num"]
        up_scale    = kwargs["upsampling"]
        bias        = kwargs["bias"]
        self.shfflescale = kwargs["shfflescale"]

        residual_layer  = []
        self.res_num    = res_num

        for _ in range(res_num):
            temp_res = OSAG(channel_num=num_feat,**kwargs)
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)
        self.input  = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.up     = pixelshuffle_block(num_feat,num_out_ch,up_scale,bias=bias)

        # self.tail   = pixelshuffle_block(num_feat,num_out_ch,up_scale,bias=bias)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, sqrt(2. / n))

        self.window_size   = kwargs["window_size"]
        self.up_scale = up_scale
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = ((self.window_size) - h % (self.window_size)) % (self.window_size)
        mod_pad_w = ((self.window_size) - w % (self.window_size)) % (self.window_size)
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        residual= self.input(x)
        out     = self.residual_layer(residual)

        # origin
        out     = torch.add(self.output(out),residual)
        out     = self.up(out)
        
        out = out[:, :, :H*self.up_scale, :W*self.up_scale]
        return  out