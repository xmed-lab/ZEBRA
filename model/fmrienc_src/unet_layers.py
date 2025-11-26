import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm



def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)
    # return F.relu(x)

def get_activation_layer(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError("activation should be relu/gelu, not {activation}.")

def get_norm_layer(in_channels, mode):
    if mode == 'bn':
        return nn.BatchNorm2d(in_channels)
    elif mode == 'in':
        return nn.InstanceNorm2d(in_channels, affine=False)
    elif mode == 'gn':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels) # num_groups is set as 32 in MaskGIT, instead of dims//32
    elif mode == 'none':
        return nn.Identity()
    else:
        raise ValueError


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, norm_type='gn', use_spect=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.use_spect = use_spect

        self.norm1 = get_norm_layer(in_channels, norm_type)
        if use_spect:
            self.conv1 = SpectralNorm(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        else:
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            if use_spect:
                self.temb_proj = SpectralNorm(torch.nn.Linear(temb_channels, out_channels))
            else:
                self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = get_norm_layer(out_channels, norm_type)
        self.dropout = torch.nn.Dropout(dropout)

        if use_spect:
            self.conv2 = SpectralNorm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        else:
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # QXL: previous models have bug here
        # if use_spect:
        #     self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # else:
        #     self.conv2 = SpectralNorm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                if use_spect:
                    self.conv_shortcut = SpectralNorm(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
                else:
                    self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                if use_spect:
                    self.nin_shortcut = SpectralNorm(torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
                else:
                    self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

# bias=False
class ResnetBlock2(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, norm_type='gn', use_spect=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.use_spect = use_spect

        self.norm1 = get_norm_layer(in_channels, norm_type)
        if use_spect:
            self.conv1 = SpectralNorm(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if temb_channels > 0:
            if use_spect:
                self.temb_proj = SpectralNorm(torch.nn.Linear(temb_channels, out_channels))
            else:
                self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = get_norm_layer(out_channels, norm_type)
        self.dropout = torch.nn.Dropout(dropout)

        if use_spect:
            self.conv2 = SpectralNorm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                if use_spect:
                    self.conv_shortcut = SpectralNorm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
                else:
                    self.conv_shortcut = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            else:
                if use_spect:
                    self.nin_shortcut = SpectralNorm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
                else:
                    self.nin_shortcut = torch.nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(h)
            else:
                x = self.nin_shortcut(h)

        return x + h


class DownBlock2D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, norm_type='bn'):
        super(DownBlock2D, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = get_norm_layer(out_features, norm_type)
        # self.norm = nn.BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = nonlinearity(out)
        # out = F.relu(out)

        return out

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, norm_type='bn'):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
            self.norm = get_norm_layer(in_channels, norm_type)
            # self.norm = nn.BatchNorm2d(in_channels, affine=True)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
            x = self.norm(x)
            x = nonlinearity(x)
            # x = F.relu(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class Downsample2(nn.Module):
    def __init__(self, in_channels, with_conv, use_spect):
        super().__init__()
        self.with_conv = with_conv
        self.use_spect = use_spect
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            if use_spect:
                self.conv = SpectralNorm(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0))
            else:
                self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class UpBlock2D(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, norm_type='bn'):
        super(UpBlock2D, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = get_norm_layer(out_features, norm_type)
        # self.norm = nn.BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        # out = F.relu(out)
        out = nonlinearity(out)
        return out

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True, norm_type='bn'):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            self.norm = get_norm_layer(in_channels, norm_type)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
            x = self.norm(x)
            x = nonlinearity(x)
            # x = F.relu(x)
        return x

class Upsample2(nn.Module):
    def __init__(self, in_channels, with_conv, use_spect):
        super().__init__()
        self.with_conv = with_conv
        self.use_spect = use_spect
        if self.with_conv:
            if use_spect:
                self.conv = SpectralNorm(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            else:
                self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x



class MaskGIT_Encoder(nn.Module):
    def __init__(self, config, sample_with_conv):
        super(MaskGIT_Encoder, self).__init__()

        ch = config['ch_base']
        in_ch = config['in_ch']
        ch_mult = config['ch_mult']
        ch_latent = config['ch_latent']
        norm_mode = config['norm_mode']
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = config['num_res_blocks']
        self.num_res_mids = config['num_res_mids']
        resolution = config['resolution']
        use_spectral_norm = config['use_spectral_norm']

        # pre-conv
        if use_spectral_norm:
            self.conv_in = SpectralNorm(torch.nn.Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.conv_in = torch.nn.Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1, bias=False)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult) # [1, 1,1,2,2,4]
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = [] # nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * in_ch_mult[i_level+1]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock2(in_channels=block_in, out_channels=block_out,
                                         conv_shortcut=False, dropout=0, temb_channels=0,
                                         norm_type=norm_mode, use_spect=use_spectral_norm))
                block_in = block_out
            down = nn.Module()
            down.block = nn.Sequential(*block)
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2(block_in, sample_with_conv, use_spectral_norm)
                curr_res = curr_res // 2
            self.down.append(down)

        # mid
        self.mid = nn.ModuleList()
        for i_block in range(self.num_res_mids):
            self.mid.append(ResnetBlock2(in_channels=block_in, out_channels=block_in,
                                       conv_shortcut=False, dropout=0, temb_channels=0,
                                       norm_type=norm_mode, use_spect=use_spectral_norm))

        # end
        self.norm_out = get_norm_layer(block_in, norm_mode)
        if use_spectral_norm:
            self.conv_out = SpectralNorm(torch.nn.Conv2d(block_in, ch_latent, kernel_size=1, stride=1))
        else:
            self.conv_out = torch.nn.Conv2d(block_in, ch_latent, kernel_size=1, stride=1)

    def forward(self, src, tgt):
        src_x = self.conv_in(src)
        tgt_x = self.conv_in(tgt)

        # encoder
        for i_level in range(self.num_resolutions):
            src_x = self.down[i_level].block(src_x)
            tgt_x = self.down[i_level].block(tgt_x)
            if i_level != self.num_resolutions-1:
                src_x = self.down[i_level].downsample(src_x)
                tgt_x = self.down[i_level].downsample(tgt_x)

        for i_level in range(self.num_res_mids):
            src_x = self.mid[i_level](src_x)
            tgt_x = self.mid[i_level](tgt_x)

        src_x = nonlinearity(self.norm_out(src_x))
        src_x = self.conv_out(src_x)
        tgt_x = nonlinearity(self.norm_out(tgt_x))
        tgt_x = self.conv_out(tgt_x)

        return src_x, tgt_x

    def forward_oneway(self, input):
        x = self.conv_in(input)

        # encoder
        for i_level in range(self.num_resolutions):
            x = self.down[i_level].block(x)
            if i_level != self.num_resolutions-1:
                x = self.down[i_level].downsample(x)

        for i_level in range(self.num_res_mids):
            x = self.mid[i_level](x)

        x = nonlinearity(self.norm_out(x))
        x = self.conv_out(x)

        return x

class MaskGIT_Decoder(nn.Module):
    def __init__(self, config, sample_with_conv):
        super(MaskGIT_Decoder, self).__init__()

        ch = config['ch_base']
        out_ch = config['out_ch']
        ch_mult = config['ch_mult']
        ch_latent = config['ch_latent']
        norm_mode = config['norm_mode']
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = config['num_res_blocks']
        self.num_res_mids = config['num_res_mids']
        resolution = config['resolution']
        use_spectral_norm = config['use_spectral_norm']

        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # in_ch_mult = (1,) + tuple(ch_mult) # [1, 1,1,2,2,4]
        in_ch_mult = tuple(ch_mult) + (ch_mult[-1],) # [1,1,2,2,4, 4]
        block_in = ch * ch_mult[-1]

        # begin
        if use_spectral_norm:
            self.conv_in = SpectralNorm(torch.nn.Conv2d(ch_latent, block_in, kernel_size=3, stride=1, padding=1))
        else:
            self.conv_in = torch.nn.Conv2d(ch_latent, block_in, kernel_size=3, stride=1, padding=1)

        # mid
        self.mid = nn.ModuleList()
        for i_block in range(self.num_res_mids):
            self.mid.append(ResnetBlock2(in_channels=block_in, out_channels=block_in,
                                       conv_shortcut=False, dropout=0, temb_channels=0,
                                       norm_type=norm_mode, use_spect=use_spectral_norm))

        # upsampling
        self.up = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = [] # nn.ModuleList()
            block_in = ch * in_ch_mult[-i_level - 1]
            block_out = ch * in_ch_mult[-i_level - 2]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock2(in_channels=block_in, out_channels=block_out,
                                         conv_shortcut=False, dropout=0, temb_channels=0,
                                         norm_type=norm_mode, use_spect=use_spectral_norm))
                block_in = block_out
            up = nn.Module()
            up.block = nn.Sequential(*block)
            if i_level < self.num_resolutions - 1:
                up.upsample = Upsample2(block_in, sample_with_conv, use_spectral_norm)
                curr_res = curr_res * 2
            self.up.append(up)

        # end
        self.norm_out = get_norm_layer(block_in, norm_mode)
        if use_spectral_norm:
            conv_out = SpectralNorm(nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1))
        else:
            conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv_out = conv_out
        self.out_norm = nn.Tanh() if config['with_tanh'] else nn.Identity()

    def forward(self, src, tgt):
        # decoder
        src_x = self.conv_in(src)
        tgt_x = self.conv_in(tgt)

        # mid
        for i_level in range(self.num_res_mids):
            src_x = self.mid[i_level](src_x)
            tgt_x = self.mid[i_level](tgt_x)

        for i_level in range(self.num_resolutions):
            src_x = self.up[i_level].block(src_x)
            tgt_x = self.up[i_level].block(tgt_x)
            if i_level != self.num_resolutions - 1:
                src_x = self.up[i_level].upsample(src_x)
                tgt_x = self.up[i_level].upsample(tgt_x)

        src_x = nonlinearity(self.norm_out(src_x))
        src_rec = self.conv_out(src_x)
        tgt_x = nonlinearity(self.norm_out(tgt_x))
        tgt_rec = self.out_norm(self.conv_out(tgt_x))

        return src_rec, tgt_rec

    def forward_oneway(self, input):
        # decoder
        x = self.conv_in(input)

        # mid
        for i_level in range(self.num_res_mids):
            x = self.mid[i_level](x)

        for i_level in range(self.num_resolutions):
            x = self.up[i_level].block(x)
            if i_level != self.num_resolutions - 1:
                x = self.up[i_level].upsample(x)

        x = nonlinearity(self.norm_out(x))
        rec = self.out_norm(self.conv_out(x))

        return rec

