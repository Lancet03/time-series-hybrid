"""Copied from HN-MVTS/src/models/moderntcn.py (Apache-2.0).

Original SHA256: 171265e29edafeb0d7332aaa633c42ae4c104f1602bc761c0d86f932bfd94d8d
Network code is preserved; imports are package-relative.
"""
import torch
import torch.nn as nn

from .neural import NeuralModel
from .layers import ReversibleInstanceNormalization as RIN
from .layers import HyperLinearLayer


def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1,bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result

def fuse_bn(conv, bn):

    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=7):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=1, groups=groups,bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=small_kernel,
                                            stride=stride, padding=small_kernel // 2, groups=groups, dilation=1,bias=False)


    def forward(self, inputs):

        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)

        return out

    def PaddingTwoEdge1d(self,x,pad_length_left,pad_length_right,pad_values=0):

        D_out,D_in,ks=x.shape
        if pad_values ==0:
            pad_left = torch.zeros(D_out,D_in,pad_length_left)
            pad_right = torch.zeros(D_out,D_in,pad_length_right)
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right) * pad_values
        x = torch.cat([pad_left,x],dims=-1)
        x = torch.cat([x,pad_right],dims=-1)
        return x

    def get_equivalent_kernel_bias(self):

        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)

        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)

            eq_b += small_b

            eq_k += self.PaddingTwoEdge1d(small_k, (self.kernel_size - self.small_kernel) // 2,
                                          (self.kernel_size - self.small_kernel) // 2, 0)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):

        super(Block, self).__init__()
        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                         kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                         small_kernel=small_size, small_kernel_merged=small_kernel_merged, nvars=nvars)
        self.norm = nn.BatchNorm1d(dmodel)

        #convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        #convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff//dmodel

    def forward(self,x):

        input = x
        B, M, D, N = x.shape
        x = x.reshape(B,M*D,N)
        x = self.dw(x)
        x = x.reshape(B,M,D,N)
        x = x.reshape(B*M,D,N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        x = input + x
        return x


class Stage(nn.Module):

    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, dw_model, nvars, small_kernel_merged=False, drop=0.1):

        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for i in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, nvars=nvars, small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class ModernTCN(nn.Module):

    def __init__(self, d_input, d_output, d_hidden, n_input, n_blocks=4):
        super().__init__()

        # defaults
        self.patch_size = 8 # default, run.py=16, scripts=8
        self.patch_stride = 4 # default, run.py=8, scripts=4
        self.stem_ratio = 6 # default, run.py= 6
        self.downsample_ratio = 2 # default, run.py= 2
        self.upsample_ratio = self.downsample_ratio
        self.ffn_ratio = 4 # default, run.py= 2, scripts= 8
        self.large_size = [31, 31, 31, 31] # default, run.py= [31, 29, 27, 13], scripts= 51
        self.small_size = [5, 5, 5, 5] # default, run.py= [5, 5, 5, 5], scripts= 5
        self.num_blocks = [1, 1, 1, 1] # default, run.py= [1, 1, 1, 1]
        self.dims = [32, 32, 32, 32] # default, run.py= [256, 256, 256, 256], scripts= [64, 64, 64, 64]
        self.dw_dims = [128, 128, 128, 128] # default, run.py= [256, 256, 256, 256]
        self.nvars = n_input
        self.small_kernel_merged = False # default, scripts= False
        self.backbone_dropout = 0.5
        self.head_dropout = 0.0

        # instance normalization
        self.instance_norm = RIN(n_input)

        # downsampling
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, self.dims[0], kernel_size=self.patch_size, stride=self.patch_stride),
            nn.BatchNorm1d(self.dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm1d(self.dims[i]),
                nn.Conv1d(self.dims[i], self.dims[i + 1], kernel_size=self.downsample_ratio, stride=self.downsample_ratio),
            )
            self.downsample_layers.append(downsample_layer)

        # backbone
        self.num_stage = 2 # WTF?
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(
                self.ffn_ratio,
                self.num_blocks[stage_idx],
                self.large_size[stage_idx],
                self.small_size[stage_idx],
                dmodel=self.dims[stage_idx],
                dw_model=self.dw_dims[stage_idx],
                nvars=self.nvars,
                small_kernel_merged=self.small_kernel_merged,
                drop=self.backbone_dropout
            )
            self.stages.append(layer)

        self.lat_layer = nn.ModuleList()
        self.smooth_layer = nn.ModuleList()
        self.up_sample_conv = nn.ModuleList()

        for i in range(self.num_stage):
            align_dim = self.dims[-1]
            lat = nn.Conv1d(self.dims[i], align_dim, kernel_size=1, stride=1)
            self.lat_layer.append(lat)
            smooth = nn.Conv1d(align_dim, align_dim, kernel_size=3, stride=1, padding=1)
            self.smooth_layer.append(smooth)
            up_conv = nn.Sequential(
                nn.ConvTranspose1d(align_dim, align_dim, kernel_size=self.upsample_ratio, stride=self.upsample_ratio),
                nn.BatchNorm1d(align_dim)
            )
            self.up_sample_conv.append(up_conv)

        # projection
        patch_num = d_input // self.patch_stride

        self.n_vars = n_input
        d_model = self.dims[-1]

        if patch_num % pow(self.downsample_ratio,(self.num_stage - 1)) == 0:
            self.head_nf = d_model * patch_num // pow(self.downsample_ratio,(self.num_stage - 1))
        else:
            self.head_nf = d_model * (patch_num // pow(self.downsample_ratio, (self.num_stage - 1))+1)

        self.projection = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.head_nf, d_output),
            nn.Dropout(self.head_dropout)
        )

    def model(self, X):

        B, M, L = X.shape

        X = X.unsqueeze(-2)
        for i in range(self.num_stage):
            B, M, D, N = X.shape
            X = X.reshape(B * M, D, N)

            if i==0:
                if self.patch_size != self.patch_stride:
                    # stem layer padding
                    pad_len = self.patch_size - self.patch_stride
                    pad = X[:,:,-1:].repeat(1, 1, pad_len)
                    X = torch.cat([X, pad],dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    X = torch.cat([X, X[:, :, -pad_len:]], dim=-1)

            X = self.downsample_layers[i](X)
            _, D_, N_ = X.shape
            X = X.reshape(B, M, D_, N_)
            X = self.stages[i](X)
        return X


    def forward(self, X): # B, C, T
        X = self.instance_norm(X.transpose(1, 2), "norm").transpose(1, 2)
        X = self.model(X)
        X = self.projection(X)
        X = self.instance_norm(X.transpose(1, 2), "denorm").transpose(1, 2)
        return X


class ModernTCNHyper(nn.Module):

    def __init__(self, d_input, d_output, d_hidden, d_hyper_hidden, n_input, d_embedding=8, embedding=None, n_blocks=4):
        super().__init__()

        # defaults
        self.patch_size = 8 # default, run.py=16, scripts=8
        self.patch_stride = 4 # default, run.py=8, scripts=4
        self.stem_ratio = 6 # default, run.py= 6
        self.downsample_ratio = 2 # default, run.py= 2
        self.upsample_ratio = self.downsample_ratio
        self.ffn_ratio = 4 # default, run.py= 2, scripts= 8
        self.large_size = [31, 31, 31, 31] # default, run.py= [31, 29, 27, 13], scripts= 51
        self.small_size = [5, 5, 5, 5] # default, run.py= [5, 5, 5, 5], scripts= 5
        self.num_blocks = [1, 1, 1, 1] # default, run.py= [1, 1, 1, 1]
        self.dims = [32, 32, 32, 32] # default, run.py= [256, 256, 256, 256], scripts= [64, 64, 64, 64]
        self.dw_dims = [128, 128, 128, 128] # default, run.py= [256, 256, 256, 256]
        self.nvars = n_input
        self.small_kernel_merged = False # default, scripts= False
        self.backbone_dropout = 0.5
        self.head_dropout = 0.0

        # instance normalization
        self.instance_norm = RIN(n_input)

        # downsampling
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, self.dims[0], kernel_size=self.patch_size, stride=self.patch_stride),
            nn.BatchNorm1d(self.dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm1d(self.dims[i]),
                nn.Conv1d(self.dims[i], self.dims[i + 1], kernel_size=self.downsample_ratio, stride=self.downsample_ratio),
            )
            self.downsample_layers.append(downsample_layer)

        # backbone
        self.num_stage = 2 # WTF?
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(
                self.ffn_ratio,
                self.num_blocks[stage_idx],
                self.large_size[stage_idx],
                self.small_size[stage_idx],
                dmodel=self.dims[stage_idx],
                dw_model=self.dw_dims[stage_idx],
                nvars=self.nvars,
                small_kernel_merged=self.small_kernel_merged,
                drop=self.backbone_dropout
            )
            self.stages.append(layer)

        self.lat_layer = nn.ModuleList()
        self.smooth_layer = nn.ModuleList()
        self.up_sample_conv = nn.ModuleList()

        for i in range(self.num_stage):
            align_dim = self.dims[-1]
            lat = nn.Conv1d(self.dims[i], align_dim, kernel_size=1, stride=1)
            self.lat_layer.append(lat)
            smooth = nn.Conv1d(align_dim, align_dim, kernel_size=3, stride=1, padding=1)
            self.smooth_layer.append(smooth)
            up_conv = nn.Sequential(
                nn.ConvTranspose1d(align_dim, align_dim, kernel_size=self.upsample_ratio, stride=self.upsample_ratio),
                nn.BatchNorm1d(align_dim)
            )
            self.up_sample_conv.append(up_conv)

        # projection
        patch_num = d_input // self.patch_stride

        self.n_vars = n_input
        d_model = self.dims[-1]

        if patch_num % pow(self.downsample_ratio,(self.num_stage - 1)) == 0:
            self.head_nf = d_model * patch_num // pow(self.downsample_ratio,(self.num_stage - 1))
        else:
            self.head_nf = d_model * (patch_num // pow(self.downsample_ratio, (self.num_stage - 1))+1)

        self.projection_hyper = nn.Sequential(
            nn.Flatten(start_dim=-2),
            HyperLinearLayer(self.head_nf, d_output, d_hyper_hidden, n_input, d_embedding, embedding),
            nn.Dropout(self.head_dropout)
        )

    def model(self, X):

        B, M, L = X.shape

        X = X.unsqueeze(-2)
        for i in range(self.num_stage):
            B, M, D, N = X.shape
            X = X.reshape(B * M, D, N)

            if i==0:
                if self.patch_size != self.patch_stride:
                    # stem layer padding
                    pad_len = self.patch_size - self.patch_stride
                    pad = X[:,:,-1:].repeat(1, 1, pad_len)
                    X = torch.cat([X, pad],dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    X = torch.cat([X, X[:, :, -pad_len:]], dim=-1)

            X = self.downsample_layers[i](X)
            _, D_, N_ = X.shape
            X = X.reshape(B, M, D_, N_)
            X = self.stages[i](X)
        return X


    def forward(self, X): # B, C, T
        X = self.instance_norm(X.transpose(1, 2), "norm").transpose(1, 2)
        X = self.model(X)
        X = self.projection_hyper(X)
        X = self.instance_norm(X.transpose(1, 2), "denorm").transpose(1, 2)
        return X


class ModernTCNModel(NeuralModel):
    """Common evaluator adapter for the HN-MVTS ModernTCN Base network."""

    def __init__(self, context, horizon, channels, seed=0, device="cpu", **params):
        super().__init__("moderntcn", context, horizon, channels, seed=seed, device=device, **params)
