#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import ssl
import math
import random
import torch
import torch.nn as nn
import torchvision
import commons
from functools import partial
try:
    import pretrainedmodels
    AVAILABLE_MODEL_PRETRAINEDMODELS = True
except ModuleNotFoundError:
    AVAILABLE_MODEL_PRETRAINEDMODELS = False
try:
    from efficientnet_pytorch import EfficientNet
    from efficientnet_pytorch.utils import round_filters
    AVAILABLE_MODEL_EFFICIENTNET = True
except ModuleNotFoundError:
    AVAILABLE_MODEL_EFFICIENTNET = False
try:
    import resnest.torch as ResNeSt
    AVAILABLE_MODEL_RESNEST = True
except ModuleNotFoundError:
    AVAILABLE_MODEL_RESNEST = False
try:
    import timm
    AVAILABLE_MODEL_TIMM = True
except:
    AVAILABLE_MODEL_TIMM = False

ssl._create_default_https_context = ssl._create_unverified_context

""" Replace """
def ReplaceGroupNorm(model):
    for attr_str in dir(model):
        attr_obj = getattr(model, attr_str)
        if type(attr_obj) == torch.nn.BatchNorm1d:
            setattr(model, attr_str, torch.nn.GroupNorm(32, attr_obj.num_features))
        elif type(attr_obj) == torch.nn.BatchNorm2d:
            setattr(model, attr_str, torch.nn.GroupNorm(32, attr_obj.num_features))
        else:
            pass
    for _, c in model.named_children():
        ReplaceGroupNorm(c)
    return model

def ReplaceSyncBatchNorm(model):
    for attr_str in dir(model):
        attr_obj = getattr(model, attr_str)
        if type(attr_obj) == torch.nn.BatchNorm1d:
            setattr(model, attr_str, NN.BatchNorm1d(attr_obj.num_features))
        elif type(attr_obj) == torch.nn.BatchNorm2d:
            setattr(model, attr_str, NN.BatchNorm2d(attr_obj.num_features))
        else:
            pass
    for _, c in model.named_children():
        ReplaceGroupNorm(c)
    return model

""" Model Tool Kit """
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

@torch.jit.script
def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return mish(x)

def gem(x, p=3, eps=1e-6):
    return nn.functional.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

class DualPool2d(nn.Module):
    def __init__(self):
        super(DualPool2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        return torch.cat([
            self.avg_pool(x),
            self.max_pool(x)
        ], dim=1)

class TriplePool2d(nn.Module):
    def __init__(self):
        super(TriplePool2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.gem_pool = GeM()

    def forward(self, x):
        return torch.cat([
            self.avg_pool(x),
            self.max_pool(x),
            self.gem_pool(x)
        ], dim=1)

# referebce https://github.com/miguelvr/dropblock/blob/master/dropblock/dropblock.py
class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """
    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"
        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)
            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            # place mask on input device
            mask = mask.to(x.device)
            # compute block mask
            block_mask = self._compute_block_mask(mask)
            # apply block mask
            out = x * block_mask[:, None, :, :]
            # scale output
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        block_mask = nn.functional.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)

class _BatchInstanceNorm(nn.modules.batchnorm._BatchNorm):
    """
        reference:
        https://github.com/hyeonseobnam/Batch-Instance-Normalization/blob/master/models/batchinstancenorm.py
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchInstanceNorm, self).__init__(num_features, eps, momentum, affine)
        self.gate = nn.Parameter(torch.Tensor(num_features))
        self.gate.data.fill_(1)
        setattr(self.gate, 'bin_gate', True)

    def forward(self, input):
        self._check_input_dim(input)

        # Batch norm
        if self.affine:
            bn_w = self.weight * self.gate
        else:
            bn_w = self.gate
        out_bn = nn.functional.batch_norm(
            input, self.running_mean, self.running_var, bn_w, self.bias,
            self.training, self.momentum, self.eps)
        
        # Instance norm
        b, c  = input.size(0), input.size(1)
        if self.affine:
            in_w = self.weight * (1 - self.gate)
        else:
            in_w = 1 - self.gate
        input = input.view(1, b * c, *input.size()[2:])
        out_in = nn.functional.batch_norm(
            input, None, None, None, None,
            True, self.momentum, self.eps)
        out_in = out_in.view(b, c, *input.size()[2:])
        out_in.mul_(in_w[None, :, None, None])

        return out_bn + out_in

class BatchInstanceNorm1d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))

class BatchInstanceNorm2d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = nn.functional.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

class CustomSplAtConv2d(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(CustomSplAtConv2d, self).__init__()
        padding = nn.modules.utils._pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited) 
        else:
            gap = x
        gap = nn.functional.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()

class CustomBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(CustomBottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = CustomSplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=nn.BatchNorm2d,
                dropblock_prob=dropblock_prob)
        elif rectified_conv:
            from rfconv import RFConv2d
            self.conv2 = RFConv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False,
                average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)
        if self.avd and self.avd_first:
            out = self.avd_layer(out)
        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)
        if self.avd and not self.avd_first:
            out = self.avd_layer(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def CustomResNeSt50(pretrained=False, root='~/.encoding/models', dropblock_prob=0.0, norm_layer=nn.BatchNorm2d, **kwargs):
    model = ResNeSt.resnet.ResNet(CustomBottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, dropblock_prob=dropblock_prob, norm_layer=norm_layer, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(ResNeSt.resnest.resnest_model_urls['resnest50'], progress=True, check_hash=True), strict=False)
    return model

class DualHead(nn.Module):
    def __init__(self, in_channels, num_classes1, num_classes2):
        super(DualHead, self).__init__()
        self.head1 = nn.Linear(in_channels, num_classes1)
        self.head2 = nn.Linear(in_channels, num_classes2)

    def forward(self, x):
        y1 = self.head1(x)
        y2 = self.head2(x)
        return y1, y2

class MixHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MixHead, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveMaxPool2d(1)
        self.heads = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        u = self.pool1(x) + self.pool2(x)
        y = self.heads(u)
        return y

class L2NormalizedHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(L2NormalizedHead, self).__init__()
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        n = x.norm(p=2, dim=1, keepdim=True)
        x = x.div(n)
        x = self.head(x)
        return x

""" Custom Model for TILE """
class CustomModel(nn.Module):
    def __init__(self, name, num_classes, pretrained=True, in_channels=3, num_images=12, pooling='avg+max', headopt='v1'):
        super(CustomModel, self).__init__()
        if name in [ 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50', 'resnext101', 'wide_resnet50', 'wide_resnet101' ]:
            if name == 'resnext50' or name == 'resnext101':
                name = name + '_32x4d'
            elif name == 'wide_resnet50' or name == 'wide_resnet101':
                name = name + '_2'
            basemodel = getattr(torchvision.models, name)(pretrained='imagenet' if pretrained else None)
            if in_channels != 3:
                basemodel.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.base = nn.Sequential(*list(basemodel.children())[:-2])
            n_channel = list(basemodel.children())[-1].in_features
        elif AVAILABLE_MODEL_PRETRAINEDMODELS and name in [ 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50', 'se_resnext101' ]:
            if name == 'se_resnext50' or name == 'se_resnext101':
                name = name + '_32x4d'
            basemodel = getattr(pretrainedmodels, name)(pretrained='imagenet' if pretrained else None)
            if in_channels != 3:
                basemodel.layer0.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.base = nn.Sequential(*list(basemodel.children())[:-2])
            n_channel = list(basemodel.children())[-1].in_features
        elif AVAILABLE_MODEL_EFFICIENTNET and name.startswith('efficientnet-b'):
            basemodel = EfficientNet.from_pretrained(name) if pretrained else EfficientNet.from_name(name)
            if in_channels != 3:
                out_channels = round_filters(32, basemodel._global_params)
                basemodel._conv_stem = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
            self.base = basemodel
            n_channel = list(basemodel.children())[-2].in_features
        elif AVAILABLE_MODEL_RESNEST and name.startswith('resnest'):
            basemodel = getattr(ResNeSt, name)(pretrained=pretrained)
            if in_channels != 3:
                basemodel.conv1[0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.base = nn.Sequential(*list(basemodel.children())[:-2])
            n_channel = list(basemodel.children())[-1].in_features
        elif AVAILABLE_MODEL_RESNEST and name == 'bin-resnest50':
            basemodel = CustomResNeSt50(pretrained=pretrained, norm_layer=BatchInstanceNorm2d)
            if in_channels != 3:
                basemodel.conv1[0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.base = nn.Sequential(*list(basemodel.children())[:-2])
            n_channel = list(basemodel.children())[-1].in_features
        elif AVAILABLE_MODEL_RESNEST and name == 'dbs-resnest50':
            basemodel = CustomResNeSt50(pretrained=pretrained, dropblock_prob=0.2)
            if in_channels != 3:
                basemodel.conv1[0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.base = nn.Sequential(*list(basemodel.children())[:-2])
            n_channel = list(basemodel.children())[-1].in_features
        elif AVAILABLE_MODEL_RESNEST and name == 'dbs-bin-resnest50':
            basemodel = CustomResNeSt50(pretrained=pretrained, dropblock_prob=0.2, norm_layer=BatchInstanceNorm2d)
            if in_channels != 3:
                basemodel.conv1[0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.base = nn.Sequential(*list(basemodel.children())[:-2])
            n_channel = list(basemodel.children())[-1].in_features
        else:
            raise NameError('指定されたモデルは定義されていません (name={})'.format(name))
        pool_layer, pool_count = self._get_pooling_(pooling)
        if headopt == 'v3':
            self.head = nn.Sequential(
                pool_layer,
                Flatten(),
                DualHead(pool_count * n_channel, 1, 6)
            )
        elif headopt == 'v2':
            self.head = nn.Sequential(
                pool_layer,
                Flatten(),
                nn.Linear(pool_count * n_channel, num_classes),
            )
        else:
            self.head = nn.Sequential(
                pool_layer,
                Flatten(),
                nn.Linear(pool_count * n_channel, 512),
                Mish(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        self.num_images = num_images

    def _get_pooling_(self, pooling):
        pooling = pooling.lower()
        if pooling == 'avg+max':
            layer = DualPool2d()
            count = 2
        elif pooling == 'avg+max+gem':
            layer = TriplePool2d()
            count = 3
        elif pooling == 'avg':
            layer = nn.AdaptiveAvgPool2d(1)
            count = 1
        elif pooling == 'max':
            layer = nn.AdaptiveMaxPool2d(1)
            count = 1
        elif pooling == 'gem':
            layer = GeM()
            count = 1
        else:
            ValueError('指定されたプーリングレイヤーは定義されていません (pooling={})'.format(pooling))
        return layer, count

    def forward(self, x, **kwargs):
        # BxNxCxHxW to (BxN)xCxHxW:
        d = x.shape
        x = x.view(-1, d[2], d[3], d[4])
        # x := (BxN)xCxHxW
        x = self.base.extract_features(x, **kwargs) if isinstance(self.base, EfficientNet) else self.base(x, **kwargs)
        # x := (BxN)xCxh'xw'
        d = x.shape
        x = x.view(-1, self.num_images, d[1], d[2], d[3]).permute(0, 2, 1, 3, 4).contiguous().view(-1, d[1], d[2] * self.num_images, d[3])
        # x := BxCx(Nxh')xw'
        x = self.head(x)
        return x

class SIREN(nn.Module):
    def __init__(self):
        super(SIREN, self).__init__()

    def forward(self, x):
        return torch.sin(x)

def replaceSIREN(model):
    for attr_str in dir(model):
        attr_obj = getattr(model, attr_str)
        if type(attr_obj) == torch.nn.ReLU:
            setattr(model, attr_str, SIREN())
    for _, c in model.named_children():
        replaceSIREN(c)
    return model

""" Utils """
# モデル取得ユーティリティ
def GetModel(conf, num_classes=6, pretrained=True, in_channels=3, uri=None, map_loaction=None, strict=True, las_process=False):
    # 必須パラメータ:
    if 'model' not in conf:
        raise NameError('モデルが指定されていません (--model)')
    name = conf['model'].lower()
    # 任意パラメータ:
    # (なし)
    # パラメータ取得:
    fetch_dict = None
    if type(pretrained) == str:
        if pretrained.startswith('https://') or pretrained.startswith('http://') or pretrained.startswith('ftp://'):
            fetch_dict = torch.hub.load_state_dict_from_url(pretrained)
            pretrained = False
        elif os.path.exists(pretrained):
            fetch_dict = torch.load(pretrained, map_location=map_loaction)
            pretrained = False
        else:
            raise FileNotFoundError('指定されたパラメータは存在しません (pretrained={})'.format(pretrained))
    state_dict = None
    if uri is not None:
        if uri.startswith('https://') or uri.startswith('http://') or uri.startswith('ftp://'):
            state_dict = torch.hub.load_state_dict_from_url(uri)
            pretrained = False
        elif os.path.exists(uri):
            state_dict = torch.load(uri, map_location=map_loaction)
            pretrained = False
        else:
            raise FileNotFoundError('指定されたパラメータは存在しません (uri={})'.format(uri))
        # パラメータを処理する
        if las_process:
            partname = name.split('+')[-1]
            if partname in [ 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50', 'resnext101', 'wide_resnet50', 'wide_resnet101' ] or partname in ['senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50', 'se_resnext101' ] or name.startswith('resnest'):
                if 'fc.weight' in state_dict:
                    state_dict['fc.head1.weight'] = state_dict['fc.weight']
                    state_dict['fc.head2.weight'] = state_dict['fc.weight']
                    del state_dict['fc.weight']
                if 'fc.bias' in state_dict:
                    state_dict['fc.head1.bias'] = state_dict['fc.bias']
                    state_dict['fc.head2.bias'] = state_dict['fc.bias']
                    del state_dict['fc.bias']
            elif partname.startswith('efficientnet-b'):
                if '_fc.weight' in state_dict:
                    state_dict['_fc.head1.weight'] = state_dict['_fc.weight']
                    state_dict['_fc.head2.weight'] = state_dict['_fc.weight']
                    del state_dict['_fc.weight']
                if '_fc.bias' in state_dict:
                    state_dict['_fc.head1.bias'] = state_dict['_fc.bias']
                    state_dict['_fc.head2.bias'] = state_dict['_fc.bias']
                    del state_dict['_fc.bias']
            elif partname.startswith('regnet'):
                if 'head.fc.weight' in state_dict:
                    state_dict['head.fc.head1.weight'] = state_dict['head.fc.weight']
                    state_dict['head.fc.head2.weight'] = state_dict['head.fc.weight']
                    del state_dict['head.fc.weight']
                if 'head.fc.bias' in state_dict:
                    state_dict['head.fc.head1.bias'] = state_dict['head.fc.bias']
                    state_dict['head.fc.head2.bias'] = state_dict['head.fc.bias']
                    del state_dict['head.fc.bias']
    # Torchvision:
    if name in [ 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50', 'resnext101', 'wide_resnet50', 'wide_resnet101' ]:
        if name == 'resnext50' or name == 'resnext101':
            name = name + '_32x4d'
        elif name == 'wide_resnet50' or name == 'wide_resnet101':
            name = name + '_2'
        model = getattr(torchvision.models, name)(pretrained='imagenet' if pretrained else None)
        if fetch_dict is not None:
            model.load_state_dict(fetch_dict)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif name.startswith('gem+resnet'):
        mname = name[4:]
        model = getattr(torchvision.models, mname)(pretrained='imagenet' if pretrained else None)
        if fetch_dict is not None:
            model.load_state_dict(fetch_dict)
        model.avgpool = GeM()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # pretrained-models.pytorch:
    elif AVAILABLE_MODEL_PRETRAINEDMODELS and name in [ 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50', 'se_resnext101' ]:
        if name == 'se_resnext50' or name == 'se_resnext101':
            name = name + '_32x4d'
        model = getattr(pretrainedmodels, name)(pretrained='imagenet' if pretrained else None)
        if fetch_dict is not None:
            model.load_state_dict(fetch_dict)
        model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)
        if in_channels != 3:
            model.layer0.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif AVAILABLE_MODEL_PRETRAINEDMODELS and name in [ 'gem+senet154', 'gem+se_resnet50', 'gem+se_resnet101', 'gem+se_resnet152', 'gem+se_resnext50', 'gem+se_resnext101' ]:
        mname = name[4:]
        if mname == 'se_resnext50' or mname == 'se_resnext101':
            mname = mname + '_32x4d'
        model = getattr(pretrainedmodels, mname)(pretrained='imagenet' if pretrained else None)
        if fetch_dict is not None:
            model.load_state_dict(fetch_dict)
        model.avg_pool = GeM()
        model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)
        if in_channels != 3:
            model.layer0.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # efficientNet.pytorch:
    elif AVAILABLE_MODEL_EFFICIENTNET and name.startswith('efficientnet-b') and 0 <= int(name[len('efficientnet-b'):]) and int(name[len('efficientnet-b'):]) <= 7:
        model = EfficientNet.from_pretrained(name) if pretrained else EfficientNet.from_name(name)
        if fetch_dict is not None:
            model.load_state_dict(fetch_dict)
        model._fc = nn.Linear(model._fc.in_features, num_classes)
        if in_channels != 3:
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
    elif AVAILABLE_MODEL_EFFICIENTNET and name.startswith('gem+efficientnet-b') and 0 <= int(name[len('gem+efficientnet-b'):]) and int(name[len('gem+efficientnet-b'):]) <= 7:
        mname = name[4:]
        model = EfficientNet.from_pretrained(mname) if pretrained else EfficientNet.from_name(mname)
        if fetch_dict is not None:
            model.load_state_dict(fetch_dict)
        model._avg_pooling = GeM()
        model._fc = nn.Linear(model._fc.in_features, num_classes)
        if in_channels != 3:
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
    # resnest:
    elif AVAILABLE_MODEL_RESNEST and name.startswith('resnest'):
        model = getattr(ResNeSt, name)(pretrained=pretrained)
        if fetch_dict is not None:
            model.load_state_dict(fetch_dict)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if in_channels != 3:
            model.conv1[0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    elif AVAILABLE_MODEL_RESNEST and name.startswith('gem+resnest'):
        mname = name[4:]
        model = getattr(ResNeSt, mname)(pretrained=pretrained)
        if fetch_dict is not None:
            model.load_state_dict(fetch_dict)
        model.avgpool = GeM()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if in_channels != 3:
            model.conv1[0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # timm:
    elif AVAILABLE_MODEL_TIMM and (name.startswith('regnet') or name.startswith('hrnet_') or name.startswith('tresnet_') or name.startswith('tf_efficientnet_b') or name.startswith('dpn') or name.startswith('ese_vovnet')):
        # regnet
        # hrnet_*
        # tresnet_*
        # tf_efficientnet_b*_ap
        model = timm.create_model(model_name=name, num_classes=num_classes, in_chans=in_channels, pretrained=pretrained)
    elif AVAILABLE_MODEL_TIMM and name.startswith('timm-resnest'):
        # resnest
        mname = name[5:]
        model = timm.create_model(model_name=mname, num_classes=num_classes, in_chans=in_channels, pretrained=pretrained)
    elif AVAILABLE_MODEL_TIMM and name.startswith('norm2+regnet'):
        # regnet
        # hrnet_*
        # tresnet_*
        # tf_efficientnet_b*_ap
        mname = name[6:]
        model = timm.create_model(model_name=mname, num_classes=num_classes, in_chans=in_channels, pretrained=pretrained)
        model.head.fc = L2NormalizedHead(model.head.fc.in_features, num_classes)
    elif AVAILABLE_MODEL_TIMM and name.startswith('nlc+regnet'):
        # regnet
        mname = name[4:]
        model = timm.create_model(model_name=mname, num_classes=num_classes, in_chans=in_channels, pretrained=pretrained)
        model.head = MixHead(model.head.fc.in_features, num_classes)
    # SIREN:
    elif name == 'siren+resnet34':
        mname = name[6:]
        if mname in [ 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50', 'resnext101', 'wide_resnet50', 'wide_resnet101' ]:
            if mname == 'resnext50' or mname == 'resnext101':
                mname = mname + '_32x4d'
            elif mname == 'wide_resnet50' or name == 'wide_resnet101':
                mname = mname + '_2'
            model = getattr(torchvision.models, mname)(pretrained='imagenet' if pretrained else None)
            if fetch_dict is not None:
                model.load_state_dict(fetch_dict)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            if in_channels != 3:
                model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            model = None
        model = replaceSIREN(model)
    # label and autoscale:
    elif AVAILABLE_MODEL_PRETRAINEDMODELS and name in [ 'las+senet154', 'las+se_resnet50', 'las+se_resnet101', 'las+se_resnet152', 'las+se_resnext50', 'las+se_resnext101' ]:
        mname = name[4:]
        if mname == 'se_resnext50' or mname == 'se_resnext101':
            mname = mname + '_32x4d'
        model = getattr(pretrainedmodels, mname)(pretrained='imagenet' if pretrained else None)
        if fetch_dict is not None:
            model.load_state_dict(fetch_dict)
        model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        model.last_linear = DualHead(model.last_linear.in_features, num_classes, num_classes)
        if in_channels != 3:
            model.layer0.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif AVAILABLE_MODEL_PRETRAINEDMODELS and name in [ 'las+gem+senet154', 'las+gem+se_resnet50', 'las+gem+se_resnet101', 'las+gem+se_resnet152', 'las+gem+se_resnext50', 'las+gem+se_resnext101' ]:
        mname = name[8:]
        if mname == 'se_resnext50' or mname == 'se_resnext101':
            mname = mname + '_32x4d'
        model = getattr(pretrainedmodels, mname)(pretrained='imagenet' if pretrained else None)
        if fetch_dict is not None:
            model.load_state_dict(fetch_dict)
        model.avg_pool = GeM()
        model.last_linear = DualHead(model.last_linear.in_features, num_classes, num_classes)
        if in_channels != 3:
            model.layer0.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif AVAILABLE_MODEL_EFFICIENTNET and name.startswith('las+efficientnet-b') and 0 <= int(name[len('las+efficientnet-b'):]) and int(name[len('las+efficientnet-b'):]) <= 7:
        mname = name[4:]
        model = EfficientNet.from_pretrained(mname) if pretrained else EfficientNet.from_name(mname)
        if fetch_dict is not None:
            model.load_state_dict(fetch_dict)
        model._fc = DualHead(model._fc.in_features, num_classes, num_classes)
        if in_channels != 3:
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
    elif AVAILABLE_MODEL_EFFICIENTNET and name.startswith('las+gem+efficientnet-b') and 0 <= int(name[len('las+gem+efficientnet-b'):]) and int(name[len('las+gem+efficientnet-b'):]) <= 7:
        mname = name[8:]
        model = EfficientNet.from_pretrained(mname) if pretrained else EfficientNet.from_name(mname)
        if fetch_dict is not None:
            model.load_state_dict(fetch_dict)
        model._avg_pooling = GeM()
        model._fc = DualHead(model._fc.in_features, num_classes, num_classes)
        if in_channels != 3:
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
    elif AVAILABLE_MODEL_RESNEST and name.startswith('las+resnest'):
        mname = name[4:]
        model = getattr(ResNeSt, mname)(pretrained=pretrained)
        if fetch_dict is not None:
            model.load_state_dict(fetch_dict)
        model.fc = DualHead(model.fc.in_features, num_classes, num_classes)
        if in_channels != 3:
            model.conv1[0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    elif AVAILABLE_MODEL_RESNEST and name.startswith('las+gem+resnest'):
        mname = name[8:]
        model = getattr(ResNeSt, mname)(pretrained=pretrained)
        if fetch_dict is not None:
            model.load_state_dict(fetch_dict)
        model.avgpool = GeM()
        model.fc = DualHead(model.fc.in_features, num_classes, num_classes)
        if in_channels != 3:
            model.conv1[0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    elif AVAILABLE_MODEL_TIMM and name.startswith('las+regnet'):
        mname = name[4:]
        model = timm.create_model(model_name=mname, num_classes=num_classes, in_chans=in_channels, pretrained=True)
        model.head.fc = DualHead(model.head.fc.in_features, num_classes, num_classes)
    # CustomModel (TILE):
    elif name.startswith('cls_p1_h2_') or name.startswith('reg_p1_h2_'):
        num_images = 12
        if 'num_images' in conf:
            num_images = conf['num_images']
        model = CustomModel(name[10:], num_classes, pretrained=pretrained, num_images=num_images, pooling='gem', headopt='v2')
    elif name.startswith('mix_p3_h3_'):
        num_images = 12
        if 'num_images' in conf:
            num_images = conf['num_images']
        model = CustomModel(name[10:], num_classes, pretrained=pretrained, num_images=num_images, pooling='avg+max+gem', headopt='v3')
    elif name.startswith('mix_p2_h3_'):
        num_images = 12
        if 'num_images' in conf:
            num_images = conf['num_images']
        model = CustomModel(name[10:], num_classes, pretrained=pretrained, num_images=num_images, pooling='avg+max', headopt='v3')
    elif name.startswith('cls_p3_h2_') or name.startswith('reg_p3_h2_'):
        num_images = 12
        if 'num_images' in conf:
            num_images = conf['num_images']
        model = CustomModel(name[10:], num_classes, pretrained=pretrained, num_images=num_images, pooling='avg+max+gem', headopt='v2')
    elif name.startswith('cls_p2_h2_') or name.startswith('reg_p2_h2_'):
        num_images = 12
        if 'num_images' in conf:
            num_images = conf['num_images']
        model = CustomModel(name[10:], num_classes, pretrained=pretrained, num_images=num_images, pooling='avg+max', headopt='v2')
    elif name.startswith('cls_p3_h1_') or name.startswith('reg_p3_h1_'):
        num_images = 12
        if 'num_images' in conf:
            num_images = conf['num_images']
        model = CustomModel(name[10:], num_classes, pretrained=pretrained, num_images=num_images, pooling='avg+max+gem')
    elif name.startswith('cls_p2_h2_') or name.startswith('reg_p2_h2_'):
        num_images = 12
        if 'num_images' in conf:
            num_images = conf['num_images']
        model = CustomModel(name[10:], num_classes, pretrained=pretrained, num_images=num_images, pooling='avg+max', headopt='v2')
    elif name.startswith('cls_v3_') or name.startswith('reg_v3_'):
        num_images = 12
        if 'num_images' in conf:
            num_images = conf['num_images']
        model = CustomModel(name[7:], num_classes, pretrained=pretrained, num_images=num_images, pooling='avg+max+gem')
        model = ReplaceGroupNorm(model)
    elif name.startswith('cls_v2_') or name.startswith('reg_v2_'):
        num_images = 12
        if 'num_images' in conf:
            num_images = conf['num_images']
        model = CustomModel(name[7:], num_classes, pretrained=pretrained, num_images=num_images, pooling='avg+max+gem')
    elif name.startswith('cls_v1_') or name.startswith('reg_v1_'):
        num_images = 12
        if 'num_images' in conf:
            num_images = conf['num_images']
        model = CustomModel(name[7:], num_classes, pretrained=pretrained, num_images=num_images)
    # error:
    else:
        raise NameError('指定された名前のモデルは定義されていません (--model={})'.format(name))
    # パラメータ読込:
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=strict)
    return model

if __name__ == '__main__':
    b = 4
    t = 16
    c = 3
    h = 128
    w = 128
    x = torch.zeros([b, c, h, w])
    # model = CustomModel('bin-resnest50', 1, num_images=t, pooling='avg+max+gem', headopt='v2')
    # model = GetModel({ 'model': 'regnety_032' }, 1, uri='pth/reg-22017798-88.pth', las_process=True)
    model = GetModel({ 'model': 'timm-resnest101e' }, pretrained='resnest101-22405ba7.pth')
    print(model)
    print(model.fc.in_features)
    # print('[1] forward dummy input:')
    y = model(x)
    # print('[1] forward dummy input: complete')
