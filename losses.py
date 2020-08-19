#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn

""" Loss Function """
class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, input, target):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

class LogCoshLoss(CustomLoss):
    def __init__(self, eps=1e-8):
        super(LogCoshLoss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        x = input - target
        # return torch.mean(torch.log(torch.cosh(x + self.eps)))
        return torch.log(torch.cosh(x + self.eps))

    def step(self):
        pass

class HingeMSELoss(CustomLoss):
    def __init__(self, margin=0.25):
        super(HingeMSELoss, self).__init__()
        self.margin = 0.25

    def forward(self, input, target):
        # return torch.mean((torch.abs(input - target) - self.margin) ** 2)
        return (torch.abs(input - target) - self.margin) ** 2

    def step(self):
        pass

class OHEMLoss(CustomLoss):
    def __init__(self, loss, decend=0.01, minval=0.7):
        super(OHEMLoss, self).__init__()
        self.loss = loss(reduction='none')
        self.rate = 1.0
        self.decend = decend
        self.minval = minval

    def forward(self, input, target):
        value = self.loss(input, target)
        if self.rate == 1:
            return torch.mean(value)
        else:
            rets, _ = torch.topk(value, int(self.rate * value.size()[0]), dim=0)
            return torch.mean(rets)

    def step(self):
        if self.rate > self.minval:
            self.rate = max(self.minval, self.rate - self.decend)

class WeakLoss(CustomLoss):
    def __init__(self, loss, ignore=0.02):
        super(WeakLoss, self).__init__()
        self.loss = loss(reduction='none')
        self.rate = (1.0 - ignore)

    def forward(self, input, target):
        value = self.loss(input, target)
        if self.rate >= 1.0:
            return torch.mean(value)
        else:
            rets, _ = torch.topk(value, int(self.rate * value.size()[0]), dim=0, largest=False)
            return torch.mean(rets)

    def step(self):
        pass

class L4Loss(CustomLoss):
    def __init__(self, reduction='mean'):
        super(L4Loss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        value = (input - target) ** 4
        if self.reduction != 'none':
            return torch.mean(value) if self.reduction == 'mean' else torch.sum(value)
        return value

    def step(self):
        pass

def GetTrainLossFunction(conf):
    # 必須パラメータ:
    if 'loss' not in conf:
        raise NameError('損失関数が指定されていません (--loss)')
    name = conf['loss'].lower()
    if name == 'mse':
        loss_fn = torch.nn.MSELoss(reduction='none')
    elif name == 'l1':
        loss_fn = torch.nn.L1Loss(reduction='none')
    elif name == 'l4':
        loss_fn = L4Loss(reduction='none')
    elif name == 'smoothl1':
        loss_fn = torch.nn.SmoothL1Loss(reduction='none')
    elif name == 'logcosh':
        loss_fn = LogCoshLoss()
    elif name == 'hmse':
        loss_fn = HingeMSELoss()
    elif name == 'bce':
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    elif name == 'ce':
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    elif name == 'ohem+mse':
        loss_fn = OHEMLoss(loss=nn.MSELoss)
    elif name == 'weak+mse':
        loss_fn = WeakLoss(loss=torch.nn.MSELoss)
    elif name == 'weak+smoothl1':
        loss_fn = WeakLoss(loss=torch.nn.SmoothL1Loss)
    else:
        raise NameError('指定された損失関数は定義されていません (--loss={})'.format(name))
    return loss_fn

def GetValidLossFunction(conf):
    # 必須パラメータ:
    if 'loss' not in conf:
    #     print('validation lossにはMSEが使用されます.')
    #     return nn.MSELoss()
        raise NameError('損失関数が指定されていません (--loss)')
    name = conf['loss'].lower()
    if name == 'mse' or name == 'weak+mse':
        loss_fn = nn.MSELoss()
    elif name == 'smoothl1' or name == 'weak+smoothl1':
        loss_fn = torch.nn.SmoothL1Loss()
    elif name == 'l4':
        loss_fn = L4Loss()
    elif name == 'bce':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif name == 'ce':
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        print('validation lossにはMSEが使用されます.')
        loss_fn = nn.MSELoss()
    return loss_fn