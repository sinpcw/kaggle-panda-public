#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import math
import torch
import torch.optim
import torchvision
from torch.optim.lr_scheduler import _LRScheduler
try:
    from adabound import AdaBound
    AVAILABLE_OPTIM_ADABOUND = True
except ModuleNotFoundError:
    AVAILABLE_OPTIM_ADABOUND = False
try:
    from radam import RAdam
    AVAILABLE_OPTIM_RADAM = True
except ModuleNotFoundError:
    AVAILABLE_OPTIM_RADAM = False

# 最適化エンジン取得ユーティリティ
def GetOptimizer(conf, parameter, **kwargs):
    # 必須パラメータ:
    if 'optimizer' not in conf:
        raise NameError('オプティマイザが指定されていません (--optimizer)')
    name = conf['optimizer'].lower()
    if 'lr' not in conf:
        conf['lr'] = 1e-3
    lr = conf['lr']
    # 任意パラメータ:
    option = { }
    if 'weight_decay' in conf:
        option['weight_decay'] = conf['weight_decay']
    # オプティマイザ選択:
    if name == 'sgd':
        if 'momentum' in conf:
            option['momentum'] = conf['momentum']
        if 'nesterov' in conf:
            option['nesterov'] = conf['momentum']
        optim = torch.optim.SGD(parameter, lr=lr, **option)
    elif name == 'adam':
        optim = torch.optim.Adam(parameter, lr=lr, **option)
    elif name == 'adadelta':
        optim = torch.optim.Adadelta(parameter, lr=lr, **option)
    elif name == 'adagrad':
        optim = torch.optim.Adagrad(parameter, lr=lr, **option)
    elif name == 'adamw':
        optim = torch.optim.AdamW(parameter, lr=lr, **option)
    elif name == 'adamax':
        optim = torch.optim.Adamax(parameter, lr=lr, **option)
    elif name == 'adabound' and AVAILABLE_OPTIM_ADABOUND:
        optim = AdaBound(parameter, lr=lr, **option)
    elif name == 'radam' and AVAILABLE_OPTIM_RADAM:
        optim = RAdam(parameter, lr=lr, **option)
    else:
        raise NameError('指定された名前のオプティマイザは定義されていません (--optimizer={})'.format(name))
    return optim

# オプティマイザの学習率を取得
def GetOptimierLR(optim):
    if len(optim.param_groups) > 1:
        return { 'lr_{}'.format(k): v['lr'] for k, v in optim.param_groups.items() }
    else:
        return optim.param_groups[0]['lr']

# スケジューラー取得ユーティリティ
def GetScheduler(conf, optim, **kwargs):
    # 必須パラメータ:
    if 'scheduler' not in conf:
        raise NameError('スケジューラが指定されていません (--scheduler)')
    name = conf['scheduler'].lower()
    # 任意パラメータ:
    option = { }
    # (なし)
    # スケジューラ選択:
    if name in [ 'null', 'none', 'fixed', 'const', 'constant' ]:
        if 'warmup' in conf:
            option['warmup'] = conf['warmup']
        scheduler = ConstantScheduler(optim, **option)
        require_call_everystep = False
    elif name == 'steplr' or name == 'multisteplr':
        if 'gamma' in conf:
            option['gamma'] = conf['gamma']
        if 'milestones' in conf:
            option['milestones'] = [ int(v.strip()) for v in conf['milestones'].split(',') ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, **option)
        require_call_everystep = False
    elif name == 'onecycle':
        if 'lr' not in kwargs:
            raise ValueError('学習率の指定が必要です (lr).')
        if 'epoch' not in kwargs:
            raise ValueError('epoch数の指定が必要です (epoch).')
        if 'steps' not in kwargs:
            raise ValueError('epoch毎のステップ数の指定が必要です (steps).')
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=kwargs['lr'], total_steps=kwargs['steps'], epochs=kwargs['epoch'])
        require_call_everystep = True
    elif name == 'cycle':
        if 'min_lr' not in kwargs:
            raise ValueError('最低学習率の指定が必要です (min_lr).')
        if 'max_lr' not in kwargs:
            raise ValueError('最大学習率の指定が必要です (max_lr).')
        scheduler = torch.optim.lr_scheduler.CyclicLR(optim, kwargs['min_lr'], kwargs['max_lr'], step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False)
        require_call_everystep = True
    elif name == 'dt':
        if 'min_lr' not in kwargs:
            raise ValueError('最低学習率の指定が必要です (min_lr).')
        if 'max_lr' not in kwargs:
            raise ValueError('最大学習率の指定が必要です (max_lr).')
        option = {
            'lr_min' : kwargs['min_lr'],
            'lr_max' : kwargs['max_lr'],
            'gamma' : kwargs['gamma'] if kwargs['gamma'] > 0 else 1.0,
            'pitch' : 2000
        }
        scheduler = DecayTriangleLR(optim, **option)
        require_call_everystep = True
    elif name == 'cosineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, **option)
        require_call_everystep = False
    elif name == 'cosineannealingwarmrestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, **option)
        require_call_everystep = False
    elif name == 'lineardecay':
        if 'warmup' in conf:
            option['warmup'] = conf['warmup']
        if 'finish' in conf:
            option['finish'] = conf['finish']
        scheduler = LinearDecayScheduler(optim, **option)
        require_call_everystep = False
    elif name == 'cosinedecay':
        if 'warmup' in conf:
            option['warmup'] = conf['warmup']
        if 'finish' in conf:
            option['finish'] = conf['finish']
        if 'lr_min' in conf:
            option['lr_min'] = conf['lr_min']
        scheduler = CosineDecayScheduler(optim, **option)
        require_call_everystep = False
    else:
        raise NameError('指定された名前のスケジューラは定義されていません (--scheduler={})'.format(name))
    return scheduler, require_call_everystep

class ConstantScheduler(_LRScheduler):
    """
    ConstantScheduler: warmupを除いて学習率を操作しない定数スケジューラ
    """
    def __init__(self, optimizer, last_epoch=-1, **kwargs):
        self.epoch_warmup = kwargs['warmup'] if 'warmup' in kwargs else 0
        super(ConstantScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if 0 < self.epoch_warmup and self.last_epoch < self.epoch_warmup:
            ratio = self.last_epoch / self.epoch_warmup
            return [ lr * ratio for lr in self.base_lrs ]
        return self.base_lrs

class LinearDecayScheduler(_LRScheduler):
    """
    LinearDecayScheduler: warmupを除いて学習率を線形で減衰させるスケジューラ
    """
    def __init__(self, optimizer, last_epoch=-1, **kwargs):
        self.lr_min = kwargs['lr_min'] if 'lr_min' in kwargs else 0
        self.epoch_warmup = kwargs['warmup'] if 'warmup' in kwargs else 0
        self.epoch_finish = kwargs['finish'] if 'finish' in kwargs else 100
        # parameter check
        if self.lr_min < 0:
            self.lr_min = 0
        if self.epoch_warmup < 0:
            self.epoch_warmup = 0
        if self.epoch_finish <= self.epoch_warmup:
            self.epoch_finish = self.epoch_warmup
        super(LinearDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if 0 < self.epoch_warmup and self.last_epoch < self.epoch_warmup:
            ratio = self.last_epoch / self.epoch_warmup
            return [ max(self.lr_min, lr * ratio) for lr in self.base_lrs ]
        elif self.last_epoch <= self.epoch_finish:
            ratio = 1.0 - (self.last_epoch - self.epoch_warmup) / (self.epoch_finish - self.epoch_warmup)
            return [ max(self.lr_min, lr * ratio) for lr in self.base_lrs ]
        return [ self.lr_min ] * len(self.base_lrs)

class CosineDecayScheduler(_LRScheduler):
    """
    CosineDecayScheduler: cosine関数による学習率減衰を実施するスケジューラ
    """
    def __init__(self, optimizer, last_epoch=-1, **kwargs):
        self.lr_min = kwargs['lr_min'] if 'lr_min' in kwargs else 0
        self.epoch_warmup = kwargs['warmup'] if 'warmup' in kwargs else 0
        self.epoch_finish = kwargs['finish'] if 'finish' in kwargs else 100
        # parameter check
        if self.lr_min < 0:
            self.lr_min = 0
        if self.epoch_warmup < 0:
            self.epoch_warmup = 0
        if self.epoch_finish <= self.epoch_warmup:
            self.epoch_finish = self.epoch_warmup
        super(CosineDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if 0 < self.epoch_warmup and self.last_epoch < self.epoch_warmup:
            ratio = math.cos(0.5 * math.pi * ((self.epoch_warmup - self.last_epoch) / self.epoch_warmup))
            return [ max(self.lr_min, lr * ratio) for lr in self.base_lrs ]
        elif self.last_epoch <= self.epoch_finish:
            ratio = math.cos(0.5 * math.pi * (self.last_epoch - self.epoch_warmup) / (self.epoch_finish - self.epoch_warmup))
            return [ max(self.lr_min, lr * ratio) for lr in self.base_lrs ]
        return [ self.lr_min ] * len(self.base_lrs)

class DecayTriangleLR(_LRScheduler):
    """
    DecayCycleLR: 
    """
    def __init__(self, optimizer, last_epoch=-1, **kwargs):
        self.lr_min = kwargs['lr_min'] if 'lr_min' in kwargs else 0
        self.lr_max = kwargs['lr_max'] if 'lr_max' in kwargs else 0
        self.gamma = kwargs['gamma'] if 'gamma' in kwargs else 0
        self.pitch = kwargs['pitch'] if 'pitch' in kwargs else 2000
        self.steps = 0
        # parameter check
        if self.lr_min < 0:
            self.lr_min = 0
        super(DecayTriangleLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        n, r = divmod(self.steps, self.pitch)
        v = (self.lr_max - self.lr_min) * self.gamma ** n
        self.steps += 1
        return [ self.lr_min + v * abs(1.0 - 2.0 * (r / self.pitch)) for lr in self.base_lrs ]

if __name__ == '__main__':
    mdl = torchvision.models.resnet.resnet18()
    opm = torch.optim.SGD(mdl.parameters(), lr=1e-2)
    sch, req = GetScheduler({ 'scheduler': 'dt' }, opm, max_lr=1e-2, min_lr=2e-4, gamma=0.95)
    for _ in range(1000):
        for i in range(1000):
            opm.zero_grad()
            opm.step()
        sch.step()
        print('lr = {}'.format(sch.get_lr()))