#!/usr/bin/bash
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
import random
import torch
from openslide import OpenSlide
from torch.utils.data.dataset import Dataset
from torchvision import utils
from torchvision import transforms
import albumentations as A
import augments
import skimage.io
import glob
from commons import IsDistributed

# 学習オーグメンテーション
def GetTrainAugment(exec, name, **kwargs):
    name = name.lower() if name is not None else 'default'
    augs = None
    if exec in [ 'cls', 'reg', 'mix', 'reg+las', 'cls+las' ]:
        if name in [ 'default', 'case1' ]:
            augs = augments.case1_cls_train_augs(name, **kwargs)
        if name in [ 'case0' ]:
            augs = augments.case0_cls_train_augs(name, **kwargs)
        elif name == 'case2':
            augs = augments.case2_cls_train_augs(name, **kwargs)
        elif name == 'case3':
            augs = augments.case3_cls_train_augs(name, **kwargs)
        elif name == 'case4':
            augs = augments.case4_cls_train_augs(name, **kwargs)
        elif name == 'case5':
            augs = augments.case5_cls_train_augs(name, **kwargs)
        elif name == 'subcase1' or name == 'scase1':
            augs = augments.subcase1_cls_train_augs(name, **kwargs)
        elif name == 'subcase2' or name == 'scase2':
            augs = augments.subcase2_cls_train_augs(name, **kwargs)
        elif name == 'subcase3' or name == 'scase3':
            augs = augments.subcase3_cls_train_augs(name, **kwargs)
        elif name == 'subcase4' or name == 'scase4':
            augs = augments.subcase4_cls_train_augs(name, **kwargs)
        elif name == 'subcase5' or name == 'scase5':
            augs = augments.subcase5_cls_train_augs(name, **kwargs)
        elif name == 'icase0':
            augs = augments.icase0_augs(name, **kwargs)
        elif name == 'icase1':
            augs = augments.icase1_augs(name, **kwargs)
        elif name == 'acase0':
            augs = augments.acase0_augs(name, **kwargs)
        elif name == 'acase1':
            augs = augments.acase1_augs(name, **kwargs)
        elif name == 'acase2':
            augs = augments.acase2_augs(name, **kwargs)
    else:
        pass
    if augs is None:
        raise NameError('指定したオーグメンテーションは定義されていません (--train_augs={})'.format(name))
    return augs

# 検証オーグメンテーション
def GetValidAugment(exec, name, **kwargs):
    name = name.lower() if name is not None else 'default'
    augs = None
    if exec in [ 'cls', 'reg', 'mix', 'reg+las', 'cls+las' ]:
        if name in [ 'default', 'case1', 'case2' ]:
            augs = augments.case1_cls_valid_augs(name, **kwargs)
        elif name == 'acase0':
            augs = augments.acase0_augs(name, **kwargs)
        elif name == 'acase1':
            augs = augments.acase1_augs(name, **kwargs)
        elif name == 'acase2':
            augs = augments.acase2_augs(name, **kwargs)
    else:
        pass
    if augs is None:
        raise NameError('指定したオーグメンテーションは定義されていません (--valid_augs={})'.format(name))
    return augs

# 学習データローダー
def GetTrainDataLoader(conf, run, train_df, augment_op=None, combine_op=None, patched_op=None):
    if run in [ 'reg', 'cls' ]:
        name = conf['{}_loader'.format(run)].lower()
        if name in [ 'default', 'fmt1', 'cmb0' ]:
            loader = GetFmt1TrainDataLoader(run, train_df, conf['image_dir'], conf['num_images'], conf['batch'], augment_op=augment_op, combine_op=combine_op, num_cols=4, num_workers=conf['num_workers'])
        elif name.startswith('auto-'):
            value = int(name[5:])
            loader = GetAutoTileTrainDataLoader(run, train_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, conf['batch'], augment_op=augment_op, combine_op=combine_op, jitter=conf['jitter'], num_workers=conf['num_workers'])
        elif name.startswith('autoscale-'):
            value = int(name[10:])
            loader = GetAutoTileTrainDataLoader(run, train_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, conf['batch'], augment_op=augment_op, combine_op=combine_op, auto_scale=True, rand_scale=True, jitter=conf['jitter'], num_workers=conf['num_workers'])
        elif name.startswith('dsf-'):
            value = int(name[4:])
            loader = GetASSTrainDataLoader(run, train_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, conf['batch'], augment_op=augment_op, combine_op=combine_op, patched_op=patched_op, auto_scale=True, rand_scale=True, kr_factor=True, jitter=conf['jitter'], num_workers=conf['num_workers'])
        elif name.startswith('lay-'):
            value = int(name[4:])
            loader = GetASSTrainDataLoader(run, train_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, conf['batch'], augment_op=augment_op, combine_op=combine_op, patched_op=patched_op, auto_scale=True, rand_scale=True, kr_layer=True, jitter=conf['jitter'], num_workers=conf['num_workers'])
        elif name.startswith('dsl-'):
            value = int(name[4:])
            loader = GetASSTrainDataLoader(run, train_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, conf['batch'], augment_op=augment_op, combine_op=combine_op, patched_op=patched_op, auto_scale=True, rand_scale=True, kr_factor=True, kr_layer=True, jitter=conf['jitter'], num_workers=conf['num_workers'])
        elif name.startswith('ass-'):
            value = int(name[4:])
            loader = GetASSTrainDataLoader(run, train_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, conf['batch'], augment_op=augment_op, combine_op=combine_op, patched_op=patched_op, auto_scale=True, rand_scale=True, jitter=conf['jitter'], num_workers=conf['num_workers'])
        elif name.startswith('asd-'):
            value = int(name[4:])
            loader = GetASSTrainDataLoader(run, train_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, conf['batch'], augment_op=augment_op, combine_op=combine_op, patched_op=patched_op, auto_scale=True, rand_scale=True, diff_stride=True, jitter=conf['jitter'], num_workers=conf['num_workers'])
        elif name.startswith('asm-'):
            value = int(name[4:])
            loader = GetASSTrainDataLoader(run, train_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, conf['batch'], augment_op=augment_op, combine_op=combine_op, patched_op=patched_op, auto_scale=True, rand_scale=True, blending=True, jitter=conf['jitter'], num_workers=conf['num_workers'])
        elif name.startswith('asr-'):
            value = int(name[4:])
            loader = GetASSTrainDataLoader(run, train_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, conf['batch'], augment_op=augment_op, combine_op=combine_op, patched_op=patched_op, auto_scale=True, rand_scale=True, randomize=True, jitter=conf['jitter'], num_workers=conf['num_workers'])
        elif name.startswith('mfs-'):
            value = int(name[4:])
            loader = GetMFSTrainDataLoader(run, train_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, conf['batch'], augment_op=augment_op, combine_op=combine_op, patched_op=patched_op, auto_scale=True, rand_scale=True, jitter=conf['jitter'], num_workers=conf['num_workers'])
        elif name.startswith('wss-'):
            value = int(name[4:])
            loader = GetASSTrainDataLoader(run, train_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, conf['batch'], augment_op=augment_op, combine_op=combine_op, patched_op=patched_op, auto_scale=True, rand_scale=True, weighted=True, jitter=conf['jitter'], num_workers=conf['num_workers'])
        elif name.startswith('las-') or name.startswith('wls-'):
            value = int(name[4:])
            loader = GetLASTrainDataLoader(run, train_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, conf['batch'], augment_op=augment_op, combine_op=combine_op, auto_scale=True, rand_scale=True, jitter=conf['jitter'], num_workers=conf['num_workers'])
        else:
            raise NameError('指定されたデータローダーは定義されていません (--{}_loader={})'.format(run, name))
    else:
        raise NameError('指定された実行処理へのデータローダーは定義されていません (--run={})'.format(run))
    return loader

# 検証データローダー
def GetValidDataLoader(conf, run, valid_df, augment_op=None, patched_op=None):
    if run in [ 'reg', 'cls' ]:
        name = conf['{}_loader'.format(run)].lower()
        if name in [ 'default', 'fmt1', 'cmb0' ]:
            loader = GetFmt1ValidDataLoader(run, valid_df, conf['image_dir'], conf['num_images'], augment_op=augment_op, combine_op=None, num_cols=4, num_workers=conf['num_workers'])
        elif name.startswith('auto-'):
            value = int(name[5:])
            loader = GetAutoTileValidDataLoader(run, valid_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, augment_op=augment_op, combine_op=None, num_workers=conf['num_workers'])
        elif name.startswith('autoscale-'):
            value = int(name[10:])
            loader = GetAutoTileValidDataLoader(run, valid_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, augment_op=augment_op, combine_op=None, auto_scale=True, rand_scale=False, jitter=0, num_workers=conf['num_workers'])
        elif name.startswith('dsf-'):
            value = int(name[4:])
            loader = GetASSValidDataLoader(run, valid_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, augment_op=augment_op, combine_op=None, patched_op=patched_op, auto_scale=True, rand_scale=False, kr_factor=True, jitter=0, num_workers=conf['num_workers'])
        elif name.startswith('lay-'):
            value = int(name[4:])
            loader = GetASSValidDataLoader(run, valid_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, augment_op=augment_op, combine_op=None, patched_op=patched_op, auto_scale=True, rand_scale=False, kr_layer=True, jitter=0, num_workers=conf['num_workers'])
        elif name.startswith('dsl-'):
            value = int(name[4:])
            loader = GetASSValidDataLoader(run, valid_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, augment_op=augment_op, combine_op=None, patched_op=patched_op, auto_scale=True, rand_scale=False, kr_factor=True, kr_layer=True, jitter=0, num_workers=conf['num_workers'])
        elif name.startswith('asd-') :
            value = int(name[4:])
            loader = GetASSValidDataLoader(run, valid_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, augment_op=augment_op, combine_op=None, patched_op=patched_op, auto_scale=True, rand_scale=False, jitter=0, diff_stride=True, num_workers=conf['num_workers'])
        elif name.startswith('ass-') or name.startswith('wss-') or name.startswith('asr-') or name.startswith('asm-') or name.startswith('mfs-'):
            value = int(name[4:])
            loader = GetASSValidDataLoader(run, valid_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, augment_op=augment_op, combine_op=None, patched_op=patched_op, auto_scale=True, rand_scale=False, jitter=0, num_workers=conf['num_workers'])
        elif name.startswith('las-') or name.startswith('wls-'):
            value = int(name[4:])
            loader = GetLASValidDataLoader(run, valid_df, conf['image_dir'], conf['num_images'], conf['gen_images'], value, augment_op=augment_op, combine_op=None, auto_scale=True, rand_scale=False, jitter=0, num_workers=conf['num_workers'])
        else:
            raise NameError('指定されたデータローダーは定義されていません (--{}_loader={})'.format(run, name))
    else:
        raise NameError('指定された実行処理へのデータローダーは定義されていません (--run={})'.format(run))
    return loader

# 学習データローダー
def GetFmt1TrainDataLoader(run, train_df, image_dir, num_images, batch, augment_op=None, combine_op=None, num_cols=4, num_workers=4, drop_last=True):
    loader = torch.utils.data.DataLoader(
        Fmt1TrainDataset(run, train_df, image_dir, num_images, augment_op=augment_op, combine_op=combine_op, num_cols=num_cols),
        batch_size=batch,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return loader

# 検証データローダー
def GetFmt1ValidDataLoader(run, valid_df, image_dir, num_images, augment_op=None, combine_op=None, num_cols=4, num_workers=4):
    loader = torch.utils.data.DataLoader(
        Fmt1TrainDataset(run, valid_df, image_dir, num_images, augment_op=augment_op, combine_op=combine_op, num_cols=num_cols),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    return loader

# Regression/Classification Dataset (fmt1)
class Fmt1TrainDataset(Dataset):
    def __init__(self, run, dataframe, image_dir, num_images, augment_op=None, combine_op=None, num_cols=4):
        self.run = run
        self.length = len(dataframe)
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.num_images = num_images
        self.augment_fn = self.__null_augment__ if augment_op is None else self.__list_augment__
        self.augment_op = augment_op
        self.combine_op = combine_op
        self.num_cols = num_cols

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 画像読込:
        imgpath = os.path.join(self.image_dir, 'train_images', self.dataframe.iat[idx, 0] + '_cmb.png')
        img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 結合オーグメント:
        if self.combine_op is not None:
            img = self.__run_augment__(self.combine_op, img)
        # 色反転処理:
        img = 255 - img
        col = self.num_cols
        dim = img.shape[1] // self.num_cols
        mat = np.zeros([self.num_images, 3, dim, dim], dtype=np.float32)
        for i in range(self.num_images):
            r, c = divmod(i, col)
            buf = img[r * dim:(r+1) * dim, c * dim:(c+1) * dim, :]
            ret = self.augment_fn(buf)
            ret = ret['image'].transpose(2, 0, 1)
            mat[i, :, :, :] = ret
        # 他情報取得:
        # provider = self.dataframe.iat[idx, 1]
        # subscore = self.dataframe.iat[idx, 3]
        # if subscore == 'negative':
        #     subscore = '0+0'
        # ラベル取得:
        if self.run == 'reg':
            lbl = np.zeros([1], dtype=np.float32)
            lbl[0] = self.dataframe.iat[idx, 2].astype(np.float32)
            return torch.Tensor(mat), torch.Tensor(lbl)
        elif self.run == 'cls':
            return torch.Tensor(mat), self.dataframe.iat[idx, 2]
        else:
            pass
        lbl = np.zeros([1], dtype=np.float32)
        lbl[0] = self.dataframe.iat[idx, 2].astype(np.float32)
        return torch.Tensor(mat), torch.Tensor(lbl), self.dataframe.iat[idx, 2]

    def __null_augment__(self, img):
        return { 'image' : img }

    def __list_augment__(self, img):
        idx = random.randint(0, len(self.augment_op) - 1)
        return self.augment_op[idx](force_apply=False, image=img)

    def __run_augment__(self, ops, img):
        if type(ops) == list:
            idx = random.randint(0, len(ops) - 1)
            return ops[idx](force_apply=False, image=img)['image']
        return ops(force_apply=False, image=img)['image']

# 学習データローダー
def GetAutoTileTrainDataLoader(run, train_df, image_dir, num_images, gen_images, image_size, batch, augment_op=None, combine_op=None, auto_scale=False, rand_scale=False, jitter=0, weighted=False, num_workers=4, drop_last=True):
    loader = torch.utils.data.DataLoader(
        AutoTileTrainDataset(run, train_df, image_dir, num_images, gen_images, image_size, augment_op=augment_op, combine_op=combine_op, auto_scale=auto_scale, jitter=jitter, rand_scale=rand_scale, weighted=weighted),
        batch_size=batch,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return loader

# 検証データローダー
def GetAutoTileValidDataLoader(run, valid_df, image_dir, num_images, gen_images, image_size, augment_op=None, combine_op=None, auto_scale=False, rand_scale=False, jitter=0, num_workers=4):
    loader = torch.utils.data.DataLoader(
        AutoTileTrainDataset(run, valid_df, image_dir, num_images, gen_images, image_size, augment_op=augment_op, combine_op=combine_op, auto_scale=auto_scale, rand_scale=rand_scale, jitter=0, weighted=False),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    return loader

# Regression/Classification Dataset (fmt1-base)
class AutoTileTrainDataset(Dataset):
    def __init__(self, run, dataframe, image_dir, num_images, gen_images, image_size, augment_op=None, combine_op=None, auto_scale=False, rand_scale=False, jitter=0, weighted=False):
        self.run = run
        self.length = len(dataframe)
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.num_images = num_images
        self.gen_images = gen_images
        self.num_column = int(np.sqrt(self.gen_images))
        self.image_size = image_size
        self.augment_fn = self.__null_augment__ if augment_op is None else self.__list_augment__
        self.augment_op = augment_op
        self.combine_op = combine_op
        self.auto_scale = auto_scale
        self.rand_scale = rand_scale
        self.jitter = jitter
        self.weighted = weighted
        if self.num_images > self.gen_images:
            raise ValueError('指定した画像枚数は準備されません (num_images < gen_images).')
        if self.dataframe.shape[1] <= 5:
            self.weighted = False
            print('CSVにweightが記載されていません.')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 画像読込:
        imgpath = os.path.join(self.image_dir, 'train_images', self.dataframe.iat[idx, 0] + '.tiff')
        scale_center = 1.50
        if self.auto_scale and self.rand_scale:
            alg = [
                cv2.INTER_LINEAR,
                cv2.INTER_AREA,
            ]
            rsf = np.clip(np.random.normal(loc=scale_center, scale=0.25, size=1), 1.00, 2.00)
            img = LoadImage(imgpath, self.gen_images, rsf, layer=1, auto_ws=True, window=self.image_size // 4, stride=self.image_size // 4, jitter=self.jitter, algorithm=alg)
        elif self.auto_scale:
            img = LoadImage(imgpath, self.gen_images, scale_center, layer=1, auto_ws=True, window=self.image_size // 4, stride=self.image_size // 4, jitter=self.jitter)
        else:
            img = LoadImage(imgpath, self.gen_images, scale_center, layer=1, auto_ws=False, window=self.image_size // 4, stride=self.image_size // 4, jitter=self.jitter)
        # 結合オーグメント:
        if self.combine_op is not None:
            img = self.__run_augment__(self.combine_op, img)
        # 色反転処理:
        img = 255 - img
        dim = self.image_size
        mat = np.zeros([self.num_images, 3, dim, dim], dtype=np.float32)
        for i in range(self.num_images):
            r, c = divmod(i, self.num_column)
            buf = img[r * dim:(r+1) * dim, c * dim:(c+1) * dim, :]
            ret = self.augment_fn(buf)
            ret = ret['image'].transpose(2, 0, 1)
            mat[i, :, :, :] = ret
        # 他情報取得:
        weight = np.ones([1], dtype=np.float32)
        if self.weighted:
            weight[0] = self.dataframe.iat[i, 4]
        # subscore = self.dataframe.iat[idx, 3]
        # if subscore == 'negative':
        #     subscore = '0+0'
        # ラベル取得:
        lbl = np.zeros([1], dtype=np.float32)
        lbl[0] = self.dataframe.iat[idx, 2].astype(np.float32)
        if self.run == 'reg':
            return torch.Tensor(mat), torch.Tensor(lbl), torch.Tensor(weight)
        elif self.run == 'cls':
            return torch.Tensor(mat), self.dataframe.iat[idx, 2], torch.Tensor(weight)
        else:
            pass
        return torch.Tensor(mat), torch.Tensor(lbl), self.dataframe.iat[idx, 2], torch.Tensor(weight)

    def __null_augment__(self, img):
        return { 'image' : img }

    def __list_augment__(self, img):
        idx = random.randint(0, len(self.augment_op) - 1)
        return self.augment_op[idx](force_apply=False, image=img)

    def __run_augment__(self, ops, img):
        if type(ops) == list:
            idx = random.randint(0, len(ops) - 1)
            return ops[idx](force_apply=False, image=img)['image']
        return ops(force_apply=False, image=img)['image']

# 学習データローダー
def GetASSTrainDataLoader(run, train_df, image_dir, num_images, gen_images, image_size, batch, augment_op=None, combine_op=None, patched_op=None, auto_scale=False, rand_scale=False, kr_factor=False, kr_layer=False, jitter=0, weighted=False, randomize=False, blending=False, diff_stride=False, num_workers=4, drop_last=True):
    dataset = ASSTrainDataset(run, train_df, image_dir, num_images, gen_images, image_size, augment_op=augment_op, combine_op=combine_op, patched_op=patched_op, auto_scale=auto_scale, rand_scale=rand_scale, kr_factor=kr_factor, kr_layer=kr_layer, jitter=jitter, weighted=weighted, randomize=randomize, blending=blending, diff_stride=diff_stride)
    sampler = torch.utils.data.DistributedSampler(dataset) if IsDistributed() else None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch,
        shuffle=(sampler is None),
        drop_last=drop_last,
        sampler=sampler,
        num_workers=num_workers
    )
    return loader

# 検証データローダー
def GetASSValidDataLoader(run, valid_df, image_dir, num_images, gen_images, image_size, augment_op=None, combine_op=None, patched_op=None, auto_scale=False, rand_scale=False, kr_factor=False, kr_layer=False, jitter=0, weighted=False, blending=False, diff_stride=False, num_workers=4):
    loader = torch.utils.data.DataLoader(
        ASSTrainDataset(run, valid_df, image_dir, num_images, gen_images, image_size, augment_op=augment_op, combine_op=combine_op, patched_op=patched_op, auto_scale=auto_scale, rand_scale=rand_scale, kr_factor=kr_factor, kr_layer=kr_layer, jitter=0, weighted=weighted, randomize=False, blending=blending, diff_stride=diff_stride),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    return loader

# Regression/Classification Dataset (auto scale simple data)
class ASSTrainDataset(Dataset):
    def __init__(self, run, dataframe, image_dir, num_images, gen_images, image_size, augment_op=None, combine_op=None, patched_op=None, auto_scale=False, rand_scale=False, kr_factor=False, kr_layer=False, jitter=0, weighted=False, randomize=False, blending=False, diff_stride=False):
        self.run = run
        self.length = len(dataframe)
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.num_images = num_images
        self.gen_images = gen_images
        self.num_column = int(np.sqrt(self.gen_images))
        self.image_size = image_size
        self.augment_fn = self.__null_augment__ if augment_op is None else self.__list_augment__
        self.augment_op = augment_op
        self.combine_op = combine_op
        self.patched_op = patched_op
        self.auto_scale = auto_scale
        self.rand_scale = rand_scale
        self.kr_factor = kr_factor
        self.kr_layer = kr_layer
        self.diff_stride = diff_stride
        self.jitter = jitter
        self.weighted = weighted
        self.randomize = randomize
        self.blending = blending
        self.imgcache = {
            'karolinska' : [ ],
            'radboud' : [ ]
        }
        if self.num_images > self.gen_images:
            raise ValueError('指定した画像枚数は準備されません (num_images < gen_images).')
        if self.weighted and self.dataframe.shape[1] <= 5:
            self.weighted = False
            print('CSVにweightが記載されていません.')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 画像読込:
        imgpath = os.path.join(self.image_dir, 'train_images', self.dataframe.iat[idx, 0] + '.tiff')
        if self.kr_factor:
            if self.dataframe.iat[idx, 1] == 'karolinska':
                scale_center_c = 2.00
                scale_center_d = 0.75
            else:
                scale_center_c = 1.00
                scale_center_d = 0.25
        else:
            scale_center_c = 2.00
            scale_center_d = 0.75
        lay = 1
        if self.kr_layer and self.dataframe.iat[idx, 1] == 'radboud':
            lay = 0
        stride = self.image_size // 4
        if self.diff_stride:
            stride = (self.image_size // 4) if self.dataframe.iat[idx, 1] == 'karolinska' else (self.image_size // 8)
        if self.auto_scale and self.rand_scale:
            alg = [
                cv2.INTER_LINEAR,
                cv2.INTER_AREA,
            ]
            # rsf = np.clip(np.random.normal(loc=scale_center, scale=0.25, size=1), 1.50, 2.50)
            rsf = np.clip(np.random.normal(loc=scale_center_c, scale=scale_center_d, size=1), 0.50, 3.50)
            img = LoadImage(imgpath, self.gen_images, rsf, layer=lay, auto_ws=True, window=self.image_size // 4, stride=stride, jitter=self.jitter, algorithm=alg)
        elif self.auto_scale:
            img = LoadImage(imgpath, self.gen_images, scale_center_c, layer=lay, auto_ws=True, window=self.image_size // 4, stride=stride, jitter=self.jitter)
        else:
            img = LoadImage(imgpath, self.gen_images, scale_center_c, layer=lay, auto_ws=False, window=self.image_size // 4, stride=stride, jitter=self.jitter)
        # 画像混合:
        # いろいろ即値で埋め込むのであとで頑張る
        subscore = self.dataframe.iat[idx, 3]
        if subscore == 'negative':
            subscore = '0+0'
        major = int(subscore[0])
        minor = int(subscore[2])
        lblval = int(self.dataframe.iat[idx, 2])
        if self.blending and lblval > 0 and major == minor:
            # プロバイダごとに処理:
            provider = self.dataframe.iat[idx, 1]
            v = np.random.rand()
            n = 1 + int(5 * np.random.rand())
            k = list(range(self.num_images))
            s = random.sample(k, n)
            core = img.copy()
            if len(self.imgcache[provider]) > 0:
                item = random.choice(self.imgcache[provider])
                bscr, bimg = item
                blbl = int(bscr[0])
                # 画像ブレンド
                # 高位のものほど混ぜるようにしておく
                p = 0.125 * (major - 3)
                if v < 0.5 + p:
                    dim = self.image_size
                    for ii in s:
                        rr, cc = divmod(ii, self.num_column)
                        c0 = dim * (cc)
                        c1 = dim * (cc+1)
                        r0 = dim * (rr)
                        r1 = dim * (rr+1)
                        img[r0:r1, c0:c1, :] = bimg[r0:r1, c0:c1, :]
                # ラベル補正:
                gleason = '{}+{}'.format(major, max(minor, blbl))
                lblval = get_isup(gleason)
            # 画像キャッシュ
            if subscore in [ '3+3', '4+4', '5+5' ]:
                self.imgcache[provider].append((subscore, core))
                if len(self.imgcache[provider]) > 30:
                    self.imgcache[provider].pop(0)
        # 結合オーグメント:
        if self.combine_op is not None:
            img = self.__run_augment__(self.combine_op, img)
        # 色反転処理:
        img = 255 - img
        dim = self.image_size
        mat = np.zeros([dim * 4, dim * 4, 3], dtype=np.float32)
        buf = np.zeros([dim * 4, dim * 4, 3], dtype=np.int8)
        for i in range(self.num_images):
            r, c = divmod(i, self.num_column)
            buf = img[r * dim:(r+1) * dim, c * dim:(c+1) * dim, :]
            a, b = buf.shape[:2]
            if a < dim or b < dim:
                buf = np.zeros([dim, dim, 3], dtype=np.uint8)
            ret = self.augment_fn(buf)
            r0 = r * dim
            r1 = r0 + dim
            c0 = c * dim
            c1 = c0 + dim
            mat[r0:r1, c0:c1, :] = ret['image']
        if self.patched_op is not None:
            mat = self.__run_augment__(self.patched_op, mat)
        mat = mat.transpose(2, 0, 1)
        # 他情報取得:
        weight = np.ones([1], dtype=np.float32)
        if self.weighted:
            weight[0] = self.dataframe.iat[i, 4]
        # ラベル取得:
        lbl = np.zeros([1], dtype=np.float32)
        # lblval = int(self.dataframe.iat[idx, 2])
        lbl[0] = lblval
        if self.randomize:
            stdv = 0.3 if lblval <= 1 or 4 <= lblval else 0.1
            lbl[0] = np.random.normal(loc=lbl[0], scale=stdv, size=1)
        if self.run == 'reg':
            return torch.Tensor(mat), torch.Tensor(lbl), torch.Tensor(weight)
        elif self.run == 'cls':
            return torch.Tensor(mat), self.dataframe.iat[idx, 2], torch.Tensor(weight)
        else:
            pass
        return torch.Tensor(mat), torch.Tensor(lbl), self.dataframe.iat[idx, 2], torch.Tensor(weight)

    def __null_augment__(self, img):
        return { 'image' : img }

    def __list_augment__(self, img):
        idx = random.randint(0, len(self.augment_op) - 1)
        return self.augment_op[idx](force_apply=False, image=img)

    def __run_augment__(self, ops, img):
        if type(ops) == list:
            idx = random.randint(0, len(ops) - 1)
            return ops[idx](force_apply=False, image=img)['image']
        return ops(force_apply=False, image=img)['image']


# 学習データローダー
def GetMFSTrainDataLoader(run, train_df, image_dir, num_images, gen_images, image_size, batch, augment_op=None, combine_op=None, patched_op=None, auto_scale=False, rand_scale=False, jitter=0, num_workers=4, drop_last=True):
    classes = 6 # 0+0, 3+3, 3+4, 4+3, { 4+4, 3+5, 5+3 }, { 4+5, 5+4, 5+5 }
    loader = torch.utils.data.DataLoader(
        MFSTrainDataset(run, train_df, image_dir, num_images, gen_images, image_size, augment_op=augment_op, combine_op=combine_op, patched_op=patched_op, auto_scale=auto_scale, rand_scale=rand_scale, jitter=jitter),
        batch_size=batch // classes,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return loader

# 検証データローダー
def GetMFSValidDataLoader(run, valid_df, image_dir, num_images, gen_images, image_size, augment_op=None, combine_op=None, patched_op=None, auto_scale=False, rand_scale=False, jitter=0, num_workers=4):
    # batch=1のためASSで実施
    loader = torch.utils.data.DataLoader(
        ASSTrainDataset(run, valid_df, image_dir, num_images, gen_images, image_size, augment_op=augment_op, combine_op=combine_op, patched_op=patched_op, auto_scale=auto_scale, rand_scale=rand_scale, jitter=0, weighted=False, randomize=False, blending=False),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    return loader

# Regression/Classification Dataset (multiple )
class MFSTrainDataset(Dataset):
    def __init__(self, run, dataframe, image_dir, num_images, gen_images, image_size, augment_op=None, combine_op=None, patched_op=None, auto_scale=False, rand_scale=False, jitter=0):
        self.run = run
        self.length = len(dataframe)
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.num_images = num_images
        self.gen_images = gen_images
        self.num_column = int(np.sqrt(self.gen_images))
        self.image_size = image_size
        self.augment_fn = self.__null_augment__ if augment_op is None else self.__list_augment__
        self.augment_op = augment_op
        self.combine_op = combine_op
        self.patched_op = patched_op
        self.auto_scale = auto_scale
        self.rand_scale = rand_scale
        self.jitter = jitter
        if self.num_images > self.gen_images:
            raise ValueError('指定した画像枚数は準備されません (num_images < gen_images).')
        self.indexlen = 0
        self.indexmap = self.__build__()
        self.indexset = None
        self.shuffle()

    def __len__(self):
        return self.indexlen

    def __getitem__(self, idx):
        cb = self.indexset[idx]
        rx = None
        ry = None 
        for i in cb:
            x, y, _ = self.__getitem_subkit__(i)
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            rx = torch.cat([rx, x], dim=0) if rx is not None else x
            ry = torch.cat([ry, y], dim=0) if ry is not None else y
        return rx, ry

    def __getitem_subkit__(self, idx):
        # 画像読込:
        imgpath = os.path.join(self.image_dir, 'train_images', self.dataframe.iat[idx, 0] + '.tiff')
        scale_center = 2.00
        if self.auto_scale and self.rand_scale:
            alg = [
                cv2.INTER_LINEAR,
                cv2.INTER_AREA,
            ]
            # rsf = np.clip(np.random.normal(loc=scale_center, scale=0.25, size=1), 1.50, 2.50)
            rsf = np.clip(np.random.normal(loc=scale_center, scale=0.75, size=1), 0.50, 3.50)
            img = LoadImage(imgpath, self.gen_images, rsf, layer=1, auto_ws=True, window=self.image_size // 4, stride=self.image_size // 4, jitter=self.jitter, algorithm=alg)
        elif self.auto_scale:
            img = LoadImage(imgpath, self.gen_images, scale_center, layer=1, auto_ws=True, window=self.image_size // 4, stride=self.image_size // 4, jitter=self.jitter)
        else:
            img = LoadImage(imgpath, self.gen_images, scale_center, layer=1, auto_ws=False, window=self.image_size // 4, stride=self.image_size // 4, jitter=self.jitter)
        # 結合オーグメント:
        if self.combine_op is not None:
            img = self.__run_augment__(self.combine_op, img)
        # 色反転処理:
        img = 255 - img
        dim = self.image_size
        mat = np.zeros([dim * 4, dim * 4, 3], dtype=np.float32)
        buf = np.zeros([dim * 4, dim * 4, 3], dtype=np.int8)
        for i in range(self.num_images):
            r, c = divmod(i, self.num_column)
            buf = img[r * dim:(r+1) * dim, c * dim:(c+1) * dim, :]
            a, b = buf.shape[:2]
            if a < dim or b < dim:
                buf = np.zeros([dim, dim, 3], dtype=np.uint8)
            ret = self.augment_fn(buf)
            r0 = r * dim
            r1 = r0 + dim
            c0 = c * dim
            c1 = c0 + dim
            mat[r0:r1, c0:c1, :] = ret['image']
        if self.patched_op is not None:
            mat = self.__run_augment__(self.patched_op, mat)
        mat = mat.transpose(2, 0, 1)
        # 他情報取得:
        weight = np.ones([1], dtype=np.float32)
        # ラベル取得:
        lbl = np.zeros([1], dtype=np.float32)
        lblval = int(self.dataframe.iat[idx, 2])
        lbl[0] = lblval
        if self.run == 'reg':
            return torch.Tensor(mat), torch.Tensor(lbl), torch.Tensor(weight)
        elif self.run == 'cls':
            return torch.Tensor(mat), self.dataframe.iat[idx, 2], torch.Tensor(weight)
        else:
            pass
        return torch.Tensor(mat), torch.Tensor(lbl), self.dataframe.iat[idx, 2], torch.Tensor(weight)

    def __null_augment__(self, img):
        return { 'image' : img }

    def __list_augment__(self, img):
        idx = random.randint(0, len(self.augment_op) - 1)
        return self.augment_op[idx](force_apply=False, image=img)

    def __run_augment__(self, ops, img):
        if type(ops) == list:
            idx = random.randint(0, len(ops) - 1)
            return ops[idx](force_apply=False, image=img)['image']
        return ops(force_apply=False, image=img)['image']

    def __build__(self):
        obj = {
            0 : [],
            1 : [],
            2 : [],
            3 : [],
            4 : [],
            5 : [],
        }
        for i in range(len(self.dataframe)):
            p = int(self.dataframe.iat[i, 2])
            obj[p].append(i)
        self.indexlen = 0
        for _, v in obj.items():
            self.indexlen = max(self.indexlen, len(v))
        return obj

    def shuffle(self):
        self.indexset = [ ]
        ids = [ random.choices(self.indexmap[i], k=self.indexlen) for i in range(6) ]
        for i in range(self.indexlen):
            self.indexset.append((ids[0][i], ids[1][i], ids[2][i], ids[3][i], ids[4][i], ids[5][i]))

# 学習データローダー
def GetLASTrainDataLoader(run, train_df, image_dir, num_images, gen_images, image_size, batch, augment_op=None, combine_op=None, auto_scale=False, rand_scale=False, jitter=0, num_workers=4, drop_last=True):
    loader = torch.utils.data.DataLoader(
        LASTrainDataset(run, train_df, image_dir, num_images, gen_images, image_size, augment_op=augment_op, combine_op=combine_op, auto_scale=auto_scale, rand_scale=rand_scale, jitter=jitter),
        batch_size=batch,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return loader

# 検証データローダー
def GetLASValidDataLoader(run, valid_df, image_dir, num_images, gen_images, image_size, augment_op=None, combine_op=None, auto_scale=False, rand_scale=False, jitter=0, num_workers=4):
    loader = torch.utils.data.DataLoader(
        LASTrainDataset(run, valid_df, image_dir, num_images, gen_images, image_size, augment_op=augment_op, combine_op=combine_op, auto_scale=auto_scale, rand_scale=rand_scale, jitter=0),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    return loader

# Regression/Classification Dataset (label with auto scale data)
class LASTrainDataset(Dataset):
    def __init__(self, run, dataframe, image_dir, num_images, gen_images, image_size, augment_op=None, combine_op=None, auto_scale=False, rand_scale=False, jitter=0):
        self.run = run
        self.length = len(dataframe)
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.num_images = num_images
        self.gen_images = gen_images
        self.num_column = int(np.sqrt(self.gen_images))
        self.image_size = image_size
        self.augment_fn = self.__null_augment__ if augment_op is None else self.__list_augment__
        self.augment_op = augment_op
        self.combine_op = combine_op
        self.auto_scale = auto_scale
        self.rand_scale = rand_scale
        self.jitter = jitter
        if self.num_images > self.gen_images:
            raise ValueError('指定した画像枚数は準備されません (num_images < gen_images).')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 画像読込:
        imgpath = os.path.join(self.image_dir, 'train_images', self.dataframe.iat[idx, 0] + '.tiff')
        scale_center = 2.00
        if self.auto_scale and self.rand_scale:
            alg = [
                cv2.INTER_LINEAR,
                cv2.INTER_AREA,
            ]
            # rsf = np.clip(np.random.normal(loc=scale_center, scale=0.25, size=1), 1.50, 2.50)
            rsf = np.clip(np.random.normal(loc=scale_center, scale=0.75, size=1), 0.50, 3.50)
            img = LoadImage(imgpath, self.gen_images, rsf, layer=1, auto_ws=True, window=self.image_size // 4, stride=self.image_size // 4, jitter=self.jitter, algorithm=alg)
        elif self.auto_scale:
            img = LoadImage(imgpath, self.gen_images, scale_center, layer=1, auto_ws=True, window=self.image_size // 4, stride=self.image_size // 4, jitter=self.jitter)
        else:
            img = LoadImage(imgpath, self.gen_images, scale_center, layer=1, auto_ws=False, window=self.image_size // 4, stride=self.image_size // 4, jitter=self.jitter)
        # 結合オーグメント:
        if self.combine_op is not None:
            img = self.__run_augment__(self.combine_op, img)
        # 色反転処理:
        img = 255 - img
        dim = self.image_size
        mat = np.zeros([dim * 4, dim * 4, 3], dtype=np.float32)
        for i in range(self.num_images):
            r, c = divmod(i, self.num_column)
            buf = img[r * dim:(r+1) * dim, c * dim:(c+1) * dim, :]
            a, b = buf.shape[:2]
            if a < dim or b < dim:
                buf = np.zeros([dim, dim, 3], dtype=np.uint8)
            ret = self.augment_fn(buf)
            r0 = r * dim
            r1 = r0 + dim
            c0 = c * dim
            c1 = c0 + dim
            mat[r0:r1, c0:c1, :] = ret['image']
        mat = mat.transpose(2, 0, 1)
        # 他情報取得:
        weight = np.zeros([1], dtype=np.float32)
        weight[0] = 1.0 if self.dataframe.iat[idx, 1] == 'karolinska' else 0.0
        # provider = self.dataframe.iat[idx, 1]
        # subscore = self.dataframe.iat[idx, 3]
        # if subscore == 'negative':
        #     subscore = '0+0'
        # ラベル取得:
        lbl = np.zeros([1], dtype=np.float32)
        lbl[0] = self.dataframe.iat[idx, 2].astype(np.float32)
        if self.run == 'reg':
            return torch.Tensor(mat), torch.Tensor(lbl), torch.Tensor(weight)
        elif self.run == 'cls':
            return torch.Tensor(mat), self.dataframe.iat[idx, 2], torch.Tensor(weight)
        else:
            pass
        return torch.Tensor(mat), torch.Tensor(lbl), self.dataframe.iat[idx, 2], torch.Tensor(weight)

    def __null_augment__(self, img):
        return { 'image' : img }

    def __list_augment__(self, img):
        idx = random.randint(0, len(self.augment_op) - 1)
        return self.augment_op[idx](force_apply=False, image=img)

    def __run_augment__(self, ops, img):
        if type(ops) == list:
            idx = random.randint(0, len(ops) - 1)
            return ops[idx](force_apply=False, image=img)['image']
        return ops(force_apply=False, image=img)['image']

# Tensor処理:
def CreateBatchTensor(tensor, size):
    b, _, h, w = tensor.shape[:]
    nh = h // size
    nw = w // size
    mat = None
    for k in range(b):
        for j in range(nh):
            for i in range(nw):
                tns = tensor[k, :, j*size:(j+1)*size, i*size:(i+1)*size].unsqueeze(0)
                mat = tns if mat is None else torch.cat((mat, tns), dim=0)
    return mat

def get_isup(gleason):
    if gleason in [ '0+0', 'negative' ]:
        return 0
    elif gleason == '3+3':
        return 1
    elif gleason == '3+4':
        return 2
    elif gleason == '4+3':
        return 3
    elif gleason in [ '4+4', '3+5', '5+3' ]:
        return 4
    elif gleason in [ '4+5', '5+4', '5+5' ]:
        return 5
    else:
        pass
    return 0

""" OTA Data Create Module """
def compute_statistics(image):
    h, w = image.shape[:2]
    value_pixel = np.sum(image, axis=-1)
    white_ratio = np.count_nonzero(value_pixel > 700) / (w * h)
    r_mean = np.mean(image[:, :, 0])
    g_mean = np.mean(image[:, :, 1])
    b_mean = np.mean(image[:, :, 2])
    return white_ratio, r_mean, g_mean, b_mean

def select_k_best_regions(regions, K):
    k_best_regions = sorted(regions, key=lambda x: x[5])[:K]
#   k_best_regions = sorted(regions, key=lambda x: x[2])[:K]
    return k_best_regions

def get_k_best_regions(regions, image, window=512):
    images = {}
    for i, itr in enumerate(regions):
        x, y = itr[0], itr[1]
        images[i] = image[y : y + window, x : x + window, :]
    return images

def detect_best_window_size(image, K, scaling_factor):
    white_ratio, _, _, _ = compute_statistics(image)
    # white_ratio, _, _ = compute_statistics(image)
    h, w = image.shape[:2]
    return max(int(np.sqrt(h * w * (1.0 - white_ratio) * scaling_factor / K)), 32)

def glue_to_one_picture(patches, window, K):
    block = int(np.sqrt(K))
    image = np.zeros((block * window, block * window, 3), dtype=np.uint8)
    for i, patch in patches.items():
        r, c = divmod(i, block)
        image[r * window : (r+1) * window, c * window : (c+1) * window, :] = patch
    return image

def generate_patches(filepath, window, stride, K, auto_ws, scaling_factor, offset=(0, 0)):
    slide = OpenSlide(filepath)
    image = np.asarray(slide.read_region((0, 0), 2, slide.level_dimensions[2]))[:, :, :3]
    image = np.array(image)
    if auto_ws:
        window = detect_best_window_size(image, 16, scaling_factor)
        stride = window
    h, w = image.shape[:2]
    regions = []
    j = 0
    while offset[1] + window + stride * j <= h:
        i = 0
        while offset[0] + window + stride * i <= w:
            x_start = offset[0] + i * stride
            y_start = offset[1] + j * stride
            patch = image[y_start : y_start + window, x_start : x_start + window, :]
            white_ratio, r_mean, g_mean, b_mean = compute_statistics(patch)
            region = (x_start, y_start, white_ratio, g_mean, b_mean, white_ratio - (2 * r_mean - g_mean - b_mean) / 255.0)
            # white_ratio, g_mean, b_mean = compute_statistics(patch)
            # region = (x_start, y_start, white_ratio, g_mean, b_mean)
            regions.append(region)
            i += 1
        j += 1
    regions = select_k_best_regions(regions, K)
    patches = get_k_best_regions(regions, image, window)
    return image, regions, patches, window

def glue_to_one_picture_from_coord(url, regions, window=200, K=16, layer=0):
    block = int(np.sqrt(K))
    slide = OpenSlide(url)
    scale = slide.level_downsamples[2] / slide.level_downsamples[layer]
    image = np.full((int(block * window * scale), int(block * window * scale), 3), 255, dtype=np.uint8)
    for i, itr in enumerate(regions):
        r, c = divmod(i, block)
        ws = int(window * scale)
        xs = int(itr[0] * slide.level_downsamples[2])
        ys = int(itr[1] * slide.level_downsamples[2])
        # patch = np.asarray(slide.read_region((xs, ys), layer, (ws, ws)))[:, :, :3]
        # 環境によっては以下の対応が必要
        wo = int(window * slide.level_downsamples[2])
        patch = np.asarray(slide.read_region((xs, ys), 0, (wo, wo)))[:, :, :3]
        # layer != 0 の場合はリサイズ
        if layer > 0:
            patch = cv2.resize(patch, (ws, ws))
        r0 = int(r * window * scale)
        r1 = r0 + ws
        c0 = int(c * window * scale)
        c1 = c0 + ws
        image[r0:r1, c0:c1, :] = patch
    slide.close()
    return image

def glue_to_one_picture_from_coord_lowlayer(url, regions, window=200, K=16, layer=1):
    block = int(np.sqrt(K))
    slide = OpenSlide(url)
    scale = slide.level_downsamples[2] / slide.level_downsamples[layer]
    slide.close()
    slide = skimage.io.MultiImage(url)[layer]
    slide = np.array(slide)
    image = np.full((int(block * window * scale), int(block * window * scale), 3), 255, dtype=np.uint8)
    # print(coordinates)
    for i, itr in enumerate(regions):
        r, c = divmod(i, block)
        ws = int(window * scale)
        x0 = int(itr[0] * scale)
        x1 = x0 + ws
        y0 = int(itr[1] * scale)
        y1 = y0 + ws
        buf = slide[y0:y1, x0:x1, :]
        h, w = buf.shape[:2]
        if h < ws or w < ws:
            dx = max(0, ws - w)
            dy = max(0, ws - h)
            buf = np.pad(buf, [[0, dy], [0, dx], [0, 0]], 'constant', constant_values=(255, 255))
        image[r * ws:(r+1) * ws, c * ws:(c+1) * ws, :] = buf
    return image

def LoadImage(filepath, K, scaling_factor, layer=0, auto_ws=True, window=128, stride=128, jitter=0, algorithm=None):
    outgrid = int(np.sqrt(K))
    outsize = window * 4
    offset = (0, 0)
    if jitter > 0:
        offset = (np.random.randint(0, jitter), np.random.randint(0, jitter))
    _, regions, _, window = generate_patches(filepath, window=window, stride=stride, K=K, auto_ws=auto_ws, scaling_factor=scaling_factor, offset=offset)
    # image = glue_to_one_picture_from_coord(filepath, regions, window=window, K=K, layer=layer)
    if layer == 0:
        image = glue_to_one_picture_from_coord(
            filepath, regions, window=window, K=K, layer=layer
        )
    else:
        image = glue_to_one_picture_from_coord_lowlayer(
            filepath, regions, window=window, K=K, layer=layer
        )
    # サイズを検証
    h, w = image.shape[:2]
    if h != outsize * outgrid or w != outsize * outgrid:
        # 縮小手法を設定可能にする
        if algorithm is not None:
            alg = cv2.INTER_LINEAR
        elif type(algorithm) == int:
            alg = algorithm
        elif type(algorithm) == list:
            alg = random.choice(algorithm)
        else:
            alg = cv2.INTER_LINEAR
        image = cv2.resize(image, (outsize * outgrid, outsize * outgrid), alg)
    return image

if __name__ == '__main__':
    # """
    import glob
    import pandas as pd
    dstpath = 'data/workout'
    srcpath = 'data/prostate-cancer-grade-assessment/train_images'
    srclist = sorted(glob.glob(os.path.join(srcpath, '**/*.tiff'), recursive=True))
    csvdata = pd.read_csv('csvs/cls_kfold_0/ota_o2u_22017707_k01.csv')
    csvlist = [ csvdata.iat[i, 1] for i in range(len(csvdata)) ]
    for src in srclist:
        name = os.path.splitext(os.path.basename(src))[0]
        if name not in csvlist:
            print('skip : {}'.format(name))
            continue
        scalefactor = 2.0
        img = LoadImage(src, 16, scalefactor, layer=0, auto_ws=True, window=128, stride=128, jitter=16, algorithm=cv2.INTER_AREA)
        dst = os.path.join(dstpath, name + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(dst, img)
        break
    """
    train_df = pd.read_csv('data/prostate-cancer-grade-assessment/train.csv')
    train_aug = GetTrainAugment('reg', 'icase1')
    tpost_aug = None # GetTrainAugment('reg', 'acase0')
    train_loader = GetTrainDataLoader(
        {
            'reg_loader' : 'mfs-512',
            'image_dir' : 'data/prostate-cancer-grade-assessment',
            'num_images' : 16,
            'gen_images' : 16,
            'batch' : 6,
            'jitter' : 0,
            'num_workers' : 0
         }, 'reg', train_df, augment_op=train_aug, patched_op=tpost_aug)
    for i, itr in enumerate(train_loader):
        x0, y0 = itr
        b, n, c, h, w = x0.shape
        x0 = x0.view(b * n, c, h, w)
        x0 = x0.detach().cpu().numpy()
        x0 = x0.transpose(0, 2, 3, 1)
        for j in range(x0.shape[0]):
            t = cv2.cvtColor(x0[j, :, :, :], cv2.COLOR_BGR2RGB)
            t = 255 - t
            cv2.imwrite('test/batch{}_index{}.png'.format(i, j), t)
    """
