#!/usr/bin/bash
# -*- coding: utf-8 -*-
import cv2
import random
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform

def icase0_augs(name, **kwargs):
    return [
        A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
        ], p=1.0),
    ]

def icase1_augs(name, **kwargs):
    return [
        A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, value=0),
        ], p=1.0),
    ]

def acase0_augs(name, **kwargs):
    return [
        A.Compose([
            A.Normalize()
        ], p=1.0),
    ]

def acase1_augs(name, **kwargs):
    return [
        A.Compose([
            A.GridDistortion(num_steps=4),
            A.Normalize()
        ], p=1.0),
    ]

def acase2_augs(name, **kwargs):
    return [
        A.Compose([
            A.Posterize(),
            A.GridDistortion(num_steps=4),
            A.Normalize()
        ], p=1.0),
        A.Compose([
            A.Downscale(),
            A.GridDistortion(num_steps=4),
            A.Normalize()
        ], p=1.0),
    ]

def case0_cls_train_augs(name, **kwargs):
    return [
        A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.Normalize()
        ], p=1.0),
    ]

def case1_cls_train_augs(name, **kwargs):
    return [
        A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize()
        ], p=1.0),
    ]


def case2_cls_train_augs(name, **kwargs):
    return [
        A.Compose([
            A.GridDistortion(distort_limit=0.1),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize()
        ], p=1.0),
    ]

def case3_cls_train_augs(name, **kwargs):
    return [
        A.Compose([
            A.ISONoise(),
            A.GridDistortion(distort_limit=0.1),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize()
        ], p=1.0),
        A.Compose([
            A.ISONoise(),
            A.OpticalDistortion(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize()
        ], p=1.0),
    ]

def case4_cls_train_augs(name, **kwargs):
    return [
        A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.Cutout(num_holes=3, max_h_size=24, max_w_size=24, fill_value=0),
            A.ShiftScaleRotate(rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize()
        ], p=1.0),
    ]

def case5_cls_train_augs(name, **kwargs):
    return [
        A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.Cutout(num_holes=3, max_h_size=48, max_w_size=48, fill_value=0),
            A.Normalize()
        ], p=1.0),
    ]

def subcase1_cls_train_augs(name, **kwargs):
    return [
        A.Compose([
            Colorout(fill=255, same=True),
        ], p=1.0),
        A.Compose([
            Colorout(fill=255, same=False),
        ], p=1.0),
        A.Compose([
            A.RandomGamma((100, 150))
        ], p=1.0),
        A.Compose([
            A.RandomBrightness(0.1)
        ], p=1.0),
    ]

def subcase2_cls_train_augs(name, **kwargs):
    return [
        A.Compose([
            A.RandomGridShuffle(grid=(4, 4)),
            A.HueSaturationValue(),
            Colorout(fill=255, same=True),
        ], p=1.0),
        A.Compose([
            A.RandomGridShuffle(grid=(4, 4)),
            A.HueSaturationValue(),
            Colorout(fill=255, same=False),
        ], p=1.0),
        A.Compose([
            A.RandomGridShuffle(grid=(4, 4)),
            A.HueSaturationValue(),
            A.RandomGamma((100, 150))
        ], p=1.0),
        A.Compose([
            A.RandomGridShuffle(grid=(4, 4)),
            A.HueSaturationValue(),
            A.RandomBrightness(0.1)
        ], p=1.0),
    ]

def subcase3_cls_train_augs(name, **kwargs):
    return [
        A.Compose([
            Colorout(fill=255, same=True),
            A.RandomGridShuffle(grid=(4, 4)),
        ], p=1.0),
        A.Compose([
            Colorout(fill=255, same=False),
            A.RandomGridShuffle(grid=(4, 4)),
        ], p=1.0),
        A.Compose([
            A.RandomBrightness(0.1),
            A.RandomGridShuffle(grid=(4, 4)),
        ], p=1.0),
        A.Compose([
            A.RandomGamma((100, 150)),
            A.RandomGridShuffle(grid=(4, 4)),
        ], p=1.0),
        A.Compose([
            A.HueSaturationValue(),
            A.RandomGridShuffle(grid=(4, 4)),
        ], p=1.0),
    ]

def subcase4_cls_train_augs(name, **kwargs):
    return [
        A.Compose([
            Colorout(fill=255, same=True),
            A.RandomGridShuffle(grid=(4, 4)),
        ], p=1.0),
        A.Compose([
            Colorout(fill=255, same=False),
            A.RandomGridShuffle(grid=(4, 4)),
        ], p=1.0),
        A.Compose([
            A.RandomBrightness(0.1),
            A.RandomGridShuffle(grid=(4, 4)),
        ], p=1.0),
        A.Compose([
            A.RandomGamma((100, 150)),
            A.RandomGridShuffle(grid=(4, 4)),
        ], p=1.0),
        A.Compose([
            A.HueSaturationValue(),
            A.RandomGridShuffle(grid=(4, 4)),
        ], p=1.0),
        A.Compose([
            A.RandomGridShuffle(grid=(4, 4)),
        ], p=1.0),
    ]

def subcase5_cls_train_augs(name, **kwargs):
    return [
        A.Compose([
            A.RandomBrightness(),
            A.RandomGridShuffle(grid=(4, 4)),
        ], p=1.0),
        A.Compose([
            A.RandomGamma(),
            A.RandomGridShuffle(grid=(4, 4)),
        ], p=1.0),
        A.Compose([
            A.HueSaturationValue(),
            A.RandomGridShuffle(grid=(4, 4)),
        ], p=1.0),
        A.Compose([
            A.RandomGridShuffle(grid=(4, 4)),
        ], p=1.0),
    ]

def case1_cls_valid_augs(name, **kwargs):
    return [
        A.Compose([
            A.Normalize()
        ], p=1.0)
    ]

class RandomCrop(DualTransform):
    def __init__(self, ratio=(0.75, 1.00), always_apply=False, p=1.0):
        super(RandomCrop, self).__init__(always_apply, p)
        self.ratio_min = ratio[0]
        self.ratio_var = ratio[1] - ratio[0]

    def apply(self, img, var=0.0, h_var=0, w_var=0, **params):
        h, w = img.shape[:2]
        ratio = np.clip(var * self.ratio_var + self.ratio_min, 0, 1)
        r_h = int(ratio * img.shape[0])
        r_w = int(ratio * img.shape[1])
        s_h = int((img.shape[0] - r_h) * h_var)
        s_w = int((img.shape[1] - r_w) * w_var)
        buf = img[s_h:s_h+r_h, s_w:s_w+r_w, :] if len(img.shape) == 3 else img[s_h:s_h+r_h, s_w:s_w+r_w]
        img = cv2.resize(buf, (h, w))
        return img

    def get_transform_init_args_names(self):
        return ("ratio")

    def get_params(self):
        return { "var": random.random(), "h_var": random.random(), "w_var": random.random() }

class SwapChannel(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(SwapChannel, self).__init__(always_apply, p)

    def apply(self, img, **params):
        buf = img.copy()
        img[:, :, 0] = buf[:, :, 2]
        img[:, :, 2] = buf[:, :, 0]
        return img

class Colorout(ImageOnlyTransform):
    def __init__(self, alpha=(0.05, 0.10), fill=255, same=True, always_apply=False, p=0.5):
        super(Colorout, self).__init__(always_apply, p)
        if type(alpha) == list or type(alpha) == tuple:
            self.alpha_min = alpha[0]
            self.alpha_var = alpha[1] - alpha[0]
        else:
            self.alpha_min = alpha
            self.alpha_var = 0
        self.fill = fill
        self.same = same

    def apply(self, img, **params):
        if len(img.shape) == 3:
            if self.same:
                r_var = self.alpha_var * np.random.rand() + self.alpha_min
                g_var = r_var
                b_var = r_var
            else:
                r_var = self.alpha_var * np.random.rand() + self.alpha_min
                g_var = self.alpha_var * np.random.rand() + self.alpha_min
                b_var = self.alpha_var * np.random.rand() + self.alpha_min
            buf = img.copy().astype(np.float32)
            buf[:, :, 0] = (1.00 - r_var) * buf[:, :, 0] + r_var * self.fill
            buf[:, :, 1] = (1.00 - g_var) * buf[:, :, 1] + g_var * self.fill
            buf[:, :, 2] = (1.00 - b_var) * buf[:, :, 2] + b_var * self.fill
        else:
            n_var = self.alpha_var * np.random.rand() + self.alpha_min
            buf = img.copy().astype(np.float32)
            buf[:, :] = (1.00 - r_var) * buf[:, :] + n_var * self.fill
        img = np.clip(buf, 0, 255).astype(np.uint8)
        return img
