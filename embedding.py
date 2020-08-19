# %% [code]
import os
import cv2
import math
import time
import numpy as np
import pandas as pd
import skimage.io
import torch
import torch.nn as nn
import torchvision
import timm
import tqdm
from openslide import OpenSlide
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import round_filters

# %% [code]
# PTH_ROOT4 = '/kaggle/input/panda-model-4'
PTH_ROOT4 = '/kqi/input/training/22466597/'

def SetPath4(file):
    return os.path.join(PTH_ROOT4, file)

# %% [code]
REG1_PTH = [
    { 'mdl': 'gem+efficientnet-b3', 'pth': SetPath4('reg-22018008-36.pth') }, # LB88 : CV 881
    { 'mdl': 'gem+efficientnet-b3', 'pth': SetPath4('reg-22018124-36.pth') }, # LBxx : CV xxx
    { 'mdl': 'gem+efficientnet-b3', 'pth': SetPath4('reg-22018125-40.pth') }, # LBxx : CV xxx
    { 'mdl': 'regnety_008', 'pth': SetPath4('reg-22017996-30.pth') }, # LB88 : CV 879
    { 'mdl': 'regnety_008', 'pth': SetPath4('reg-22018128-62.pth') }, # LBxx : CV xxx
    { 'mdl': 'regnety_016', 'pth': SetPath4('reg-22018126-29.pth') }, # LBxx : CV xxx
]

REG2_PTH = [
    { 'mdl': 'gem+efficientnet-b3', 'pth': SetPath4('reg-22018008-97.pth') }, # LBxx : CV 912 (karolinska)
    { 'mdl': 'gem+efficientnet-b3', 'pth': SetPath4('reg-22018124-57.pth') }, # LBxx : CV xxx
    { 'mdl': 'gem+efficientnet-b3', 'pth': SetPath4('reg-22018125-54.pth') }, # LBxx : CV xxx
    { 'mdl': 'regnety_008', 'pth': SetPath4('reg-22017996-90.pth') }, # LBxx : CV 907 (karolinska)
    { 'mdl': 'regnety_008', 'pth': SetPath4('reg-22018128-62.pth') }, # LBxx : CV xxx
    { 'mdl': 'regnety_016', 'pth': SetPath4('reg-22018126-29.pth') }, # LBxx : CV xxx
]

REG1_IMG = 512 # 画像サイズ
REG1_GEN = 16  # 用意数
REG1_NUM = 16  # 使用数
REG1_TTA = 5   # TTA
REG1_ESM_COUNT = len(REG1_PTH)

REG2_IMG = 512 # 画像サイズ
REG2_GEN = 16  # 用意数
REG2_NUM = 16  # 使用数
REG2_TTA = 5   # TTA
REG2_ESM_COUNT = len(REG2_PTH)

# common:
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ROOT_DIR = './data/prostate-cancer-grade-assessment'
ROOT_DIR = '/kqi/parent/'

# %% [code]
target_dir = os.path.join(ROOT_DIR, 'train_images')
target_df = pd.read_csv(os.path.join(ROOT_DIR, 'train.csv'))
# DEBUG
# target_df = target_df.iloc[0:50, :]
# target_df = target_df.iloc[0:5, :]

# %% [code]
"""
Data Creattor
"""
def compute_statistics(image):
    h, w = image.shape[:2]
    value_pixel = np.sum(image, axis=-1)
    white_ratio = np.count_nonzero(value_pixel > 700) / (w * h)
    g_mean = np.mean(image[:, :, 1])
    b_mean = np.mean(image[:, :, 2])
    return white_ratio, g_mean, b_mean

def select_k_best_regions(regions, K):
    k_best_regions = sorted(regions, key=lambda x: x[2])[:K]
    return k_best_regions

def get_k_best_regions(regions, image, window=512):
    images = {}
    for i, itr in enumerate(regions):
        x, y = itr[0], itr[1]
        images[i] = image[y : y + window, x : x + window, :]
    return images

def detect_best_window_size(image, K, scaling_factor):
    white_ratio, _, _ = compute_statistics(image)
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
            white_ratio, g_mean, b_mean = compute_statistics(patch)
            region = (x_start, y_start, white_ratio, g_mean, b_mean)
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
    ws = int(window * scale)
    wo = int(window * slide.level_downsamples[2])
    for i, itr in enumerate(regions):
        r, c = divmod(i, block)
        xs = int(itr[0] * slide.level_downsamples[2])
        ys = int(itr[1] * slide.level_downsamples[2])
        if layer > 0:
            patch = np.asarray(slide.read_region((xs, ys), 0, (wo, wo)))[:, :, :3]
            # patch = cv2.resize(patch, (ws, ws), cv2.INTER_AREA)
            patch = cv2.resize(patch, (ws, ws))
        else:
            patch = np.asarray(slide.read_region((xs, ys), 0, (ws, ws)))[:, :, :3]
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

def LoadImage(filepath, K, scaling_factor, layer=0, auto_ws=True, window=128, stride=128, jitter=0):
    offset = (0, 0)
    outgrid = int(np.sqrt(K))
    outsize = window * 4
    if jitter > 0:
        offset = (np.random.randint(0, jitter), np.random.randint(0, jitter))
    _, regions, _, window = generate_patches(filepath, window=window, stride=stride, K=K, auto_ws=auto_ws, scaling_factor=scaling_factor, offset=offset)
    if layer == 0:
        image = glue_to_one_picture_from_coord(
            filepath, regions, window=window, K=K, layer=layer
        )
    else:
        image = glue_to_one_picture_from_coord_lowlayer(
            filepath, regions, window=window, K=K, layer=layer
        )
        image = cv2.resize(image, (outsize * outgrid, outsize * outgrid))
    return image

# %% [code]
"""
DataLoader
"""
augops = A.Compose([
    A.RandomGridShuffle(grid=(4, 4)),
    A.Normalize()
], p=1.0)

def augment_fn(img):
    ret = augops(force_apply=False, image=img)
    return ret['image']

def GetImage(tdir, idn, image_size, scale_factor, jitter=0):
    # 想定: scale_factor = 2.0
    gen_images = 16
    num_images = 16
    num_column = int(np.sqrt(num_images))
    imgpath = os.path.join(tdir, idn + '.tiff')
    img = LoadImage(imgpath, gen_images, scale_factor, layer=1, auto_ws=True, window=image_size // 4, stride=image_size // 4, jitter=jitter)
    img = 255 - img
    dim = image_size
    mat = np.zeros([dim * 4, dim * 4, 3], dtype=np.float32)
    for i in range(num_images):
        r, c = divmod(i, num_column)
        buf = img[r * dim:(r+1) * dim, c * dim:(c+1) * dim, :]
        ret = augment_fn(buf)
        r0 = r * dim
        c0 = c * dim
        r1 = r0 + dim
        c1 = c0 + dim
        mat[r0:r1, c0:c1, :] = ret
    mat = mat.transpose(2, 0, 1)[np.newaxis, :, :, :]
    return torch.Tensor(mat)

# %% [code]
"""
ModelModule
"""
def gem(x, p=3, eps=1e-6):
    return nn.functional.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

@torch.jit.script
def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return mish(x)

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

class DualHead(nn.Module):
    def __init__(self, in_channels, num_classes1, num_classes2):
        super(DualHead, self).__init__()
        self.head1 = nn.Linear(in_channels, num_classes1)
        self.head2 = nn.Linear(in_channels, num_classes2)

    def forward(self, x):
        y1 = self.head1(x)
        y2 = self.head2(x)
        return y1, y2
    

class L2NormalizedHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(L2NormalizedHead, self).__init__()
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        n = x.norm(p=2, dim=1, keepdim=True)
        x = x.div(n)
        x = self.head(x)
        return x

# %% [code]
def GetModel(pths, num_classes, num_images):
    models = [ ]
    for pth in pths:
        name = pth['mdl']
        if name.startswith('gem+resnet34'):
            model = torchvision.models.resnet34(pretrained=None)
            model.avgpool = GeM()
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(torch.load(pth['pth'], map_location=DEVICE))
            model.fc = Identity()
        elif name.startswith('regnet'):
            model = timm.create_model(model_name=name, num_classes=num_classes, pretrained=False)
            model.load_state_dict(torch.load(pth['pth'], map_location=DEVICE))
            model.head.fc = Identity()
        elif name.startswith('norm2+regnet'):
            mname = name[6:]
            model = timm.create_model(model_name=mname, num_classes=num_classes, pretrained=False)
            model.head.fc = L2NormalizedHead(model.head.fc.in_features, num_classes)
            model.load_state_dict(torch.load(pth['pth'], map_location=DEVICE))
            model.head.fc = Identity()
        elif name.startswith('gem+efficientnet-b'):
            mname = name[4:]
            model = EfficientNet.from_name(mname)
            model._avg_pooling = GeM()
            model._fc = nn.Linear(model._fc.in_features, num_classes)
            model.load_state_dict(torch.load(pth['pth'], map_location=DEVICE))
            model._fc = Identity()
        elif name.startswith('efficientnet-b'):
            model = EfficientNet.from_name(name)
            model._fc = nn.Linear(model._fc.in_features, num_classes)
            model.load_state_dict(torch.load(pth['pth'], map_location=DEVICE))
            model._fc = Identity()
        else:
            raise NameError()
        model = model.to(DEVICE)
        model.eval()
        models.append(model)
    return models

# %% [code]
## Inference:
partnum = 1000
partial = int(math.ceil(len(target_df) / partnum))
for n in range(partial):
    result = [ ]
    vecnum = 0
    with torch.no_grad():
        models1 = GetModel(REG1_PTH, 1, REG1_NUM)
        models2 = GetModel(REG2_PTH, 1, REG2_NUM)
        restnum = min(len(target_df), (n+1) * partnum)
        print('section #{}'.format(n+1))
        set_time = time.time()
        # tqdmiter = tqdm.tqdm(total=restnum - n * partnum)
        for i in range(n * partnum, restnum):
            idn = target_df.iat[i, 0]
            prv = target_df.iat[i, 1]
            y1 = None
            sf = [ 1.8, 1.9, 2.0, 2.1, 2.2 ]
            # sf = [ 1.3, 1.4, 1.5, 1.6, 1.7 ]
            if prv == 'karolinska':
                for k in range(REG2_TTA):
                    sv = sf[k]
                    x1 = GetImage(target_dir, idn, REG2_IMG, sv, 0).to(DEVICE)
                    t1 = None
                    for mdl in models2:
                        u1 = mdl(x1)
                        t1 = torch.cat([t1, u1], dim=1) if t1 is not None else u1
                    y1 = (y1 + t1) if y1 is not None else t1
                y1 = y1 / REG2_TTA
                y1 = y1.detach().cpu().numpy()[0, :]
            else:
                for k in range(REG1_TTA):
                    sv = sf[k]
                    x1 = GetImage(target_dir, idn, REG1_IMG, sv, 0).to(DEVICE)
                    t1 = None
                    for mdl in models1:
                        u1 = mdl(x1)
                        t1 = torch.cat([t1, u1], dim=1) if t1 is not None else u1
                    y1 = (y1 + t1) if y1 is not None else t1
                y1 = y1 / REG1_TTA
                y1 = y1.detach().cpu().numpy()[0, :]
            """
            for k in range(REG1_TTA):
                sv = sf[k]
                x1 = GetImage(target_dir, idn, REG1_IMG, sv, 0).to(DEVICE)
                for mdl in models1:
                    u1 = mdl(x1)
                    y1 = y1 + u1
            y1 = y1 / (len(models1) * REG1_TTA)
            y1 = y1.detach().cpu().numpy()[0, 0]
            """
            ### reg-ensemble:
            z = y1

            ### capture:
            if vecnum > 0 and vecnum != z.shape[0]:
                raise ValueError()
            vecnum = z.shape[0]
            prvids = 1.0 if prv == 'karolinska' else 0.0
            res = '{},{},{}'.format(idn, prv, prvids)
            for k in range(vecnum):
                res = res + ',{}'.format(z[k])
            result.append(res)

            # tqdm:
            # tqdmiter.update()
        # tqdmiter.close()
        print('  elapsed time: {:.3f}'.format(time.time() - set_time))

    with open('/kqi/output/feature_vector_{:02}.csv'.format(n + 1), mode='w') as f:
    # with open('feature_vector_{:02}.csv'.format(n + 1), mode='w') as f:
        s = 'image_id,data_provider,feat_provider'
        for i in range(vecnum):
            s = s + ',feat_{}'.format(i)
        f.write(s + '\n')
        for item in result:
            f.write(item + '\n')