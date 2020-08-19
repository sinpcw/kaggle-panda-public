#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import cv2
import datetime
import numpy as np
import pandas as pd
from absl import app
from absl import flags
import torch
import models
import commons
import datasets
import augments
import albumentations as A
try:
    from apex import amp
    AVAILABLE_AMP = True
except ModuleNotFoundError:
    AVAILABLE_AMP = False

""" 引数 """
FLAGS = flags.FLAGS
# Data I/O:
flags.DEFINE_string('image_dir', '../input/', '入力ディレクトリを指定します.')
flags.DEFINE_string('input_csv', 'train.csv', '学習用入力データ(train.csv)を指定します.')
flags.DEFINE_string('output_dir', './output', '出力先ディレクトリを指定します.')
flags.DEFINE_integer('patch', 256, '画像のサイズ単位を指定します.')
# Preprocess:
flags.DEFINE_integer('tile', 128, 'タイル化する画像のサイズを指定します.')
flags.DEFINE_integer('upsample', None, 'タイル化する画像のアップサンプリングに使用するサイズを指定します.')
# Environment:
flags.DEFINE_enum('run', 'local', [ 'local', 'server' ], '実行マシンを指定します.')
flags.DEFINE_string('device', None, '実行デバイスを指定します.')
# Auto Mixed Precision:
flags.DEFINE_bool('amp', True, 'AMP(Auto-Mixed-Precision)の使用有無を指定します.')
flags.DEFINE_string('amp_level', 'O1', 'AMP(Auto-Mixed-Precision)のfp16化レベルを指定します.')
# Model:
flags.DEFINE_string('model', 'resnet34', 'モデル名を指定する.')
flags.DEFINE_string('model_uri', None, 'モデルパラメータファイルパスを指定します.')
# Memory:
flags.DEFINE_integer('patch_limit', 16, '推論に使用する最大パッチ数を指定します.')

""" ヘルパー """
# Device設定
def SetupDevice(conf):
    if 'device' in conf and conf['device'] is not None:
        if conf['device'] == 'cuda' and not torch.cuda.is_available():
            print('[警告] CUDAデバイスが使用できません. CPUモードで実行します.')
            conf['device'] = 'cpu'
    else:
        conf['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return conf

# ディレクトリ設定
def SetupOutput(conf):
    os.makedirs(conf['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(conf['output_dir'], 'train_images'), exist_ok=True)
    return conf

# Auto Mixed Precision設定
def SetupAMP(conf, model):
    if AVAILABLE_AMP and conf['amp']:
        model = amp.initialize(model, opt_level=conf['amp_level'])
    return model

# 現在時間を記録
def Now():
    t = datetime.datetime.now()
    return t.strftime('%Y/%m/%d/ %H:%M:%S')

""" 推論用関数 """
def Padding(img, patch, value):
    h, w = img.shape[:2]
    ph = (patch - (h % patch)) if h % patch > 0 else 0
    pw = (patch - (w % patch)) if w % patch > 0 else 0
    if len(img.shape) == 3:
        c = img.shape[2]
        ret = np.ones([h + ph, w + pw, c], dtype=np.uint8) * value
        ret[0:h, 0:w, :] = img[:, :, :]
    else:
        ret = np.ones([h + ph, w + pw], dtype=np.uint8) * value
        ret[0:h, 0:w] = img[:, :]
    return ret

def InferAug():
    return A.Compose([
        A.Normalize(),
    ], p=1.0)

def ApplyAugment(ops, img):
    ret = ops(force_apply=False, image=img)
    return ret['image']

def GetItem(df, image_dir, idx):
    idn = df.iat[idx, 0]
    osi = datasets.OpenSlideImage(os.path.join(image_dir, 'train_images', idn + '.tiff'))
    return osi, osi.load(2), idn

def Numpy2Tensor(x):
    if len(x.shape) == 2:
        x = x[np.newaxis, np.newaxis, :, :]
    elif len(x.shape) == 3:
        x = x.transpose((2, 0, 1))
        x = x[np.newaxis, :, :, :]
    else:
        pass
    return torch.Tensor(x)

def CreateTile(img, msk, size=128, nums=16):
    h, w = img.shape[:2]
    nh = h // size
    nw = w // size
    area = [ ]
    for r in range(nh):
        for c in range(nw):
            sums = np.sum(msk[r * size:(r+1) * size, c * size:(c+1) * size] / 255.0)
            area.append((r, c, sums))
    area = sorted(area, key=lambda x: -x[2])
    result = [ ]
    for i in range(nums):
        if i >= len(area) or area[i][2] <= 0:
            result.append({
                'image' : np.ones([size, size, 3], dtype=np.uint8) * 255,
                'umask' : np.zeros([size, size], dtype=np.uint8),
                'urect' : None
            })
        else:
            r = area[i][0]
            c = area[i][1]
            result.append({
                'image' : img[r * size:(r+1) * size, c * size:(c+1) * size, :],
                'umask' : msk[r * size:(r+1) * size, c * size:(c+1) * size],
                'urect' : (c * size, r * size, (c + 1) * size, (r + 1) * size)
            })
    return result

""" メイン関数 """
def main(argv):
    conf = commons.LoadArgs(FLAGS)
    conf = SetupOutput(conf)
    conf = SetupDevice(conf)
    if conf['model_uri'] is None:
        print('モデルパラメータが指定されていません. --model_uriで有効なパラメータを指定してください.')
    N_PATCH = conf['patch_limit']
    # 入力データ:
    patch = conf['patch']
    target_df = pd.read_csv(conf['input_csv'])
    infer_aug = InferAug()
    # モデル:
    model = models.GetModel(conf, num_classes=1, uri=conf['model_uri']).to(conf['device'])
    model = SetupAMP(conf, model)
    model = torch.nn.DataParallel(model)
    print('[{}] Start Preprocess (with Segmentation)'.format(Now()))
    model.eval()
    with torch.no_grad():
        count = len(target_df)
        wtqdm = commons.WrapTqdm(total=count, run=conf['run'])
        for i in range(count):
            osi, image, name = GetItem(target_df, conf['image_dir'], i)
            image = Padding(image, patch, 255)
            nh = image.shape[0] // patch
            nw = image.shape[1] // patch
            # segmentation:
            umask = np.zeros([nh * patch, nw * patch], dtype=np.uint8)
            x = ApplyAugment(infer_aug, image)
            x = Numpy2Tensor(x)
            x = datasets.CreateBatchTensor(x, patch)
            # メモリオーバー対策で分割して推論処理を実行
            nb, nc = divmod(x.shape[0], N_PATCH)
            for ni in range(nb):
                u = x[ni * N_PATCH:(ni + 1) * N_PATCH, :, :, :].to(conf['device'])
                y = model(u)
                y = torch.sigmoid(y).detach().cpu().numpy()
                y = np.where(y > 0.5, 255, 0).astype(np.uint8)
                k = ni * N_PATCH
                for j in range(N_PATCH):
                    r, c = divmod(k + j, nw)
                    umask[r * patch:(r+1) * patch, c * patch:(c + 1) * patch] = y[j, 0, :, :]
            if nc > 0:
                u = x[nb * N_PATCH:, :, :, :].to(conf['device'])
                y = model(u)
                y = torch.sigmoid(y).detach().cpu().numpy()
                y = np.where(y > 0.5, 255, 0).astype(np.uint8)
                k = nb * N_PATCH
                for j in range(nc):
                    r, c = divmod(k + j, nw)
                    umask[r * patch:(r+1) * patch, c * patch:(c + 1) * patch] = y[j, 0, :, :]
            # 前処理: タイルプロセス
            size = conf['tile']
            rows = 4
            cols = 4
            nums = rows * cols
            tiles = CreateTile(image, umask, size, nums)
            if conf['upsample'] is None:
                mat = np.zeros([rows * size, cols * size, 3], dtype=np.uint8)
                for j, tile in enumerate(tiles):
                    r, c = divmod(j, cols)
                    img = tile['image']
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    mat[r * size:(r+1) * size, c * size:(c+1) * size, :] = img[:, :, :]
            else:
                usp = conf['upsample']
                mat = np.ones([rows * usp, cols * usp, 3], dtype=np.uint8) * 255
                for j, tile in enumerate(tiles):
                    r, c = divmod(j, cols)
                    rect = tile['urect']
                    if rect is not None:
                        var = osi.upscale(2)
                        rect = (
                            int(rect[0] * var),
                            int(rect[1] * var),
                            int(rect[2] * var),
                            int(rect[3] * var),
                        )
                        img = osi.load_rect(1, rect, fill=255)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        img = cv2.resize(img, (usp, usp))
                        mat[r * usp:(r+1) * usp, c * usp:(c+1) * usp, :] = img[:, :, :]
            cv2.imwrite(os.path.join(conf['output_dir'], 'train_images', '{}_cmb.png'.format(name)), mat)
            wtqdm.update()
        wtqdm.close()

if __name__ == "__main__":
    flags.mark_flags_as_required([ ])
    app.run(main)
