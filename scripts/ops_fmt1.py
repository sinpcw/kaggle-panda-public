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
# Environment:
flags.DEFINE_enum('run', 'local', [ 'local', 'server' ], '実行マシンを指定します.')

""" ヘルパー """
# ディレクトリ設定
def SetupOutput(conf):
    os.makedirs(conf['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(conf['output_dir'], 'train_images'), exist_ok=True)
    return conf

# 現在時間を記録
def Now():
    t = datetime.datetime.now()
    return t.strftime('%Y/%m/%d %H:%M:%S')

""" 推論用関数 """
def Padding(img, patch, value):
    h, w = img.shape[:2]
    ph = (patch - (h % patch)) if h % patch > 0 else 0
    pw = (patch - (w % patch)) if w % patch > 0 else 0
    pshape = [[0, ph], [0, pw], [0, 0]] if len(img.shape) == 3 else [[0, ph], [0, pw]]
    return np.pad(img, pshape, constant_values=value)

def GetItem(df, image_dir, idx):
    idn = df.iat[idx, 0]
    osi = datasets.OpenSlideImage(os.path.join(image_dir, 'train_images', idn + '.tiff'))
    return osi, osi.load(1), idn

""" メイン関数 """
def main(argv):
    conf = commons.LoadArgs(FLAGS)
    conf = SetupOutput(conf)
    # 入力データ:
    patch = conf['patch']
    target_df = pd.read_csv(conf['input_csv'])
    # モデル:
    print('[{}] Start Preprocess'.format(Now()))
    count = len(target_df)
    wtqdm = commons.WrapTqdm(total=count, run=conf['run'])
    for i in range(count):
        osi, image, name = GetItem(target_df, conf['image_dir'], i)
        image = Padding(image, patch, 255)
        nh = image.shape[0] // patch
        nw = image.shape[1] // patch
        # 輝度抽出:
        values = [ ]
        for hi in range(nh):
            for wi in range(nw):
                roi = image[hi * patch:(hi + 1) * patch, wi * patch:(wi + 1) * patch, :].astype(np.uint8)
                values.append((hi, wi, np.sum(roi)))
        values = sorted(values, key=lambda x: x[2])
        # 前処理: タイルプロセス
        rows = 4
        cols = 4
        mat = np.ones([rows * patch, cols * patch, 3], dtype=np.uint8) * 255
        for j, part in enumerate(values):
            if rows * cols <= j:
                break
            r = part[0]
            c = part[1]
            cs = patch * (c)
            cd = patch * (c + 1)
            rs = patch * (r)
            rd = patch * (r + 1)
            rect = (
                int(cs),
                int(rs),
                int(cd),
                int(rd),
            )
            img = osi.load_rect(1, rect, fill=255)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (patch, patch))
            row, col = divmod(j, cols)
            mat[row * patch:(row+1) * patch, col * patch:(col+1) * patch, :] = img[:, :, :]
        cv2.imwrite(os.path.join(conf['output_dir'], 'train_images', '{}_cmb.png'.format(name)), mat)
        wtqdm.update()
    wtqdm.close()

if __name__ == "__main__":
    flags.mark_flags_as_required([ ])
    app.run(main)
