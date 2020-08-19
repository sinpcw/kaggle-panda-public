#!/usr/bin/bash
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
from openslide import OpenSlide
from absl import app, flags
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string('train_csv', '/kqi/parent/train.csv', '入力CSV(.csv)を指定します')
flags.DEFINE_string('image_dir', '/kqi/parent/train_images', '入力画像ディレクトリを指定します')
flags.DEFINE_string('output_dir', '/kqi/output', '出力先ディレクトリを指定します.')
flags.DEFINE_integer('image_size', 256, 'タイル1枚のサイズを指定します.')

""" OTA Data Create Module """
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

def generate_patches(filepath, window, stride, K, auto_ws, scaling_factor):
    slide = OpenSlide(filepath)
    image = np.asarray(slide.read_region((0, 0), 2, slide.level_dimensions[2]))[:, :, :3]
    image = np.array(image)
    if auto_ws:
        window = detect_best_window_size(image, 16, scaling_factor)
        stride = window
    h, w = image.shape[:2]
    regions = []
    j = 0
    while window + stride * j <= h:
        i = 0
        while window + stride * i <= w:
            x_start = i * stride
            y_start = j * stride
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
    for i, itr in enumerate(regions):
        r, c = divmod(i, block)
        ws = int(window * scale)
        patch = np.asarray(slide.read_region((int(itr[0] * slide.level_downsamples[2]), int(itr[1] * slide.level_downsamples[2])), layer, (ws, ws)))[:, :, :3]
        r0 = int(r * window * scale)
        r1 = r0 + ws
        c0 = int(c * window * scale)
        c1 = c0 + ws
        image[r0:r1, c0:c1, :] = patch
    slide.close()
    return image

def LoadImage(filepath, K, scaling_factor, layer=0):
    window = 128
    stride = 128
    _, regions, _, window = generate_patches(filepath, window=window, stride=stride, K=K, auto_ws=True, scaling_factor=scaling_factor)
    if window <= 32 * (4 ** layer):
        print('[WARNING] window size = {} (filepath={}, layer={})'.format(window, filepath, layer))
    image = glue_to_one_picture_from_coord(filepath, regions, window=window, K=K, layer=layer)
    return image

def main(argv):
    train_csv = pd.read_csv(FLAGS.train_csv)
    image_dir = FLAGS.image_dir
    image_col = 4
    image_size = FLAGS.image_size
    output_dir = FLAGS.output_dir
    scalelist = [ 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 ]
    for scale in scalelist:
        os.makedirs(os.path.join(output_dir, 'scale_{:.1f}'.format(scale), 'train_images'), exist_ok=True)
    t = tqdm(total=len(train_csv))
    mat = np.zeros([image_col * image_size, image_col * image_size, 3], dtype=np.uint8)
    for idx in range(len(train_csv)):
        imgpath = os.path.join(image_dir, 'train_images', train_csv.iat[idx, 0] + '.tiff')
        for scale in scalelist:
            img = LoadImage(imgpath, image_col * image_col, scale)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(os.path.join(output_dir, 'scale_{:.1f}'.format(scale), train_csv.iat[idx, 0] + '.png'), img)
            tilesize = img.shape[0] // 4
            for j in range(image_col * image_col):
                r, c = divmod(j, image_col)
                r0 = image_size * (r)
                r1 = image_size * (r+1)
                c0 = image_size * (c)
                c1 = image_size * (c+1)
                buf = cv2.resize(img[tilesize * r:tilesize * (r+1), tilesize * c:tilesize * (c+1), :], (image_size, image_size))
                buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
                mat[r0:r1, c0:c1, :] = buf
            cv2.imwrite(os.path.join(output_dir, 'scale_{:.1f}'.format(scale), 'train_images', train_csv.iat[idx, 0] + '.png'), mat)
        t.update()
    t.close()
    print('Complete:')
    print('  Scale Factors = {}'.format(scalelist))

if __name__ == '__main__':
    app.run(main)