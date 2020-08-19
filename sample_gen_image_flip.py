import os
import glob
import numpy as np
import random
import pandas as pd

import cv2
import skimage.io
import openslide
import skimage.io
from skimage.transform import resize, rescale
import albumentations as A

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.metrics as metrics

from typing import Optional

def compute_statistics(image):
    width, height = image.shape[0], image.shape[1]
    num_pixels = width * height

    num_white_pixels = 0

    summed_matrix = np.sum(image, axis=-1)
    num_white_pixels = np.count_nonzero(summed_matrix > 700)
    ratio_white_pixels = num_white_pixels / num_pixels
    green_concentration = np.mean(image[:, :, 1])
    blue_concentration = np.mean(image[:, :, 2])

    return (
        ratio_white_pixels,
        green_concentration,
        blue_concentration,
    )


def select_k_best_regions(regions, k=20):
    k_best_regions = sorted(regions, key=lambda tup: tup[2])[:k]
    return k_best_regions

def get_k_best_regions(coordinates, image, window_size=512):
    regions = {}
    for i, tup in enumerate(coordinates):
        x, y = tup[0], tup[1]
        regions[i] = image[x : x + window_size, y : y + window_size, :]

    return regions


def detect_best_window_size(image, K=16, scaling_factor=1.0):
    (ratio_white_pixels, green_concentration, blue_concentration,) = compute_statistics(image)
    h, w = image.shape[:2]
    return max(int(np.sqrt(h * w * (1.0 - ratio_white_pixels) * scaling_factor / K)), 32,)


def generate_patches(
    slide_path, window_size=128, stride=128, k=20, auto_ws=False, scaling_factor=1.0,
):

    image = skimage.io.MultiImage(slide_path)[2]
    image = np.array(image)

    if auto_ws:
        window_size = detect_best_window_size(image, K=k, scaling_factor=scaling_factor)
        stride = window_size

    max_width, max_height = image.shape[0], image.shape[1]
    regions_container = []
    i = 0

    while window_size + stride * i <= max_height:
        j = 0

        while window_size + stride * j <= max_width:
            x_top_left_pixel = j * stride
            y_top_left_pixel = i * stride

            patch = image[
                x_top_left_pixel : x_top_left_pixel + window_size, y_top_left_pixel : y_top_left_pixel + window_size, :,
            ]

            (ratio_white_pixels, green_concentration, blue_concentration,) = compute_statistics(patch)

            region_tuple = (
                x_top_left_pixel,
                y_top_left_pixel,
                ratio_white_pixels,
                green_concentration,
                blue_concentration,
            )
            regions_container.append(region_tuple)

            j += 1

        i += 1

    k_best_region_coordinates = select_k_best_regions(regions_container, k=k)
    k_best_regions = get_k_best_regions(k_best_region_coordinates, image, window_size)

    return (
        image,
        k_best_region_coordinates,
        k_best_regions,
        window_size,
    )


def glue_to_one_picture_from_coord(url, coordinates, window_size=200, k=16, layer=0):
    side = int(np.sqrt(k))
    slide = openslide.OpenSlide(url)
    lv2_scale = slide.level_downsamples[2]
    scale = slide.level_downsamples[2] / slide.level_downsamples[layer]
    image = np.full((int(side * window_size * scale), int(side * window_size * scale), 3,), 255, dtype=np.uint8,)
    for i, patch_coord in enumerate(coordinates):
        x = i // side
        y = i % side
        patch = np.asarray(slide.read_region((int(patch_coord[1] * lv2_scale), int(patch_coord[0] * lv2_scale),), layer, (int(window_size * scale), int(window_size * scale),),))[:, :, :3]
        image[
            int(x * window_size * scale) : int(x * window_size * scale) + int(window_size * scale), int(y * window_size * scale) : int(y * window_size * scale) + int(window_size * scale), :,
        ] = patch
    slide.close()
    return image

def glue_to_one_picture_from_coord_lowlayer(url, coordinates, window_size=200, k=16, layer=1):
    side = int(np.sqrt(k))
    slide = openslide.OpenSlide(url)
    lv2_scale = slide.level_downsamples[2]
    scale = slide.level_downsamples[2] / slide.level_downsamples[layer]
    slide.close()

    slide = skimage.io.MultiImage(url)[layer]
    slide = np.array(slide)

    image = np.full((int(side * window_size * scale), int(side * window_size * scale), 3,), 255, dtype=np.uint8,)
    for i, patch_coord in enumerate(coordinates):
        x = i // side
        y = i % side
        patch = slide[
            int(patch_coord[0] * scale) : int(patch_coord[0] * scale) + int(window_size * scale), int(patch_coord[1] * scale) : int(patch_coord[1] * scale) + int(window_size * scale), :,
        ]
        image[int(x * window_size * scale) : int(x * window_size * scale) + patch.shape[0], int(y * window_size * scale) : int(y * window_size * scale) + patch.shape[1], :,] = patch
    return image


def glue_to_one_picture(image_patches, window_size=200, k=16):
    side = int(np.sqrt(k))
    image = np.zeros((side * window_size, side * window_size, 3), dtype=np.uint8,)
    for i, patch in image_patches.items():
        x = i // side
        y = i % side
        image[x * window_size : (x + 1) * window_size, y * window_size : (y + 1) * window_size, :,] = patch
    return image


def load_img(
    img_name, K=16, scaling_factor=1.0, layer=0, auto_ws=True, window_size=128,
):
    WINDOW_SIZE = window_size
    STRIDE = window_size
    # K = 16
    (image, best_coordinates, best_regions, win,) = generate_patches(img_name, window_size=WINDOW_SIZE, stride=STRIDE, k=K, auto_ws=auto_ws, scaling_factor=scaling_factor,)
    WINDOW_SIZE = win
    STRIDE = WINDOW_SIZE
    # print(win)
    # glued_image = glue_to_one_picture(best_regions, window_size=WINDOW_SIZE, k=K)
    if layer == 0:
        glued_image = glue_to_one_picture_from_coord(img_name, best_coordinates, window_size=WINDOW_SIZE, k=K, layer=layer,)
    else:
        glued_image = glue_to_one_picture_from_coord_lowlayer(img_name, best_coordinates, window_size=WINDOW_SIZE, k=K, layer=layer,)
    return glued_image

#### for flip TTA

def generate_patches_list(
    slide_path, window_size=128, stride=128, k=20, auto_ws=False, scaling_factor=1.0,
):
    base_image = skimage.io.MultiImage(slide_path)[2]
    base_image = np.array(base_image)

    if auto_ws:
        window_size = detect_best_window_size(base_image, K=k, scaling_factor=scaling_factor)
        stride = window_size

    k_best_region_coordinates = [ ]

    max_width, max_height = base_image.shape[0], base_image.shape[1]

    for q in range(4):
        if q == 0:
            image = base_image
        elif q == 1:
            image = base_image[::-1, :, :]
        elif q == 2:
            image = base_image[:, ::-1, :]
        elif q == 3:
            image = base_image[::-1, ::-1, :]
        else:
            pass

        regions_container = []
        i = 0
        while window_size + stride * i <= max_height:
            j = 0
            while window_size + stride * j <= max_width:
                x_top_left_pixel = j * stride
                y_top_left_pixel = i * stride

                patch = image[
                    x_top_left_pixel : x_top_left_pixel + window_size, y_top_left_pixel : y_top_left_pixel + window_size, :,
                ]

                (ratio_white_pixels, green_concentration, blue_concentration,) = compute_statistics(patch)

                region_tuple = (
                    x_top_left_pixel,
                    y_top_left_pixel,
                    ratio_white_pixels,
                    green_concentration,
                    blue_concentration,
                )
                regions_container.append(region_tuple)
                j += 1
            i += 1
        obj = select_k_best_regions(regions_container, k=k)
        k_best_region_coordinates.append(obj)
        # k_best_regions = get_k_best_regions(k_best_region_coordinates, image, window_size)
    return k_best_region_coordinates, window_size

def glue_to_one_picture_from_coord_list_lowlayer(url, coordinates, window_size=200, k=16, layer=1):
    side = int(np.sqrt(k))
    slide = openslide.OpenSlide(url)
    lv2_scale = slide.level_downsamples[2]
    scale = slide.level_downsamples[2] / slide.level_downsamples[layer]
    slide.close()

    slide = skimage.io.MultiImage(url)[layer]
    slide = np.array(slide)

    nums = len(coordinates)
    
    image = np.full((nums, int(side * window_size * scale), int(side * window_size * scale), 3), 255, dtype=np.uint8,)
    for q in range(nums):
        if q == 0:
            buff = slide
        elif q == 1:
            buff = slide[::-1, :, :]
        elif q == 2:
            buff = slide[:, ::-1, :]
        elif q == 3:
            buff = slide[::-1, ::-1, :]
        else:
            pass
        for i, patch_coord in enumerate(coordinates[q]):
            x, y = divmod(i, side)
            patch = buff[
                int(patch_coord[0] * scale) : int(patch_coord[0] * scale) + int(window_size * scale), int(patch_coord[1] * scale) : int(patch_coord[1] * scale) + int(window_size * scale), :,
            ]
            image[q, int(x * window_size * scale) : int(x * window_size * scale) + patch.shape[0], int(y * window_size * scale) : int(y * window_size * scale) + patch.shape[1], :,] = patch

        work = buff.copy()
        for i, patch_coord in enumerate(coordinates[q]):
            r1 = int(patch_coord[0] * scale)
            r2 = int(patch_coord[0] * scale) + int(window_size * scale)
            c1 = int(patch_coord[1] * scale)
            c2 = int(patch_coord[1] * scale) + int(window_size * scale)
            cv2.rectangle(work, (c1, r1), (c2, r2), (255, 0, 0), thickness=8)
        if q == 0:
            pass
        elif q == 1:
            work = work[::-1, :, :]
        elif q == 2:
            work = work[:, ::-1, :]
        elif q == 3:
            work = work[::-1, ::-1, :]
        else:
            pass
        work = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        cv2.imwrite('flip_test_mark_{}.png'.format(q + 1), work)

    return image


def load_img_with_shift_for_DEBUG(
    img_name, K=16, scaling_factor=1.0, layer=1, auto_ws=True, window_size=128,
):
    WINDOW_SIZE = window_size
    STRIDE = window_size
    # K = 16
    best_coordinates, win = generate_patches_list(img_name, window_size=WINDOW_SIZE, stride=STRIDE, k=K, auto_ws=auto_ws, scaling_factor=scaling_factor)
    WINDOW_SIZE = win
    STRIDE = WINDOW_SIZE

    if layer == 0:
        raise ValueError('only support layer >= 1')
    else:
        glued_images = glue_to_one_picture_from_coord_list_lowlayer(img_name, best_coordinates, window_size=WINDOW_SIZE, k=K, layer=layer,)
    return glued_images, best_coordinates

def load(df, idx):
    data_dir = 'data/prostate-cancer-grade-assessment'
    img_name = os.path.join(os.path.join(data_dir, 'train_images/'), df.loc[idx, 'image_id'] + '.tiff')
    print(img_name)
    buffer, region = load_img_with_shift_for_DEBUG(img_name, K=16, scaling_factor=2.0, layer=1, auto_ws=True, window_size=128)
    # images = np.full((buffer.shape[0], 2048, 2048, 3), 255, dtype=np.uint8)
    # for q in range(buffer.shape[0]):
    #     images[q, :, :, :] = cv2.resize(buffer[q, :, :, :], (2048, 2048))
    # return images
    return buffer, region

def main():
    tdf = pd.read_csv('data/prostate-cancer-grade-assessment/train.csv')
    img, reg = load(tdf, 1000)
    for i in range(4):
        buf = img[i, :, :, :]
        buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
        cv2.imwrite('flip_test_{}.png'.format(i + 1), buf)
        """
        print('[')
        for j in reg[i]:
            print('{:5},{:5}'.format(j[0], j[1]))
        print(']')
        """

if '__main__' == __name__:
    main()
