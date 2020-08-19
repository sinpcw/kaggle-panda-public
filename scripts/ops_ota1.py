import os
import cv2
import PIL
import random
import openslide
import skimage.io
import matplotlib
import numpy as np
import pandas as pd

# train_df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv').sample(n=100, random_state=0).reset_index(drop=True)
# train_df = pd.read_csv('./data/prostate-cancer-grade-assessment/train.csv')
train_df = pd.read_csv('/kqi/parent/train.csv')

images = list(train_df['image_id'])
labels = list(train_df['isup_grade'])

# data_dir = '../input/prostate-cancer-grade-assessment/train_images/'
# data_dir = './data/prostate-cancer-grade-assessment/train_images/'
data_dir = '/kqi/parent/train_images/'

def compute_statistics(image):
    width, height = image.shape[0], image.shape[1]
    num_pixels = width * height
    num_white_pixels = 0

    summed_matrix = np.sum(image, axis=-1)
    # Note: A 3-channel white pixel has RGB (255, 255, 255)
    num_white_pixels = np.count_nonzero(summed_matrix > 700)
    #num_white_pixels = np.count_nonzero(summed_matrix > 255*3-1)
    ratio_white_pixels = num_white_pixels / num_pixels

    #red_concentration = np.mean(image[:,:,0])
    #red_concentration = np.mean(image[:,:,0][summed_matrix!=255*3])    
    green_concentration = np.mean(image[:,:,1])
    #green_concentration = np.mean(image[:,:,1][summed_matrix<200*3])
    blue_concentration = np.mean(image[:,:,2])
    #blue_concentration = np.mean(image[:,:,2][summed_matrix<200*3])

    return ratio_white_pixels, green_concentration, blue_concentration

def get_k_best_regions(coordinates, image, window_size=512):
    regions = {}
    for i, tup in enumerate(coordinates):
        x, y = tup[0], tup[1]
        regions[i] = image[x : x+window_size, y : y+window_size, :]
    return regions

def select_k_best_regions(regions, k=20):
    #regions = [x for x in regions if x[3] > 100 and x[4] > 100]
    k_best_regions = sorted(regions, key=lambda tup: tup[2])[:k]
    return k_best_regions

def detect_best_window_size(image,K=16):
    #image = skimage.io.MultiImage(slide_path)[2]
    #image = np.array(image)
    ratio_white_pixels, green_concentration, blue_concentration = compute_statistics(image)
    # print(ratio_white_pixels, green_concentration, blue_concentration)
    h, w = image.shape[:2]
    return int(np.sqrt(h * w * (1.0 - ratio_white_pixels)* 1.0 / K))

def generate_patches(slide_path, window_size=128, stride=128, k=20, auto_ws=False):
    image = skimage.io.MultiImage(slide_path)[2]
    image = np.array(image)
    if auto_ws:
        window = detect_best_window_size(image,K=K)
        # bug-fix:
        if window > 0:
            window_size = window
        # -------:
        stride = window_size
    max_width, max_height = image.shape[0], image.shape[1]
    regions_container = []
    i = 0
    while window_size + stride*i <= max_height:
        j = 0
        while window_size + stride*j <= max_width:
            x_top_left_pixel = j * stride
            y_top_left_pixel = i * stride
            patch = image[
                x_top_left_pixel : x_top_left_pixel + window_size,
                y_top_left_pixel : y_top_left_pixel + window_size,
                :
            ]
            ratio_white_pixels, green_concentration, blue_concentration = compute_statistics(patch)
            region_tuple = (x_top_left_pixel, y_top_left_pixel, ratio_white_pixels, green_concentration, blue_concentration)
            regions_container.append(region_tuple)
            j += 1
        i += 1
    k_best_region_coordinates = select_k_best_regions(regions_container, k=k)
    k_best_regions = get_k_best_regions(k_best_region_coordinates, image, window_size)
    return image, k_best_region_coordinates, k_best_regions, window_size

def glue_to_one_picture(image_patches, window_size=200, k=16):
    side = int(np.sqrt(k))
    image = np.zeros((side*window_size, side*window_size, 3), dtype=np.int16)
    for i, patch in image_patches.items():
        x = i // side
        y = i % side
        image[
            x * window_size : (x+1) * window_size,
            y * window_size : (y+1) * window_size,
            :
        ] = patch
    return image

def glue_to_one_picture_from_coord(url, coordinates, window_size=200, k=16):
    side = int(np.sqrt(k))
    slide = openslide.OpenSlide(url)
    scale = slide.level_downsamples[2]
    # print(scale)
    image = np.zeros((int(side*window_size*scale), int(side*window_size*scale), 3), dtype=np.int16)
    # print(coordinates)
    for i, patch_coord in enumerate(coordinates):
        x = i // side
        y = i % side
        patch = np.asarray(slide.read_region((int(patch_coord[1]*scale), int(patch_coord[0]*scale)), 0, (int(window_size*scale),int(window_size*scale))))[:,:,:3]
        image[
            int(x * window_size*scale) : int(x * window_size*scale) + int(window_size*scale),
            int(y * window_size*scale) : int(y * window_size*scale) + int(window_size*scale),
            :
        ] = patch
    return image

WINDOW_SIZE = 128
STRIDE = 128
R = 4
C = 4
K = R * C
# RESOLUTIONS = [ 1024, 1536, 2048, 3072, 4096 ]
RESOLUTIONS = [ 1024, 1536, 2048 ]

pen_marked_images = [
    'fd6fe1a3985b17d067f2cb4d5bc1e6e1',
    'ebb6a080d72e09f6481721ef9f88c472',
    'ebb6d5ca45942536f78beb451ee43cc4',
    'ea9d52d65500acc9b9d89eb6b82cdcdf',
    'e726a8eac36c3d91c3c4f9edba8ba713',
    'e90abe191f61b6fed6d6781c8305fe4b',
    'fd0bb45eba479a7f7d953f41d574bf9f',
    'ff10f937c3d52eff6ad4dd733f2bc3ac',
    'feee2e895355a921f2b75b54debad328',
    'feac91652a1c5accff08217d19116f1c',
    'fb01a0a69517bb47d7f4699b6217f69d',
    'f00ec753b5618cfb30519db0947fe724',
    'e9a4f528b33479412ee019e155e1a197',
    'f062f6c1128e0e9d51a76747d9018849',
    'f39bf22d9a2f313425ee201932bac91a',
]

for res in RESOLUTIONS:
    nw = res // 4
    nh = res // 4
    # os.makedirs('data/process/ota1/{}x{}/train_images/'.format(nw, nh), exist_ok=True)
    os.makedirs('/kqi/output/{}x{}/train_images/'.format(nw, nh), exist_ok=True)

for i, img in enumerate(images[:]):
    url = data_dir + img + '.tiff'
    image, best_coordinates, best_regions, win = generate_patches(url, window_size=WINDOW_SIZE, stride=STRIDE, k=K, auto_ws=True)
    WINDOW_SIZE = win
    STRIDE = WINDOW_SIZE
    glued_image = glue_to_one_picture_from_coord(url, best_coordinates, window_size=WINDOW_SIZE, k=K)
    glued_image = cv2.cvtColor(glued_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    dw = glued_image.shape[1] // 4
    dh = glued_image.shape[0] // 4
    for res in RESOLUTIONS:
        nw = res // 4
        nh = res // 4
        dump = np.zeros([res, res, 3], dtype=np.uint8)
        for r in range(4):
            for c in range(4):
                buff = glued_image[r*dh:(r+1)*dh, c*dw:(c+1)*dw, :]
                buff = cv2.resize(buff, (nw, nh))
                dump[r*nh:(r+1)*nh, c*nw:(c+1)*nw, :] = buff
        # cv2.imwrite('data/process/ota1/{}x{}/train_images/'.format(nw, nh) + img + '.png', dump)
        cv2.imwrite('/kqi/output/{}x{}/train_images/'.format(nw, nh) + img + '_cmb.png', dump)
