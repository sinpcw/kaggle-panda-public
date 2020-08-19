import os
import numpy as np
import cv2
import glob
from openslide import OpenSlide
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', './', '入力ディレクトリを指定する.')
flags.DEFINE_string('output_dir', './output', '出力ディレクトリを指定する.')
flags.DEFINE_integer('level', 1, '変換する対象レベルを指定する.')

# OpenSlideImage
class OpenSlideImage(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.instance = OpenSlide(filepath)

    def open(self, filepath):
        self.filepath = filepath
        self.instance = OpenSlide(filepath)

    def size(self):
        return self.instance.dimensions

    def level(self):
        return self.instance.level_count

    def get_size(self, level):
        return self.instance.level_dimensions[level]

    def load(self, level):
        p = (0, 0)
        s = self.get_size(level)
        return np.asarray(self.instance.read_region(p, level, s))

def main(argv):
    level = FLAGS.level

    os.makedirs(os.path.join(FLAGS.output_dir, 'train_images'), exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(FLAGS.input_dir, 'train_images', '*.tiff')))
    for img in imgs:
        name = os.path.basename(img)
        data = OpenSlideImage(img)
        mat = data.load(level)
        mat = mat[:, :, 0:3] # drop alpha
        cv2.imwrite(os.path.join(FLAGS.output_dir, 'train_images', name[:-5] + '.png'), mat)

    os.makedirs(os.path.join(FLAGS.output_dir, 'train_label_masks'), exist_ok=True)
    msks = sorted(glob.glob(os.path.join(FLAGS.input_dir, 'train_label_masks', '*.tiff')))
    for msk in msks:
        name = os.path.basename(msk)
        data = OpenSlideImage(msk)
        mat = data.load(level)
        mat = mat[:, :, 0:3] # drop alpha
        cv2.imwrite(os.path.join(FLAGS.output_dir, 'train_label_masks', name[:-5] + '.png'), mat)

if __name__ == '__main__':
    app.run(main)