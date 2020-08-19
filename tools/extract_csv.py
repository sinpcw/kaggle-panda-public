#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import glob
import pandas as pd
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', './', '画像ファイルがあるディレクトリを指定する.')
flags.DEFINE_string('input_csv', 'train.csv', '学習データCSVを指定する.')
flags.DEFINE_string('output_csv', 'extract_train.csv', '出力する学習データCSVを指定する.')

def extract(df, root_dir):
    files = sorted(glob.glob(os.path.join(root_dir, '*.tiff')))
    files = [ os.path.basename(f)[:-5] for f in files if not f.endswith('_mask.tiff') ]
    return df[df['image_id'].isin(files)]

def main(argv):
    df = pd.read_csv(FLAGS.input_csv)
    df = extract(df, FLAGS.input_dir)
    df.to_csv(FLAGS.output_csv, index=None)

if __name__ == '__main__':
    app.run(main)