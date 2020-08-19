#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import glob
import pandas as pd
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('input_csv', 'csvs/train.csv', '学習データCSVを指定する.')
flags.DEFINE_string('output_dir', './csvs', '分割した学習データCSVを出力するディレクトリを指定する.')
flags.DEFINE_integer('nfold', 8, 'フォールド数を指定する.')

def split_dataprovider(df):
    kdf = df[df['data_provider'] == 'karolinska']
    rdf = df[df['data_provider'] == 'radboud']
    return kdf, rdf

def main(argv):
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    df = pd.read_csv(FLAGS.input_csv)
    kdf, rdf = split_dataprovider(df)
    kdf.to_csv(os.path.join(FLAGS.output_dir, 'train_kdf.csv'), index=None)
    rdf.to_csv(os.path.join(FLAGS.output_dir, 'train_rdf.csv'), index=None)
    kdfs = kdf.sort_values('isup_grade')
    rdfs = rdf.sort_values('isup_grade')
    for n in range(FLAGS.nfold):
        x = kdfs.iloc[n::FLAGS.nfold, :]
        x.to_csv(os.path.join(FLAGS.output_dir, 'train_kdf_k{}.csv'.format(n)), index=None)
        y = rdfs.iloc[n::FLAGS.nfold, :]
        y.to_csv(os.path.join(FLAGS.output_dir, 'train_rdf_k{}.csv'.format(n)), index=None)
        z = pd.concat([x, y])
        z.to_csv(os.path.join(FLAGS.output_dir, 'train_sdf_k{}.csv'.format(n)), index=None)

if __name__ == '__main__':
    app.run(main)