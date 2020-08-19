import os
import tqdm
import numpy as np
import pandas as pd
from openslide import OpenSlide
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input_csv', 'train.csv', '入力データCSVを指定する.')
flags.DEFINE_string('input_dir', './', '入力ディレクトリを指定する.')
flags.DEFINE_string('output_csv', 'train_out.csv', '出力ディレクトリを指定する.')
flags.DEFINE_integer('level', 2, '変換する対象レベルを指定する.')

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
    idf = pd.read_csv(FLAGS.input_csv)
    out = [
        'image_id,data_provider,isup_grade,gleason_score\n'
    ]
    item = tqdm.tqdm(total=len(idf))
    for i in range(len(idf)):
        image_id = idf.iat[i, 0]
        img_path = os.path.abspath(os.path.join(FLAGS.input_dir, 'train_label_masks', image_id + '_mask.tiff'))
        if not os.path.exists(img_path):
            continue
        osi = OpenSlideImage(img_path)
        if FLAGS.level >= osi.level():
            continue
        mat = osi.load(FLAGS.level)[:, :, 0].astype(np.uint8)
        cnt = np.count_nonzero(mat)
        if cnt / (mat.shape[0] * mat.shape[1]) < 0.05:
            continue
        col3 = idf.iat[i, 3] if idf.iat[i, 3] != 'negative' else '0+0'
        out.append('{},{},{},{}\n'.format(
            idf.iat[i, 0],
            idf.iat[i, 1],
            idf.iat[i, 2],
            col3,
        ))
        item.update()
    item.close()
    with open(FLAGS.output_csv, mode='w') as f:
        for itr in out:
            f.write(itr)

if __name__ == '__main__':
    app.run(main)