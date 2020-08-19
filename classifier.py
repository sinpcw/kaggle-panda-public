import os
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score
from absl import app, flags

FLAGS = flags.FLAGS

def qwk(y_true, y_pred):
    return cohen_kappa_score(y_pred, y_true, weights='quadratic')

def capture(feats, train, valid):
    td = { }
    for i in range(len(train)):
        td[train.iat[i, 0]] = train.iat[i, 2]
    vd = { }
    for i in range(len(valid)):
        vd[valid.iat[i, 0]] = valid.iat[i, 2]
    train_f = { }
    train_l = { }
    valid_f = { }
    valid_l = { }
    columns = len(feats.columns)
    for i in range(len(feats)):
        ids = feats.iat[i, 0]
        obj = feats.iloc[i, 2:columns]
        if ids in td:
            train_f[ids] = obj
            train_l[ids] = td[ids]
        else:
            valid_f[ids] = obj
            valid_l[ids] = vd[ids]
    x_train = pd.DataFrame(train_f.values(), index=train_f.keys())
    y_train = pd.DataFrame(train_l.values(), index=train_l.keys())
    x_valid = pd.DataFrame(valid_f.values(), index=valid_f.keys())
    y_valid = pd.DataFrame(valid_l.values(), index=valid_l.keys())
    return x_train, y_train, x_valid, y_valid

def main(argv):
    # df = pd.read_csv('data/prostate-cancer-grade-assessment/train.csv')
    train = pd.read_csv('csvs/cls3_kfold_0/nfold_train.csv')
    valid = pd.read_csv('csvs/cls3_kfold_0/nfold_valid.csv')
    print('load feature_vector:')
    feats = None
    for id in range(20):
        f = 'feat2/feature_vector_{:02}.csv'.format(id + 1)
        if not os.path.exists(f):
            break
        c = pd.read_csv(f)
        feats = pd.concat([feats, c]) if feats is not None else c
        print('  step: {}'.format(id + 1))
    print('  row: {}'.format(len(feats)))
    print('  col: {}'.format(len(feats.columns)))
    print('setup dataset:')
    x_train, y_train, x_valid, y_valid = capture(feats, train, valid)
    train_data = lgb.Dataset(x_train, label=y_train)
    valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data)
    cparams = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 6,
        'verbose': 0,
    }
    print('LightGBM Train Start:')
    print('  train = {}'.format(len(x_train)))
    print('  valid = {}'.format(len(x_valid)))
    gbm = lgb.train(cparams, train_data, valid_sets=valid_data, num_boost_round=150, verbose_eval=5)
    y_true = [ y_valid.iat[i, 0] for i in range(len(y_valid)) ]
    y_pred = [ ]
    predicts = gbm.predict(x_valid)
    for x in predicts:
        y_pred.append(np.argmax(x))
    acc = accuracy_score(y_true, y_pred)
    val = qwk(y_true, y_pred)
    print('ACC={:.5f}, QWK={:.5f}'.format(acc, val))

    with open('classifier.pkl', mode='wb') as f:
        pickle.dump(gbm, f)

if __name__ == '__main__':
    app.run(main)
