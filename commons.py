#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import tqdm
import numpy as np
import random
import torch
from torch import distributed
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import cohen_kappa_score, confusion_matrix
try:
    import neptune
    AVAILABLE_NEPTUNE = True
except ModuleNotFoundError:
    AVAILABLE_NEPTUNE = False

# DistributedDataParallelの状態確認
def IsDistributed():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False

# 各種の乱数シードを設定する
def SetSeed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# absl: 引数パラメータを出力する
def SaveArgs(filepath, command):
    with open(filepath, mode='w') as f:
        for k, v in command.items():
            f.write('--{}={} \\\n'.format(k, v))

# absl: 引数を辞書型に変換する
def LoadArgs(args):
    return { f : args[f].value for f in args }

# 辞書型から値を取得する
def GetValue(conf, keyword, default):
    return (conf[keyword] if keyword is not None and keyword in conf else default)

# 辞書型から小文字化した文字列値を取得する
def GetLowerName(conf, keyword, default):
    return (conf[keyword] if keyword is not None and keyword in conf else default).lower()

# Tensorboardラッパー
class WrapTensorboard(object):
    def __init__(self, log_dir='./'):
        super(WrapTensorboard, self).__init__()
        self.writer = None
        if log_dir is not None and os.path.exists(log_dir):
            self.writer = SummaryWriter(log_dir=log_dir)

    def __del__(self):
        self.close()

    def open(self, log_dir):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            raise UserWarning('TensorBoard(SummryWriter)は既に設定されています.')

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def writeScalar(self, keyword, x, value):
        if self.writer is not None:
            if type(value) == dict:
                self.writer.add_scalars(keyword, value, x)
            else:
                self.writer.add_scalar(keyword, value, x)

# tqdmラッパー
class WrapTqdm(object):
    def __init__(self, run='local', **kwargs):
        super(WrapTqdm, self).__init__()
        if run is None or run.lower() == 'local':
            self.instance = tqdm.tqdm(**kwargs)
        else:
            self.instance = None

    def __del__(self):
        self.close()

    def set_description(self, text):
        if self.instance is not None:
            self.instance.set_description(text, refresh=True)

    def update(self):
        if self.instance is not None:
            self.instance.update()

    def close(self):
        if self.instance is not None:
            self.instance.close()
            self.instance = None

# Quaderic Weighted Kapper
def qwk(y_pred, y_true):
    return cohen_kappa_score(y_pred, y_true, weights='quadratic')

# Quaderic Weighted Kapper (extension)
def qwk_ext(y1, y2, labels=None, weights=None, sample_weight=None, verbose=False):
    confusion = confusion_matrix(y1, y2, labels=labels, sample_weight=sample_weight)
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    if weights is None:
        w_mat = np.ones([n_classes, n_classes], dtype=np.int)
        w_mat.flat[::n_classes + 1] = 0
    elif weights == "linear" or weights == "quadratic":
        w_mat = np.zeros([n_classes, n_classes], dtype=np.int)
        w_mat += np.arange(n_classes)
        if weights == "linear":
            w_mat = np.abs(w_mat - w_mat.T)
        else:
            w_mat = (w_mat - w_mat.T) ** 2
    else:
        raise ValueError("Unknown kappa weighting type.")
    m = (n_classes - 1) ** 2
    o = np.sum(w_mat * confusion)/(np.sum(confusion) * m)
    e = np.sum(w_mat * expected)/(np.sum(confusion) * m)
    k = o / e
    if verbose:
        print(confusion)
    return 1 - k, o, e

# NeptuneLoggerラッパー
class WrapNeptuneLogger(object):
    def __init__(self, project, name, token, activate=True):
        self.__active__ = False
        if AVAILABLE_NEPTUNE and activate:
            neptune.init(project_qualified_name=project, api_token=token)
            neptune.create_experiment(name=name)
            self.__active__ = True

    def __del__(self):
        self.close()

    def open(self, project, name, token):
        if AVAILABLE_NEPTUNE and not self.__active__:
            neptune.init(project_qualified_name=project, api_token=token)
            neptune.create_experiment(name=name)
            self.__active__ = True

    def close(self):
        if AVAILABLE_NEPTUNE and self.__active__:
            neptune.stop()
            self.__active__ = False

    def write(self, data):
        if AVAILABLE_NEPTUNE and self.__active__:
            for k, v in data.items():
                neptune.log_metric(k, v)

def GetNeptuneLogger(name, activate):
    token = '(set your token)'
    # return WrapNeptuneLogger('(set your project)', name, token, activate=activate)
    return WrapNeptuneLogger('(set your project)', name, token, activate=False)
