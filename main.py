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
import losses
import models
import optims
import commons
import datasets
import itertools
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel, convert_syncbn_model
    AVAILABLE_AMP = True
except ModuleNotFoundError:
    AVAILABLE_AMP = False

""" 引数 """
FLAGS = flags.FLAGS
# Random:
flags.DEFINE_integer('seed', 20200609, '乱数シードを指定します.')
# Data I/O:
flags.DEFINE_string('image_dir', '../input/', '入力ディレクトリを指定します.')
flags.DEFINE_string('train_csv', 'train.csv', '学習用入力データ(train.csv)を指定します.')
flags.DEFINE_string('valid_csv', 'valid.csv', '検証用入力データ(valid.csv)を指定します.')
flags.DEFINE_string('output_dir', './output', '出力先ディレクトリを指定します.')
flags.DEFINE_integer('num_images', 16, 'REG/CLS: 画像パッチ使用数を指定します.')
flags.DEFINE_integer('gen_images', 16, 'REG/CLS: 画像パッチ生成数を指定します. Nの2乗値を設定する必要があります.')
flags.DEFINE_integer('jitter', 0, 'REG/CLS: 画像パッチ作成に使用するジッター量を指定します.')
# Environment:
flags.DEFINE_enum('run', 'local', [ 'local', 'server' ], '実行マシンを指定します.')
flags.DEFINE_string('exec', 'train', '実行タスクを指定します.')
flags.DEFINE_string('device', None, '実行デバイスを指定します.')
# Auto Mixed Precision:
flags.DEFINE_bool('amp', True, 'AMP(Auto-Mixed-Precision)の使用有無を指定します.')
flags.DEFINE_string('amp_level', 'O1', 'AMP(Auto-Mixed-Precision)のfp16化レベルを指定します.')
# Training Parameter:
flags.DEFINE_integer('epoch', 100, 'エポック数を指定します.')
flags.DEFINE_integer('batch', 1, 'バッチサイズを指定します.')
# Model:
flags.DEFINE_string('model', 'custom_efficientnet-b3', 'モデル名を指定します.')
flags.DEFINE_string('model_uri', None, 'モデルパラメータファイルパスを指定します.')
flags.DEFINE_string('fetch_uri', None, 'モデルパラメータファイルパスを指定します. 既定のパラメータ以外を使用する場合に設定します.')
# Optimizer:
flags.DEFINE_enum('optimizer', 'adam', [ 'sgd', 'adam', 'adamw', 'adabound', 'radam' ], '最適化手法を指定します.')
flags.DEFINE_float('lr', 1e-3, '学習率を指定します')
flags.DEFINE_float('momentum', 0.9, 'SGD: momentumの係数を指定します.')
flags.DEFINE_bool('nesterov', False, 'SGD: Nesterov 加速勾配の有効/無効を指定します.')
flags.DEFINE_float('weight_decay', 0, 'SGD/Adam/AdamW: weight_decayの係数を指定します.')
# Scheduler:
flags.DEFINE_string('scheduler', 'none', '学習率減衰のスケジュールを指定します.')
flags.DEFINE_float('lr_min', 0, '学習率減衰のスケジュールの最小学習率を指定します.')
flags.DEFINE_float('finish', 100, '学習率減衰のスケジュール完了エポック数を指定します.')
flags.DEFINE_float('warmup', 0, '学習率減衰のスケジュールのウォームアップエポック数を指定します.')
flags.DEFINE_string('milestones', '20,40,60,80', 'steplr: 減衰するエポック数を指定します.')
flags.DEFINE_float('gamma', 0.7, 'steplr: 減衰する係数を指定します.')
# Augment:
flags.DEFINE_string('train_augs', None, '学習オーグメンテーション指定します.')
flags.DEFINE_string('image_augs', None, '学習オーグメンテーション指定します. ここで指定したオーグメンテーションはパッチ化前画像全体に適用します.')
flags.DEFINE_string('valid_augs', None, '検証オーグメンテーション指定します.')
flags.DEFINE_string('tpost_augs', None, '学習オーグメンテーション指定します. ここで指定したオーグメンテーションはパッチ化後画像全体に適用します.')
flags.DEFINE_string('vpost_augs', None, '検証オーグメンテーション指定します. ここで指定したオーグメンテーションはパッチ化後画像全体に適用します.')
# Loss:
flags.DEFINE_string('loss', 'mse', 'REG: 損失関数を指定します.')
# Optional:
flags.DEFINE_float('grad_clip', None, 'REG: 勾配を抑制する値を指定します.')
flags.DEFINE_float('flood', None, 'REG: flood値を設定します.')
# DataLoader:
flags.DEFINE_integer('num_workers', 0, 'ワーカー数を指定します.')
# DataLoader:
flags.DEFINE_string('reg_loader', 'default', 'REG: データローダーを選択します.')
flags.DEFINE_string('cls_loader', 'default', 'CLS: データローダーを選択します.')
# Mix:
flags.DEFINE_string('mix_select', 'cls', 'MIX: スコアを出力対象を選択します (cls/reg).')
# neptune:
flags.DEFINE_bool('neptune', False, 'neptuneロガーを使用有無を指定します.')

flags.DEFINE_integer('accumerate', 0, 'Batch Accumerateの蓄積数を指定します.')

flags.DEFINE_bool('sync_batch', False, 'syncBatch.')

""" ヘルパー """
# Device設定
def SetupDevice(conf):
    if 'device' in conf and conf['device'] is not None:
        if conf['device'] == 'cuda' and not torch.cuda.is_available():
            print('[警告] CUDAデバイスが使用できません. CPUモードで実行します.')
            conf['device'] = 'cpu'
    else:
        conf['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return conf

# ディレクトリ設定
def SetupOutput(conf):
    os.makedirs(conf['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(conf['output_dir'], 'csvs'), exist_ok=True)
    return conf

# Auto Mixed Precision設定
def SetupAMP(conf, model, optim):
    if optim is not None:
        if AVAILABLE_AMP and conf['amp']:
            model, optim = amp.initialize(model, optim, opt_level=conf['amp_level'])
        return model, optim
    else:
        if AVAILABLE_AMP and conf['amp']:
            model = amp.initialize(model, opt_level=conf['amp_level'])
        return model

# 混同行列出力
def OutputCFM(conf, name, y_pred, y_true):
    cfm = np.zeros([6, 6], dtype=np.int32)
    for i in range(len(y_pred)):
        cfm[y_pred[i], y_true[i]] += 1
    with open(os.path.join(conf['output_dir'], 'csvs/cfm_' + name + '.csv'), mode='w') as f:
        f.write(',,y_true\n')
        f.write(',,0,1,2,3,4,5\n')
        f.write('y_pred,{},{},{},{},{},{},{}\n'.format(0, cfm[0, 0], cfm[0, 1], cfm[0, 2], cfm[0, 3], cfm[0, 4], cfm[0, 5]))
        for i in range(1, 6):
            f.write(',{},{},{},{},{},{},{}\n'.format(i, cfm[i, 0], cfm[i, 1], cfm[i, 2], cfm[i, 3], cfm[i, 4], cfm[i, 5]))
    with open(os.path.join(conf['output_dir'], 'acc_' + name + '.csv'), mode='w') as f:
        f.write('y_pred,y_true\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(y_pred[i], y_true[i]))

# 現在時間を記録
def Now():
    t = datetime.datetime.now()
    return t.strftime('%Y/%m/%d %H:%M:%S')

""" メイン関数 """
def main(argv):
    conf = commons.LoadArgs(FLAGS)
    conf = SetupOutput(conf)
    conf = SetupDevice(conf)
    commons.SetSeed(conf['seed'])
    if conf['exec'] in [ 'train+reg', 'train+cls' ]:
        function = train_regcls
    elif conf['exec'] in [ 'valid+reg', 'valid+cls', 'valid+reg+mfs' ]:
        function = valid_regcls
    elif conf['exec'] in [ 'train+reg+mfs', 'train+cls+mfs' ]:
        function = train_regcls_mfs
    elif conf['exec'] in [ 'train+reg+wls' ]:
        function = train_regcls_wls
    elif conf['exec'] in [ 'valid+reg+wls' ]:
        function = valid_regcls_wls
    elif conf['exec'] in [ 'train+reg+las', 'train+cls+las' ]:
        function = train_regcls_las
    elif conf['exec'] in [ 'valid+reg+las', 'valid+cls+las' ]:
        function = valid_regcls_las
    elif conf['exec'] in [ 'valid+reg+esm','valid+cls+esm' ]:
        function = valid_regcls_esm
    else:
        raise NameError('指定されたタスクは定義されていません (--exec={})'.format(conf['exec']))
    function(conf)

""" Regression/Classification """
def train_regcls(conf):
    run = conf['exec'][6:]
    # 入力データ:
    train_df = pd.read_csv(conf['train_csv'])
    valid_df = pd.read_csv(conf['valid_csv'])
    train_aug = datasets.GetTrainAugment(run, conf['train_augs'])
    valid_aug = datasets.GetValidAugment(run, conf['valid_augs'])
    image_aug = datasets.GetTrainAugment(run, conf['image_augs']) if conf['image_augs'] is not None else None
    tpost_aug = datasets.GetTrainAugment(run, conf['tpost_augs']) if conf['tpost_augs'] is not None else None
    vpost_aug = datasets.GetValidAugment(run, conf['vpost_augs']) if conf['vpost_augs'] is not None else None
    train_loader = datasets.GetTrainDataLoader(conf, run, train_df, augment_op=train_aug, patched_op=tpost_aug, combine_op=image_aug)
    valid_loader = datasets.GetValidDataLoader(conf, run, valid_df, augment_op=valid_aug, patched_op=vpost_aug)
    train_steps = len(train_loader) // conf['batch']
    # モデル:
    num_classes = 1 if run == 'reg' else 6
    if conf['fetch_uri'] is not None:
        pretrained = conf['fetch_uri']
    else:
        pretrained = True
    model = models.GetModel(conf, num_classes=num_classes, pretrained=pretrained, uri=conf['model_uri']).to(conf['device'])
    optim = optims.GetOptimizer(conf, model.parameters())
    model, optim = SetupAMP(conf, model, optim)
    model = torch.nn.DataParallel(model)
    scheduler, require_call_everystep = optims.GetScheduler(conf, optim, lr=conf['lr'], epoch=conf['epoch'], steps=train_steps, max_lr=conf['lr'], min_lr=conf['lr'] * 0.02, gamma=conf['gamma'])
    # 損失関数タイプ:
    train_loss_fn = losses.GetTrainLossFunction(conf).to(conf['device'])
    valid_loss_fn = losses.GetValidLossFunction(conf).to(conf['device'])
    # 学習主処理:
    epoch = conf['epoch']
    score_t = 0
    score_k = 0
    # ロガー設定:
    train_name = os.environ.get('TRAINING_ID', 'unknown') + '_{}'.format(run.upper())
    wboard = commons.WrapTensorboard(log_dir=conf['output_dir'])
    neplog = commons.GetNeptuneLogger(train_name, activate=conf['neptune'])
    # 学習開始:
    print('[{}] Start Train: {}'.format(Now(), run.upper()))
    for e in range(epoch):
        train_loss, y_pred_t, y_true_t = train_step_regcls(conf, model, train_loader, train_loss_fn, optim, scheduler if require_call_everystep else None)
        if not require_call_everystep:
            scheduler.step()
        # Loss Weightの更新:
        if isinstance(train_loss_fn, losses.CustomLoss):
            train_loss_fn.step()
        valid_loss, qwk, obs, exp, v_acc, y_pred, y_true = valid_step_regcls(conf, run, model, valid_loader, valid_loss_fn)
        # valid_dfからkarolinska/radboudでもそれぞれ計算する
        y_true_k = [ ]
        y_pred_k = [ ]
        y_true_r = [ ]
        y_pred_r = [ ]
        y_true_pri = [ ]
        y_pred_pri = [ ]
        y_true_pub = [ ]
        y_pred_pub = [ ]
        for i in range(len(valid_df)):
            if valid_df.iat[i, 1] == 'karolinska':
                y_true_k.append(y_true[i])
                y_pred_k.append(y_pred[i])
            else:
                y_true_r.append(y_true[i])
                y_pred_r.append(y_pred[i])
            if (valid_df.iat[i, 1] == 'karolinska' and valid_df.iat[i, 2] >= 3) or (valid_df.iat[i, 1] == 'radboud' and valid_df.iat[i, 2] < 3):
                y_true_pub.append(y_true[i])
                y_pred_pub.append(y_pred[i])
            else:
                y_true_pri.append(y_true[i])
                y_pred_pri.append(y_pred[i])
        t_acc = 0
        t_num = len(y_true_t)
        for i in range(t_num):
            if y_pred_t[i] == y_true_t[i]:
                t_acc += 1.0
        t_acc /= t_num
        qwk_k = commons.qwk(y_pred_k, y_true_k)
        qwk_r = commons.qwk(y_pred_r, y_true_r)
        qwk_t = commons.qwk(y_pred_t, y_true_t)
        sim_qwk_pri = commons.qwk(y_pred_pri, y_true_pri)
        sim_qwk_pub = commons.qwk(y_pred_pub, y_true_pub)
        # Log:
        updated = (score_t < qwk) or (score_k < qwk_k)
        log_str = '[{}] Epoch: {:4}/{:4}, lr: {:.3e}, t-loss: {:.4e}, v-loss: {:.4e}, t-acc: {:.3f}, v-acc: {:.3f}, qwk_t: {:.5f}, qwk_k: {:.5f}, qwk_r: {:.5f}, qwk: {:.5f}, pub-qwk {:.5f}, pri-qwk: {:.5f}'.format(Now(), e, epoch, optims.GetOptimierLR(optim), train_loss, valid_loss, t_acc, v_acc, qwk_t, qwk_k, qwk_r, qwk, sim_qwk_pub, sim_qwk_pri)
        log_str = log_str + (' (*)' if updated else '')
        print(log_str)
        # Tensorboard:
        wboard.writeScalar('lr', e, optims.GetOptimierLR(optim))
        wboard.writeScalar('qwk', e, qwk)
        wboard.writeScalar('qwk_karolinska', e, qwk_k)
        wboard.writeScalar('qwk_radboud', e, qwk_r)
        wboard.writeScalar('obs', e, obs)
        wboard.writeScalar('exp', e, exp)
        wboard.writeScalar('acc', e, { 'train': t_acc, 'valid': v_acc })
        wboard.writeScalar('loss', e, { 'train' : train_loss, 'valid' : valid_loss })
        wboard.writeScalar('sim_qwk', e, { 'public' : sim_qwk_pub, 'private' : sim_qwk_pri })
        # Neptune Logger:
        neplog.write({ 'val_qwk' : qwk, 'val_qwk_karolinska' : qwk_k, 'val_qwk_radboud' : qwk_r, 'val_obs': obs, 'val_exp': exp, 'avg_val_loss' : valid_loss, 'avg_train_loss' : train_loss })
        # Parameter:
        if updated:
            if score_t < qwk:
                score_t = qwk
            if score_k < qwk_k:
                score_k = qwk_k
            torch.save(model.module.state_dict(), os.path.join(conf['output_dir'], 'epoch{}.pth'.format(e)))
            OutputCFM(conf, 'epoch{}'.format(e), y_pred, y_true)
        # UserCommand:
        if os.path.exists('save') and not os.path.exists(os.path.join(conf['output_dir'], 'epoch{}.pth'.format(e))):
            print('[UserCommand] save parameter (epoch={}).'.format(e))
            torch.save(model.module.state_dict(), os.path.join(conf['output_dir'], 'epoch{}.pth'.format(e)))
            os.remove('save')
        if os.path.exists('exit') or os.path.exists('exit_{}'.format(e)):
            print('[UserCommand] exit train loop.')
            break
    torch.save(model.module.state_dict(), os.path.join(conf['output_dir'], 'final.pth'))
    wboard.close()
    neplog.close()

# 検証: regression/classification
def valid_regcls(conf):
    run = conf['exec'][6:]
    if conf['model_uri'] is None:
        print('モデルパラメータが指定されていません. --model_uriで有効なパラメータを指定してください.')
    # 入力データ:
    valid_df = pd.read_csv(conf['valid_csv'])
    valid_aug = datasets.GetValidAugment(run, conf['valid_augs'])
    vpost_aug = datasets.GetValidAugment(run, conf['vpost_augs'])
    valid_loader = datasets.GetValidDataLoader(conf, run, valid_df, augment_op=valid_aug, patched_op=vpost_aug)
    # モデル:
    model = models.GetModel(conf, num_classes=1, uri=conf['model_uri']).to(conf['device'])
    model = SetupAMP(conf, model, None)
    model = torch.nn.DataParallel(model)
    # 損失関数タイプ:
    loss_fn = losses.GetValidLossFunction(conf).to(conf['device'])
    print('[{}] Start Valid: {}'.format(Now(), run.upper()))
    valid_loss, qwk, obs, exp, acc, y_pred, y_true = valid_step_regcls(conf, run, model, valid_loader, loss_fn)
    # valid_dfからkarolinska/radboudでもそれぞれ計算する
    y_true_k = [ ]
    y_pred_k = [ ]
    y_true_r = [ ]
    y_pred_r = [ ]
    for i in range(len(valid_df)):
        if valid_df.iat[i, 1] == 'karolinska':
            y_true_k.append(y_true[i])
            y_pred_k.append(y_pred[i])
        else:
            y_true_r.append(y_true[i])
            y_pred_r.append(y_pred[i])
    qwk_k = commons.qwk(y_pred_k, y_true_k)
    qwk_r = commons.qwk(y_pred_r, y_true_r)
    # スコア表示
    log_str = '[{}] valid-loss: {:.6e}, acc: {:.5f}, qwk_k: {:.5f}, qwk_r: {:.5f}, qwk: {:.5f}, obs: {:.5f}, exp: {:.5f}'.format(Now(), valid_loss, acc, qwk_k, qwk_r, qwk, obs, exp)
    print(log_str)
    # 混同行列出力:
    OutputCFM(conf, 'valid', y_pred, y_true)

# 検証: regression/classification + esm
def valid_regcls_esm(conf):
    sts = conf['exec'][6:].split('+')
    run = sts[0]
    if conf['model_uri'] is None:
        print('モデルパラメータが指定されていません. --model_uriで有効なパラメータを指定してください.')
    uris = conf['model_uri'].split(':')
    # 入力データ:
    valid_df = pd.read_csv(conf['valid_csv'])
    valid_aug = datasets.GetValidAugment(run, conf['valid_augs'])
    valid_loader = datasets.GetValidDataLoader(conf, run, valid_df, augment_op=valid_aug)
    # モデル:
    model = [ ]
    for uri in uris:
        defs = uri.split('=')
        if len(defs) == 1:
            mdl = {
                'model' : conf['model']
            }
            uri = defs[0]
        else:
            mdl = {
                'model' : defs[0]
            }
            uri = defs[1]
        item = models.GetModel(mdl, num_classes=1, uri=uri).to(conf['device'])
        if conf['device'] != 'cpu':
            item = SetupAMP(conf, item, None)
            item = torch.nn.DataParallel(item)
        item.eval()
        model.append(item)
    # 損失関数タイプ:
    loss_fn = losses.GetValidLossFunction(conf).to(conf['device'])
    print('[{}] Start Valid: {}'.format(Now(), run.upper()))
    valid_loss, qwk, obs, exp, acc, y_pred, y_true = valid_step_regcls_esm(conf, run, model, valid_loader, loss_fn)
    log_str = '[{}] valid-loss: {:.6e}, acc: {:.5f}, qwk: {:.5f}, obs: {:.5f}, exp: {:.5f}'.format(Now(), valid_loss, acc, qwk, obs, exp)
    print(log_str)
    # 混同行列出力:
    OutputCFM(conf, 'valid_esm', y_pred, y_true)

# 学習ステップ (エポック処理)
def train_step_regcls(conf, model, loader, loss_fn, optim, scheduler):
    value = 0
    count = 0
    flood = conf['flood']
    device = conf['device']
    y_pred = [ ]
    y_true = [ ]
    zero = True
    accm = 0
    divs = 1 + conf['accumerate']
    model.train()
    wtqdm = commons.WrapTqdm(total=len(loader), run=conf['run'])
    for _, itr in enumerate(loader):
        x0, y0, w0 = itr
        x0 = x0.to(device)
        y0 = y0.to(device)
        w0 = w0.to(device)
        y_ = model(x0)
        # Batch Accumerate:
        if zero:
            zero = False
            accm = 0
            optim.zero_grad()
        # Compute Loss
        if type(loss_fn) in [ losses.OHEMLoss, losses.WeakLoss ]:
            loss = loss_fn(y_, y0)
        else:
            loss = torch.mean(w0 * loss_fn(y_, y0))
        if flood is not None:
            loss = (loss - flood).abs() + flood
        loss = loss / divs
        if AVAILABLE_AMP and conf['amp']:
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # Gradient Clip:
        if conf['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf['grad_clip'])
        # Batch Accumerate:
        if accm >= conf['accumerate']:
            optim.step()
            if scheduler is not None:
                scheduler.step()
            zero = True
            accm = 0
        else:
            accm = accm + 1
        # capture:
        y_ = y_.detach().cpu().numpy()
        y0 = y0.detach().cpu().numpy()
        for i in range(y_.shape[0]):
            y_pred.append(int(np.clip(y_[i] + 0.5, 0, 5)))
            y_true.append(int(y0[i]))
        value += conf['batch'] * loss.item()
        count += conf['batch']
        wtqdm.set_description('  train loss = {:e}'.format(value / count))
        wtqdm.update()
    wtqdm.close()
    value /= count
    return value, y_pred, y_true

# 検証ステップ (エポック処理)
def valid_step_regcls(conf, run, model, loader, loss_fn):
    acc_score = 0
    value = 0
    steps = 0
    count = len(loader)
    y_true = np.zeros([count], dtype=np.int)
    y_pred = np.zeros([count], dtype=np.int)
    device = conf['device']
    model.eval()
    with torch.no_grad():
        wtqdm = commons.WrapTqdm(total=count, run=conf['run'])
        for i, itr in enumerate(loader):
            x0, y0, _ = itr
            x0 = x0.to(device)
            y0 = y0.to(device)
            y_ = model(x0)
            loss = loss_fn(y_, y0)
            value += loss.item()
            steps += 1
            wtqdm.set_description('  valid loss = {:e}'.format(value / steps))
            wtqdm.update()
            y0 = y0.detach().cpu().numpy()
            y_ = y_.detach().cpu().numpy()
            if run == 'reg':
                y_ = np.clip(y_ + 0.5, 0, 5)
            else:
                y_ = np.argmax(y_)
            y_pred[i] = int(y_)
            y_true[i] = int(y0)
            acc_score = acc_score + (1 if int(y_) == int(y0) else 0)
        wtqdm.close()
    qwk_score, obs_score, exp_score = commons.qwk_ext(y_pred, y_true, weights='quadratic')
    acc_score = acc_score / steps
    value /= steps
    return value, qwk_score, obs_score, exp_score, acc_score, y_pred, y_true

# 検証ステップ (エポック処理)
def valid_step_regcls_esm(conf, run, model, loader, loss_fn):
    acc_score = 0
    value = 0
    steps = 0
    count = len(loader)
    ratio = 1.0 / len(model)
    y_true = np.zeros([count], dtype=np.int)
    y_pred = np.zeros([count], dtype=np.int)
    device = conf['device']
    with torch.no_grad():
        wtqdm = commons.WrapTqdm(total=count, run=conf['run'])
        for i, itr in enumerate(loader):
            x0, y0, _ = itr
            x0 = x0.to(device)
            y0 = y0.to(device)
            y_ = None
            """
            for mdl in model:
                z_ = mdl(x0)
                loss = loss_fn(z_, y0)
                value += loss.item()
                steps += 1
                y_ = (ratio * z_) if y_ is None else (ratio * z_ + y_)
            """
            z_ = torch.zeros([len(model)], device=conf['device'])
            for k, mdl in enumerate(model):
                z_[k] = mdl(x0)
                loss = loss_fn(z_[k], y0)
                value += loss.item()
                steps += 1
            d_ = torch.max(z_) - torch.min(z_)
            # if True:
            if d_ > 1.5:
                y_ = torch.max(z_)
            else:
                y_ = torch.mean(z_)
            wtqdm.set_description('  valid loss = {:e}'.format(value / steps))
            wtqdm.update()
            y0 = y0.detach().cpu().numpy()
            y_ = y_.detach().cpu().numpy()
            if run == 'reg':
                y_ = np.clip(y_ + 0.5, 0, 5)
            else:
                y_ = np.argmax(y_)
            y_pred[i] = int(y_)
            y_true[i] = int(y0)
            acc_score = acc_score + (1 if int(y_) == int(y0) else 0)
        wtqdm.close()
    qwk_score, obs_score, exp_score = commons.qwk_ext(y_pred, y_true, weights='quadratic')
    acc_score = acc_score / steps
    value /= steps
    return value, qwk_score, obs_score, exp_score, acc_score, y_pred, y_true

""" Regression/Classification """
# 学習: regression/classification
def train_regcls_mfs(conf):
    sts = conf['exec'][6:].split('+')
    run = sts[0]
    # 入力データ:
    train_df = pd.read_csv(conf['train_csv'])
    valid_df = pd.read_csv(conf['valid_csv'])
    train_aug = datasets.GetTrainAugment(run, conf['train_augs'])
    valid_aug = datasets.GetValidAugment(run, conf['valid_augs'])
    image_aug = datasets.GetTrainAugment(run, conf['image_augs']) if conf['image_augs'] is not None else None
    tpost_aug = datasets.GetTrainAugment(run, conf['tpost_augs']) if conf['tpost_augs'] is not None else None
    vpost_aug = datasets.GetValidAugment(run, conf['vpost_augs']) if conf['vpost_augs'] is not None else None
    train_loader = datasets.GetTrainDataLoader(conf, run, train_df, augment_op=train_aug, patched_op=tpost_aug, combine_op=image_aug)
    train_loader.dataset.shuffle() # for DEBUG
    valid_loader = datasets.GetValidDataLoader(conf, run, valid_df, augment_op=valid_aug, patched_op=vpost_aug)
    train_steps = len(train_loader) // conf['batch']
    # モデル:
    num_classes = 1 if run == 'reg' else 6
    if conf['fetch_uri'] is not None:
        pretrained = conf['fetch_uri']
    else:
        pretrained = True
    model = models.GetModel(conf, num_classes=num_classes, pretrained=pretrained, uri=conf['model_uri']).to(conf['device'])
    optim = optims.GetOptimizer(conf, model.parameters())
    model, optim = SetupAMP(conf, model, optim)
    model = torch.nn.DataParallel(model)
    scheduler, require_call_everystep = optims.GetScheduler(conf, optim, lr=conf['lr'], epoch=conf['epoch'], steps=train_steps, max_lr=conf['lr'], min_lr=conf['lr'] * 0.02, gamma=conf['gamma'])
    # 損失関数タイプ:
    train_loss_fn = losses.GetTrainLossFunction(conf).to(conf['device'])
    valid_loss_fn = losses.GetValidLossFunction(conf).to(conf['device'])
    # 学習主処理:
    epoch = conf['epoch']
    score_t = 0
    score_k = 0
    # ロガー設定:
    train_name = os.environ.get('TRAINING_ID', 'unknown') + '_{}'.format(run.upper())
    wboard = commons.WrapTensorboard(log_dir=conf['output_dir'])
    neplog = commons.GetNeptuneLogger(train_name, activate=conf['neptune'])
    # 学習開始:
    print('[{}] Start Train: {}'.format(Now(), run.upper()))
    for e in range(epoch):
        train_loss, y_pred_t, y_true_t = train_step_regcls_mfs(conf, model, train_loader, train_loss_fn, optim, scheduler if require_call_everystep else None)
        train_loader.dataset.shuffle()
        if not require_call_everystep:
            scheduler.step()
        # Loss Weightの更新:
        if isinstance(train_loss_fn, losses.CustomLoss):
            train_loss_fn.step()
        valid_loss, qwk, obs, exp, v_acc, y_pred, y_true = valid_step_regcls(conf, run, model, valid_loader, valid_loss_fn)
        # valid_dfからkarolinska/radboudでもそれぞれ計算する
        y_true_k = [ ]
        y_pred_k = [ ]
        y_true_r = [ ]
        y_pred_r = [ ]
        for i in range(len(valid_df)):
            if valid_df.iat[i, 1] == 'karolinska':
                y_true_k.append(y_true[i])
                y_pred_k.append(y_pred[i])
            else:
                y_true_r.append(y_true[i])
                y_pred_r.append(y_pred[i])
        t_acc = 0
        t_num = len(y_true_t)
        for i in range(t_num):
            if y_pred_t[i] == y_true_t[i]:
                t_acc += 1.0
        t_acc /= t_num
        qwk_k = commons.qwk(y_pred_k, y_true_k)
        qwk_r = commons.qwk(y_pred_r, y_true_r)
        qwk_t = commons.qwk(y_pred_t, y_true_t)
        # Log:
        updated = (score_t < qwk) or (score_k < qwk_k)
        log_str = '[{}] Epoch: {:4}/{:4}, lr: {:.3e}, t-loss: {:.4e}, v-loss: {:.4e}, t-acc: {:.3f}, v-acc: {:.3f}, qwk_t: {:.5f}, qwk_k: {:.5f}, qwk_r: {:.5f}, qwk: {:.5f}, obs: {:.5f}, exp: {:.5f}'.format(Now(), e, epoch, optims.GetOptimierLR(optim), train_loss, valid_loss, t_acc, v_acc, qwk_t, qwk_k, qwk_r, qwk, obs, exp)
        log_str = log_str + (' (*)' if updated else '')
        print(log_str)
        # Tensorboard:
        wboard.writeScalar('lr', e, optims.GetOptimierLR(optim))
        wboard.writeScalar('qwk', e, qwk)
        wboard.writeScalar('qwk_karolinska', e, qwk_k)
        wboard.writeScalar('qwk_radboud', e, qwk_r)
        wboard.writeScalar('obs', e, obs)
        wboard.writeScalar('exp', e, exp)
        wboard.writeScalar('acc', e, { 'train': t_acc, 'valid': v_acc })
        wboard.writeScalar('loss', e, { 'train' : train_loss, 'valid' : valid_loss })
        # Neptune Logger:
        neplog.write({ 'val_qwk' : qwk, 'val_qwk_karolinska' : qwk_k, 'val_qwk_radboud' : qwk_r, 'val_obs': obs, 'val_exp': exp, 'avg_val_loss' : valid_loss, 'avg_train_loss' : train_loss })
        # Parameter:
        if updated:
            if score_t < qwk:
                score_t = qwk
            if score_k < qwk_k:
                score_k = qwk_k
            torch.save(model.module.state_dict(), os.path.join(conf['output_dir'], 'epoch{}.pth'.format(e)))
            OutputCFM(conf, 'epoch{}'.format(e), y_pred, y_true)
        # UserCommand:
        if os.path.exists('save') and not os.path.exists(os.path.join(conf['output_dir'], 'epoch{}.pth'.format(e))):
            print('[UserCommand] save parameter (epoch={}).'.format(e))
            torch.save(model.module.state_dict(), os.path.join(conf['output_dir'], 'epoch{}.pth'.format(e)))
            os.remove('save')
        if os.path.exists('exit') or os.path.exists('exit_{}'.format(e)):
            print('[UserCommand] exit train loop.')
            break
    torch.save(model.module.state_dict(), os.path.join(conf['output_dir'], 'final.pth'))
    wboard.close()
    neplog.close()

# 学習ステップ (エポック処理)
def train_step_regcls_mfs(conf, model, loader, loss_fn, optim, scheduler):
    value = 0
    count = 0
    flood = conf['flood']
    device = conf['device']
    y_pred = [ ]
    y_true = [ ]
    model.train()
    wtqdm = commons.WrapTqdm(total=len(loader), run=conf['run'])
    for _, itr in enumerate(loader):
        x0, y0 = itr
        b, n, c, h, w = x0.shape
        x0 = x0.view(b * n, c, h, w)
        y0 = y0.view(b * n, 1)
        x0 = x0.to(device)
        y0 = y0.to(device)
        y_ = model(x0)
        if type(loss_fn) in [ losses.OHEMLoss, losses.WeakLoss ]:
            loss = loss_fn(y_, y0)
        else:
            loss = loss_fn(y_, y0)
            loss = torch.mean(loss)
        if flood is not None:
            loss = (loss - flood).abs() + flood
        optim.zero_grad()
        if AVAILABLE_AMP and conf['amp']:
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # Gradient Clip:
        if conf['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf['grad_clip'])
        optim.step()
        if scheduler is not None:
            scheduler.step()
        # capture:
        y_ = y_.detach().cpu().numpy()
        y0 = y0.detach().cpu().numpy()
        for i in range(y_.shape[0]):
            y_pred.append(int(np.clip(y_[i] + 0.5, 0, 5)))
            y_true.append(int(y0[i]))
        value += conf['batch'] * loss.item()
        count += conf['batch']
        wtqdm.set_description('  train loss = {:e}'.format(value / count))
        wtqdm.update()
    wtqdm.close()
    value /= count
    return value, y_pred, y_true

""" Regression/Classification (label and autoscale) """
# 学習: regression/classification
def train_regcls_las(conf):
    sts = conf['exec'][6:].split('+')
    run = sts[0]
    # 入力データ:
    train_df = pd.read_csv(conf['train_csv'])
    valid_df = pd.read_csv(conf['valid_csv'])
    train_aug = datasets.GetTrainAugment(run, conf['train_augs'])
    valid_aug = datasets.GetValidAugment(run, conf['valid_augs'])
    image_aug = datasets.GetTrainAugment(run, conf['image_augs']) if conf['image_augs'] is not None else None
    train_loader = datasets.GetTrainDataLoader(conf, run, train_df, augment_op=train_aug, combine_op=image_aug)
    valid_loader = datasets.GetValidDataLoader(conf, run, valid_df, augment_op=valid_aug)
    train_steps = len(train_loader) // conf['batch']
    # モデル:
    num_classes = 1 if run == 'reg' else 6
    if conf['fetch_uri'] is not None:
        pretrained = conf['fetch_uri']
    else:
        pretrained = True
    model = models.GetModel(conf, num_classes=num_classes, pretrained=pretrained, uri=conf['model_uri'], las_process=True).to(conf['device'])
    optim = optims.GetOptimizer(conf, model.parameters())
    model, optim = SetupAMP(conf, model, optim)
    model = torch.nn.DataParallel(model)
    scheduler, require_call_everystep = optims.GetScheduler(conf, optim, lr=conf['lr'], epoch=conf['epoch'], steps=train_steps, max_lr=conf['lr'], min_lr=conf['lr'] * 0.02, gamma=conf['gamma'])
    # 損失関数タイプ:
    train_loss_fn = losses.GetTrainLossFunction(conf).to(conf['device'])
    valid_loss_fn = losses.GetValidLossFunction(conf).to(conf['device'])
    # 学習主処理:
    epoch = conf['epoch']
    score_t = 0
    score_k = 0
    # ロガー設定:
    train_name = os.environ.get('TRAINING_ID', 'unknown') + '_{}'.format(run.upper())
    wboard = commons.WrapTensorboard(log_dir=conf['output_dir'])
    neplog = commons.GetNeptuneLogger(train_name, activate=conf['neptune'])
    # 学習開始:
    print('[{}] Start Train: {}'.format(Now(), run.upper()))
    for e in range(epoch):
        train_loss = train_step_regcls_las(conf, model, train_loader, train_loss_fn, optim, scheduler if require_call_everystep else None)
        if not require_call_everystep:
            scheduler.step()
        # Loss Weightの更新:
        if isinstance(train_loss_fn, losses.CustomLoss):
            train_loss_fn.step()
        valid_loss, qwk, obs, exp, acc, y_pred, y_true = valid_step_regcls_las(conf, run, model, valid_loader, valid_loss_fn)
        # valid_dfからkarolinska/radboudでもそれぞれ計算する
        y_true_k = [ ]
        y_pred_k = [ ]
        y_true_r = [ ]
        y_pred_r = [ ]
        for i in range(len(valid_df)):
            if valid_df.iat[i, 1] == 'karolinska':
                y_true_k.append(y_true[i])
                y_pred_k.append(y_pred[i])
            else:
                y_true_r.append(y_true[i])
                y_pred_r.append(y_pred[i])
        qwk_k = commons.qwk(y_pred_k, y_true_k)
        qwk_r = commons.qwk(y_pred_r, y_true_r)
        # Log:
        updated = (score_t < qwk) or (score_k < qwk_k)
        log_str = '[{}] Epoch: {:4}/{:4}, lr: {:.3e}, t-loss: {:.6e}, v-loss: {:.6e}, acc: {:.5f}, qwk_k: {:.5f}, qwk_r: {:.5f}, qwk: {:.5f}, obs: {:.5f}, exp: {:.5f}'.format(Now(), e, epoch, optims.GetOptimierLR(optim), train_loss, valid_loss, acc, qwk_k, qwk_r, qwk, obs, exp)
        log_str = log_str + (' (*)' if updated else '')
        print(log_str)
        # Tensorboard:
        wboard.writeScalar('lr', e, optims.GetOptimierLR(optim))
        wboard.writeScalar('qwk', e, qwk)
        wboard.writeScalar('qwk_karolinska', e, qwk_k)
        wboard.writeScalar('qwk_radboud', e, qwk_r)
        wboard.writeScalar('obs', e, obs)
        wboard.writeScalar('exp', e, exp)
        wboard.writeScalar('acc', e, acc)
        wboard.writeScalar('loss', e, { 'train' : train_loss, 'valid' : valid_loss })
        # Neptune Logger:
        neplog.write({ 'val_qwk' : qwk, 'val_qwk_karolinska' : qwk_k, 'val_qwk_radboud' : qwk_r, 'val_obs': obs, 'val_exp': exp, 'avg_val_loss' : valid_loss, 'avg_train_loss' : train_loss })
        # Parameter:
        if updated:
            if score_t < qwk:
                score_t = qwk
            if score_k < qwk_k:
                score_k = qwk_k
            torch.save(model.module.state_dict(), os.path.join(conf['output_dir'], 'epoch{}.pth'.format(e)))
            OutputCFM(conf, 'epoch{}'.format(e), y_pred, y_true)
        # UserCommand:
        if os.path.exists('save') and not os.path.exists(os.path.join(conf['output_dir'], 'epoch{}.pth'.format(e))):
            print('[UserCommand] save parameter (epoch={}).'.format(e))
            torch.save(model.module.state_dict(), os.path.join(conf['output_dir'], 'epoch{}.pth'.format(e)))
            os.remove('save')
        if os.path.exists('exit') or os.path.exists('exit_{}'.format(e)):
            print('[UserCommand] exit train loop.')
            break
    torch.save(model.module.state_dict(), os.path.join(conf['output_dir'], 'final.pth'))
    wboard.close()
    neplog.close()

# 検証: regression/classification
def valid_regcls_las(conf):
    sts = conf['exec'][6:].split('+')
    run = sts[0]
    if conf['model_uri'] is None:
        print('モデルパラメータが指定されていません. --model_uriで有効なパラメータを指定してください.')
    # 入力データ:
    valid_df = pd.read_csv(conf['valid_csv'])
    valid_aug = datasets.GetValidAugment(run, conf['valid_augs'])
    valid_loader = datasets.GetValidDataLoader(conf, run, valid_df, augment_op=valid_aug)
    # モデル:
    model = models.GetModel(conf, num_classes=1, uri=conf['model_uri'], las_process=False).to(conf['device'])
    model = SetupAMP(conf, model, None)
    model = torch.nn.DataParallel(model)
    # 損失関数タイプ:
    # validationについては比較するためにREGはMSEで固定, CLSは選択式
    loss_fn = (losses.GetValidLossFunction('mse') if run == 'reg' else losses.GetValidLossFunction(conf['loss'])).to(conf['device'])
    print('[{}] Start Valid: {}'.format(Now(), run.upper()))
    valid_loss, qwk, obs, exp, acc, y_pred, y_true = valid_step_regcls_las(conf, run, model, valid_loader, loss_fn)
    # valid_dfからkarolinska/radboudでもそれぞれ計算する
    y_true_k = [ ]
    y_pred_k = [ ]
    y_true_r = [ ]
    y_pred_r = [ ]
    for i in range(len(valid_df)):
        if valid_df.iat[i, 1] == 'karolinska':
            y_true_k.append(y_true[i])
            y_pred_k.append(y_pred[i])
        else:
            y_true_r.append(y_true[i])
            y_pred_r.append(y_pred[i])
    qwk_k = commons.qwk(y_pred_k, y_true_k)
    qwk_r = commons.qwk(y_pred_r, y_true_r)
    # スコア表示
    log_str = '[{}] valid-loss: {:.6e}, acc: {:.5f}, qwk_k: {:.5f}, qwk_r: {:.5f}, qwk: {:.5f}, obs: {:.5f}, exp: {:.5f}'.format(Now(), valid_loss, acc, qwk_k, qwk_r, qwk, obs, exp)
    print(log_str)
    # 混同行列出力:
    OutputCFM(conf, 'valid', y_pred, y_true)

# 学習ステップ (エポック処理)
def train_step_regcls_las(conf, model, loader, loss_fn, optim, scheduler):
    value = 0
    count = 0
    device = conf['device']
    model.train()
    wtqdm = commons.WrapTqdm(total=len(loader), run=conf['run'])
    for _, itr in enumerate(loader):
        x0, y0, w0 = itr
        x0 = x0.to(device)
        y0 = y0.to(device)
        w0 = w0.to(device)
        yk, yr = model(x0)
        loss = w0 * loss_fn(yk, y0) + (1 - w0) * loss_fn(yr, y0) 
        loss = torch.mean(loss)
        optim.zero_grad()
        if AVAILABLE_AMP and conf['amp']:
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # Gradient Clip:
        if conf['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf['grad_clip'])
        optim.step()
        if scheduler is not None:
            scheduler.step()
        value += conf['batch'] * loss.item()
        count += conf['batch']
        wtqdm.set_description('  train loss = {:e}'.format(value / count))
        wtqdm.update()
    wtqdm.close()
    value /= count
    return value

# 検証ステップ (エポック処理)
def valid_step_regcls_las(conf, run, model, loader, loss_fn):
    acc_score = 0
    value = 0
    steps = 0
    count = len(loader)
    y_true = np.zeros([count], dtype=np.int)
    y_pred = np.zeros([count], dtype=np.int)
    device = conf['device']
    model.eval()
    with torch.no_grad():
        wtqdm = commons.WrapTqdm(total=count, run=conf['run'])
        for i, itr in enumerate(loader):
            x0, y0, w0 = itr
            x0 = x0.to(device)
            y0 = y0.to(device)
            w0 = w0.to(device)
            yk, yr = model(x0)
            loss = loss_fn(yk, y0) * w0 + loss_fn(yr, y0) * (1 - w0)
            loss = torch.mean(loss)
            value += loss.item()
            steps += 1
            wtqdm.set_description('  valid loss = {:e}'.format(value / steps))
            wtqdm.update()
            y0 = y0.detach().cpu().numpy()
            yk = yk.detach().cpu().numpy()
            yr = yr.detach().cpu().numpy()
            w0 = w0.detach().cpu().numpy()
            # karolinska/radboud select
            if w0 == 1.0:
                y_ = yk
            else:
                y_ = yr
            if run == 'reg':
                y_ = np.clip(y_ + 0.5, 0, 5)
            else:
                y_ = np.argmax(y_)
            y_pred[i] = int(y_)
            y_true[i] = int(y0)
            acc_score = acc_score + (1 if int(y_) == int(y0) else 0)
        wtqdm.close()
    qwk_score, obs_score, exp_score = commons.qwk_ext(y_pred, y_true, weights='quadratic')
    acc_score = acc_score / steps
    value /= steps
    return value, qwk_score, obs_score, exp_score, acc_score, y_pred, y_true


""" Regression/Classification (weighted label score) """
# 学習: regression/classification
def train_regcls_wls(conf):
    sts = conf['exec'][6:].split('+')
    run = sts[0]
    # 入力データ:
    train_df = pd.read_csv(conf['train_csv'])
    valid_df = pd.read_csv(conf['valid_csv'])
    train_aug = datasets.GetTrainAugment(run, conf['train_augs'])
    valid_aug = datasets.GetValidAugment(run, conf['valid_augs'])
    image_aug = datasets.GetTrainAugment(run, conf['image_augs']) if conf['image_augs'] is not None else None
    train_loader = datasets.GetTrainDataLoader(conf, run, train_df, augment_op=train_aug, combine_op=image_aug)
    valid_loader = datasets.GetValidDataLoader(conf, run, valid_df, augment_op=valid_aug)
    train_steps = len(train_loader) // conf['batch']
    # モデル:
    num_classes = 1 if run == 'reg' else 6
    if conf['fetch_uri'] is not None:
        pretrained = conf['fetch_uri']
    else:
        pretrained = True
    model = models.GetModel(conf, num_classes=num_classes, pretrained=pretrained, uri=conf['model_uri'], las_process=True).to(conf['device'])
    optim = optims.GetOptimizer(conf, model.parameters())
    model, optim = SetupAMP(conf, model, optim)
    model = torch.nn.DataParallel(model)
    scheduler, require_call_everystep = optims.GetScheduler(conf, optim, lr=conf['lr'], epoch=conf['epoch'], steps=train_steps, max_lr=conf['lr'], min_lr=conf['lr'] * 0.02, gamma=conf['gamma'])
    # 損失関数タイプ:
    loss_fns = [
        torch.nn.MSELoss(reduction='none'),
        torch.nn.SmoothL1Loss(reduction='none')
    ]
    # 学習主処理:
    epoch = conf['epoch']
    score_t = 0
    score_k = 0
    # ロガー設定:
    train_name = os.environ.get('TRAINING_ID', 'unknown') + '_{}'.format(run.upper())
    wboard = commons.WrapTensorboard(log_dir=conf['output_dir'])
    neplog = commons.GetNeptuneLogger(train_name, activate=conf['neptune'])
    # 学習開始:
    print('[{}] Start Train: {}'.format(Now(), run.upper()))
    for e in range(epoch):
        train_loss = train_step_regcls_wls(conf, model, train_loader, loss_fns, optim, scheduler if require_call_everystep else None)
        if not require_call_everystep:
            scheduler.step()
        valid_loss, qwk, obs, exp, acc, y_pred, y_true = valid_step_regcls_wls(conf, run, model, valid_loader, loss_fns)
        # valid_dfからkarolinska/radboudでもそれぞれ計算する
        y_true_k = [ ]
        y_pred_k = [ ]
        y_true_r = [ ]
        y_pred_r = [ ]
        for i in range(len(valid_df)):
            if valid_df.iat[i, 1] == 'karolinska':
                y_true_k.append(y_true[i])
                y_pred_k.append(y_pred[i])
            else:
                y_true_r.append(y_true[i])
                y_pred_r.append(y_pred[i])
        qwk_k = commons.qwk(y_pred_k, y_true_k)
        qwk_r = commons.qwk(y_pred_r, y_true_r)
        # Log:
        updated = (score_t < qwk) or (score_k < qwk_k)
        log_str = '[{}] Epoch: {:4}/{:4}, lr: {:.3e}, t-loss: {:.6e}, v-loss: {:.6e}, acc: {:.5f}, qwk_k: {:.5f}, qwk_r: {:.5f}, qwk: {:.5f}, obs: {:.5f}, exp: {:.5f}'.format(Now(), e, epoch, optims.GetOptimierLR(optim), train_loss, valid_loss, acc, qwk_k, qwk_r, qwk, obs, exp)
        log_str = log_str + (' (*)' if updated else '')
        print(log_str)
        # Tensorboard:
        wboard.writeScalar('lr', e, optims.GetOptimierLR(optim))
        wboard.writeScalar('qwk', e, qwk)
        wboard.writeScalar('qwk_karolinska', e, qwk_k)
        wboard.writeScalar('qwk_radboud', e, qwk_r)
        wboard.writeScalar('obs', e, obs)
        wboard.writeScalar('exp', e, exp)
        wboard.writeScalar('acc', e, acc)
        wboard.writeScalar('loss', e, { 'train' : train_loss, 'valid' : valid_loss })
        # Neptune Logger:
        neplog.write({ 'val_qwk' : qwk, 'val_qwk_karolinska' : qwk_k, 'val_qwk_radboud' : qwk_r, 'val_obs': obs, 'val_exp': exp, 'avg_val_loss' : valid_loss, 'avg_train_loss' : train_loss })
        # Parameter:
        if updated:
            if score_t < qwk:
                score_t = qwk
            if score_k < qwk_k:
                score_k = qwk_k
            torch.save(model.module.state_dict(), os.path.join(conf['output_dir'], 'epoch{}.pth'.format(e)))
            OutputCFM(conf, 'epoch{}'.format(e), y_pred, y_true)
        # UserCommand:
        if os.path.exists('save') and not os.path.exists(os.path.join(conf['output_dir'], 'epoch{}.pth'.format(e))):
            print('[UserCommand] save parameter (epoch={}).'.format(e))
            torch.save(model.module.state_dict(), os.path.join(conf['output_dir'], 'epoch{}.pth'.format(e)))
            os.remove('save')
        if os.path.exists('exit') or os.path.exists('exit_{}'.format(e)):
            print('[UserCommand] exit train loop.')
            break
    torch.save(model.module.state_dict(), os.path.join(conf['output_dir'], 'final.pth'))
    wboard.close()
    neplog.close()

# 検証: regression/classification
def valid_regcls_wls(conf):
    sts = conf['exec'][6:].split('+')
    run = sts[0]
    if conf['model_uri'] is None:
        print('モデルパラメータが指定されていません. --model_uriで有効なパラメータを指定してください.')
    # 入力データ:
    valid_df = pd.read_csv(conf['valid_csv'])
    valid_aug = datasets.GetValidAugment(run, conf['valid_augs'])
    valid_loader = datasets.GetValidDataLoader(conf, run, valid_df, augment_op=valid_aug)
    # モデル:
    model = models.GetModel(conf, num_classes=1, uri=conf['model_uri'], las_process=False).to(conf['device'])
    model = SetupAMP(conf, model, None)
    model = torch.nn.DataParallel(model)
    # 損失関数タイプ:
    # validationについては比較するためにREGはMSEで固定, CLSは選択式
    print('wlsではlossが自動で設定されます.')
    loss_fns = [
        torch.nn.MSELoss(reduction='none'),
        torch.nn.SmoothL1Loss(reduction='none')
    ]
    print('[{}] Start Valid: {}'.format(Now(), run.upper()))
    valid_loss, qwk, obs, exp, acc, y_pred, y_true = valid_step_regcls_wls(conf, run, model, valid_loader, loss_fns)
    # valid_dfからkarolinska/radboudでもそれぞれ計算する
    y_true_k = [ ]
    y_pred_k = [ ]
    y_true_r = [ ]
    y_pred_r = [ ]
    for i in range(len(valid_df)):
        if valid_df.iat[i, 1] == 'karolinska':
            y_true_k.append(y_true[i])
            y_pred_k.append(y_pred[i])
        else:
            y_true_r.append(y_true[i])
            y_pred_r.append(y_pred[i])
    qwk_k = commons.qwk(y_pred_k, y_true_k)
    qwk_r = commons.qwk(y_pred_r, y_true_r)
    # スコア表示
    log_str = '[{}] valid-loss: {:.6e}, acc: {:.5f}, qwk_k: {:.5f}, qwk_r: {:.5f}, qwk: {:.5f}, obs: {:.5f}, exp: {:.5f}'.format(Now(), valid_loss, acc, qwk_k, qwk_r, qwk, obs, exp)
    print(log_str)
    # 混同行列出力:
    OutputCFM(conf, 'valid', y_pred, y_true)

# 学習ステップ (エポック処理)
def train_step_regcls_wls(conf, model, loader, loss_fns, optim, scheduler):
    value = 0
    count = 0
    device = conf['device']
    model.train()
    wtqdm = commons.WrapTqdm(total=len(loader), run=conf['run'])
    for _, itr in enumerate(loader):
        x0, y0, w0 = itr
        x0 = x0.to(device)
        y0 = y0.to(device)
        w0 = w0.to(device)
        yk, yr = model(x0)
        loss = loss_fns[0](yk, y0) * w0 + loss_fns[1](yr, y0) * (1 - w0)
        loss = torch.mean(loss)
        optim.zero_grad()
        if AVAILABLE_AMP and conf['amp']:
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # Gradient Clip:
        if conf['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf['grad_clip'])
        optim.step()
        if scheduler is not None:
            scheduler.step()
        value += conf['batch'] * loss.item()
        count += conf['batch']
        wtqdm.set_description('  train loss = {:e}'.format(value / count))
        wtqdm.update()
    wtqdm.close()
    value /= count
    return value

# 検証ステップ (エポック処理)
def valid_step_regcls_wls(conf, run, model, loader, loss_fns):
    acc_score = 0
    value = 0
    steps = 0
    count = len(loader)
    y_true = np.zeros([count], dtype=np.int)
    y_pred = np.zeros([count], dtype=np.int)
    device = conf['device']
    model.eval()
    with torch.no_grad():
        wtqdm = commons.WrapTqdm(total=count, run=conf['run'])
        for i, itr in enumerate(loader):
            x0, y0, w0 = itr
            x0 = x0.to(device)
            y0 = y0.to(device)
            w0 = w0.to(device)
            yk, yr = model(x0)
            loss = loss_fns[0](yk, y0) * w0 + loss_fns[1](yr, y0) * (1 - w0)
            loss = torch.mean(loss)
            value += loss.item()
            steps += 1
            wtqdm.set_description('  valid loss = {:e}'.format(value / steps))
            wtqdm.update()
            y0 = y0.detach().cpu().numpy()
            yk = yk.detach().cpu().numpy()
            yr = yr.detach().cpu().numpy()
            w0 = w0.detach().cpu().numpy()
            # karolinska/radboud select
            if w0 == 1.0:
                y_ = yk
            else:
                y_ = yr
            if run == 'reg':
                y_ = np.clip(y_ + 0.5, 0, 5)
            else:
                y_ = np.argmax(y_)
            y_pred[i] = int(y_)
            y_true[i] = int(y0)
            acc_score = acc_score + (1 if int(y_) == int(y0) else 0)
        wtqdm.close()
    qwk_score, obs_score, exp_score = commons.qwk_ext(y_pred, y_true, weights='quadratic')
    acc_score = acc_score / steps
    value /= steps
    return value, qwk_score, obs_score, exp_score, acc_score, y_pred, y_true

""" エントリポイント """
if __name__ == "__main__":
    flags.mark_flags_as_required([ ])
    app.run(main)
