#!/bin/bash

FOLD=0
PYTHON=python3.7

mkdir -p outputs/reg_train_k${FOLD}

${PYTHON} main.py \
--exec=train+cls \
--run=local \
--model=reg_p3_h2_resnest50 \
--output_dir=outputs/reg_train_k${FOLD} \
--image_dir=data/process/fmt1_256x256/ \
--train_csv=csvs/cls_kfold_${FOLD}/nfold_train.csv \
--valid_csv=csvs/cls_kfold_${FOLD}/nfold_valid.csv \
--cls_loader=cmb0 \
--loss=ce \
--epoch=30 \
--batch=16 \
--optimizer=adamw \
--lr=1e-4 \
--weight_decay=1e-5 \
--scheduler=cosinedecay \
--lr_min=1e-6
