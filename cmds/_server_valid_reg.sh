#!/bin/bash

PYTHON=python3.7
mkdir -p outputs/reg_valid

${PYTHON} main.py \
--exec=valid+reg \
--run=local \
--model=reg_p3_h1_resnest50 \
--model_uri=pth/reg-22017537-37.pth \
--output_dir=outputs/reg_valid \
--image_dir=data/prostate-cancer-grade-assessment \
--valid_csv=data/prostate-cancer-grade-assessment/train_sample.csv \
--reg_loader=auto-384 \
--num_images=16 \
--gen_images=16

# train_sample.csv : train.csv ‚Ì csv.loc[0:50, :]
# --model_uri=pth/cls-22017066-68.pth \
#--valid_csv=csvs/cls_kfold_0/nfold_valid.csv \
#--valid_csv=data/prostate-cancer-grade-assessment/train_sample.csv \
#--valid_csv=data/prostate-cancer-grade-assessment/train.csv \
