mkdir outputs
mkdir outputs\reg_train

python main.py ^
--exec=train+reg ^
--image_dir=data\process\fmt1_256x256 ^
--output_dir=outputs\reg_train ^
--train_csv=csvs\cls_kfold_0\nfold_train.csv ^
--valid_csv=csvs\cls_kfold_0\nfold_valid.csv ^
--reg_loader=fmt1 ^
--model=reg_p3_h2_resnest50 ^
--batch=16 ^
--num_images=12
--optimizer=adamw ^
--lr=1e-4 ^
--weight_decay=1e-5 ^
--scheduler=cosinedecay ^
--lr_min=1e-6 ^
