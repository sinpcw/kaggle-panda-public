mkdir outputs
mkdir outputs\reg_valid_esm

python main.py ^
--exec=valid+reg+esm ^
--run=local ^
--model=gem+efficientnet-b3 ^
--model_uri=gem+efficientnet-b3=pth\reg-22018008-36.pth:regnety_008=pth\reg-22017996-30.pth ^
--output_dir=outputs/reg_valid ^
--image_dir=data\prostate-cancer-grade-assessment ^
--valid_csv=csvs\cls3_kfold_0\nfold_valid.csv ^
--reg_loader=ass-512 ^
--num_images=16 ^
--gen_images=16 ^
--loss=mse
