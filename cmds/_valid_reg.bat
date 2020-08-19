mkdir outputs
mkdir outputs\reg_valid

python main.py ^
--exec=valid+reg ^
--run=local ^
--model=gem+efficientnet-b3 ^
--model_uri=pth\reg-22018008-36.pth ^
--output_dir=outputs/reg_valid ^
--image_dir=data\prostate-cancer-grade-assessment ^
--valid_csv=csvs\cls3_kfold_0\nfold_valid_sample.csv ^
--reg_loader=ass-512 ^
--num_images=16 ^
--gen_images=16 ^
--loss=mse
