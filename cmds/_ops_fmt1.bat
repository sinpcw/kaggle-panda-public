mkdir data\process\256x256_fmt1

python ops_fmt1.py ^
--run=local ^
--image_dir=data\prostate-cancer-grade-assessment ^
--input_csv=data\prostate-cancer-grade-assessment\train.csv ^
--output_dir=data\process\256x256_fmt1 ^
--patch=128 ^
--tile=256 ^
--tile_count=16

