wget https://www.dropbox.com/s/xxas2q6e5vl17ha/S2S_model_best.pth.tar?dl=1
RESUME3='S2S_model_best.pth.tar?dl=1'
python3 test_s2s.py --resume3 $RESUME3 --data_dir $1 --save_dir $2

