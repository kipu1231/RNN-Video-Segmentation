wget https://www.dropbox.com/s/sa8a8xvzba5fmcr/S2S_model_best.pth.tar?dl=1
RESUME3='S2S_model_best.pth.tar?dl=1'
python3 test_p3.py --resume3 $RESUME3 --data_dir $1 --save_dir $2

