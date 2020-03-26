wget https://www.dropbox.com/s/8m025c88etw4tie/CNN_model_best.pth.tar?dl=1
RESUME1='CNN_model_best.pth.tar?dl=1'
python3 test_p1.py --resume1 $RESUME1 --data_dir $1 --csv_dir $2 --save_dir $3

