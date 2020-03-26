wget https://www.dropbox.com/s/t8agmvfjl5ssltv/CNN_model_best.pth.tar?dl=1
RESUME1='CNN_model_best.pth.tar?dl=1'
python3 test_CNN.py --resume1 $RESUME1 --data_dir $1 --csv_dir $2 --save_dir $3

