wget https://www.dropbox.com/s/1l6xb6h046sws5o/RNN_model_best.pth_46.tar?dl=1
RESUME2='RNN_model_best.pth_46.tar?dl=1'
python3 test_RNN.py --resume2 $RESUME2 --data_dir $1 --csv_dir $2 --save_dir $3
