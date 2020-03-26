from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Image Segmentation for HW2')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='hw4_data',
                        help="root path to data directory")
    parser.add_argument('--workers', default=0, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--csv_dir', type=str, default='hw4_data',
                        help="root path to gt csv")

    # training parameters
    parser.add_argument('--gpu', default=0, type=int,
                        help='In homework, please always set to 0')
    parser.add_argument('--epoch', default=500, type=int,
                        help="num of validation iterations")
    parser.add_argument('--val_epoch', default=1, type=int,
                        help="num of validation iterations")
    parser.add_argument('--train_batch', default=64, type=int,
                        help="train batch size")
    parser.add_argument('--test_batch', default=64, type=int,
                        help="test batch size")
    parser.add_argument('--lr', default=0.0004, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.00001, type=float,
                        help="initial learning rate")

    # resume trained model
    parser.add_argument('--resume', type=str, default='',
                        help="path to the trained model")
    parser.add_argument('--resume1', type=str, default='',
                        help="path to the CNN model")
    parser.add_argument('--resume2', type=str, default='',
                        help="path to the RNN model")
    parser.add_argument('--resume3', type=str, default='',
                        help="path to the RNN Seq2seq model")
    # others
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--random_seed', type=int, default=999)

    args = parser.parse_args()

    return args
