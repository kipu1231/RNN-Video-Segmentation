import os
import torch
import math
import parser
import models
import data_c

import numpy as np
import csv
import torch.nn as nn
from torchvision import models as md
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    args = parser.arg_parse()

    path_pred = "./output_p3"
    path_gt = "hw4_data/FullLengthVideos/labels/valid"
    video = "OP01-R07-Pizza"
    path_imgs = "hw4_data/FullLengthVideos/videos/valid"

    ''' read images and save '''
    img_list = []
    for filename in os.listdir(os.path.join(path_imgs, video)):
        img_list.append(filename)
    img_list.sort()
    print(len(img_list))

    ''' open files and save label in array '''
    pred = []
    with open(os.path.join(path_pred, video + '.txt'), 'r') as preds:
        for line in preds:
            pred.append(int(line.split('\n')[0]))

    gt = []
    with open(os.path.join(path_gt, video + '.txt'), 'r') as gts:
        for line in gts:
            gt.append(int(line.split('\n')[0]))

    ''' prepare plot '''
    colors = {0: 'darkslategrey', 1: 'maroon', 2: 'grey', 3: 'orange', 4: 'lightcoral', 5: 'lightskyblue', 6: 'plum', 7: 'yellow',
                  8: 'sienna', 9: 'k', 10: 'green'}

    # Creates just a figure and only one subplot
    fig, ax = plt.subplots(4, 2, figsize=(18, 5), gridspec_kw={'height_ratios': [1, 2, 5, 2], 'width_ratios': [1, 12]})
    fig.tight_layout(h_pad=-2, w_pad=-2)

    ''' define acxes '''
    ax[1, 0].text(0.04, 0.4, "GroundTruth", fontsize=14)
    ax[2, 0].text(0.17, 0.47, "Frames", fontsize=14)
    ax[3, 0].text(0.1, 0.4, "Prediction", fontsize=14)

    for i in range(4):
        ax[i, 0].axis('off')
        ax[i, 1].axis('off')

    ''' add bar with labels and ground truth to plot '''
    start = 0
    end = len(img_list)
    for i in range(start, end):
        l = i - 1
        if i == 0:
            l = 0
        ax[1, 1].barh(0, 5, left=l, color=colors[gt[i]])
        ax[3, 1].barh(0, 5, left=l, color=colors[pred[i]])

    ''' add images to plot'''
    step_num = math.floor((end - start - 1) / 11)
    img_dir = os.path.join(path_imgs, video)
    for i in range(11):
        x = 0.105 + i * 0.0763
        ax_img = fig.add_axes([x, 0.264, 0.0763, 0.4])
        img = plt.imread(os.path.join(img_dir, img_list[start + i * step_num]))
        ax_img.imshow(img)
        ax_img.axis('off')

    ''' add description'''
    ax[0, 1].text(0., 0.3, "Other", fontsize=12)
    ax[0, 1].text(0.13, 0.3, "Cut", fontsize=12)
    ax[0, 1].text(0.35, 0.3, "Inspect/Read", fontsize=12)
    ax[0, 1].text(0.45, 0.3, "Move around", fontsize=12)
    ax[0, 1].text(0.665, 0.3, "Put", fontsize=12)
    ax[0, 1].text(0.775, 0.3, "Transfer", fontsize=12)
    ax[0, 1].text(0.87, 0.3, "Take and Open", fontsize=12)

    ''' plot and save'''
    plt.savefig("fig3_3.jpg")
    plt.close()


