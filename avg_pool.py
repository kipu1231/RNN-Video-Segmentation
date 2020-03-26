import os
import torch

import parser
import models
import data_c
import test_p1

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models as md

from tensorboardX import SummaryWriter


def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)


def perform_avg(features, label_list):
    ''' Average Pooling of feature maps'''
    cls_id_vgl = ""
    end = 'false'
    itr = 0
    cls = []
    label = []

    for lbl in label_list:
        a = lbl.split('_')
        cls_action_label = float(a[1])
        label.append(cls_action_label)

    for lbl in label_list:
        a = lbl.split('_')
        cls_id = int(a[0])

        if len(label_list) == itr + 1:
            end = 'true'

        if itr == 0:
            vid_frame = features[itr].view(1, 4096)
            h = "first"

        elif itr != 0 and cls_id == cls_id_vgl:
            vid_frame = torch.cat((vid_frame, features[itr].view(1, 4096)), dim=0)
            if end == 'true':
                vid_mean = torch.mean(vid_frame, dim=0)
                avg_frame = torch.cat((avg_frame, vid_mean.view(1, 4096)), dim=0)
                cls.append(label[itr - 1])

        elif itr != 0 and cls_id != cls_id_vgl:
            if h == 'first':
                vid_mean = torch.mean(vid_frame, dim=0)
                avg_frame = vid_mean.view(1, 4096)
                cls.append(label[itr - 1])
                h = 'second'
            else:
                vid_mean = torch.mean(vid_frame, dim=0)
                avg_frame = torch.cat((avg_frame, vid_mean.view(1, 4096)), dim=0)
                cls.append(label[itr - 1])

            vid_frame = features[itr].view(1, 4096)
            print(vid_frame.size())

        cls_id_vgl = cls_id
        itr = itr + 1

    return avg_frame, cls