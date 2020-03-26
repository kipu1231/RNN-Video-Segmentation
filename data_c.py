import os
import json
import torch
import scipy.misc
import reader
import numpy as np
from skimage import io

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class FullVid(Dataset):
    def __init__(self, args, mode='train'):
        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.img_dir = os.path.join(self.data_dir, 'FullLengthVideos')

        ''' read videos '''
        if mode == 'train':
            self.vid_path = os.path.join(self.img_dir, "videos/" + "train")
        if mode == 'val':
            self.vid_path = os.path.join(self.img_dir, "videos/" + "valid")
        if mode == 'test':
            self.vid_path = os.path.join(args.data_dir)

        self.vid = sorted(os.listdir(self.vid_path))

        ''' read labels '''
        if mode == 'train':
            self.txt_path = os.path.join(self.img_dir, "labels/" + "train")
        if mode == 'val':
            self.txt_path = os.path.join(self.img_dir, "labels/" + "valid")
        if mode == 'test':
            self.txt_path = os.path.join(args.data_dir)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
        ])

    def __len__(self):
        self.len = len(self.vid)
        return self.len

    def __getitem__(self, idx):
        path = os.path.join(self.vid_path, self.vid[idx])
        f_name = sorted(os.listdir(path))

        imgs = []
        label_test = []
        for i in range(len(f_name)):
            img_path = os.path.join(path, f_name[i])
            imgs.append(self.transform(Image.fromarray(np.uint8(io.imread(img_path)))).unsqueeze(0))

        label_test = torch.ones(len(imgs))
        imgs = torch.cat(imgs)

        if self.mode == 'test':
            return imgs, label_test

        elif self.mode != 'test':
            label_path = os.path.join(self.txt_path, self.vid[idx] + ".txt")
            labels = []

            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    labels.append(int(line))

            labels = torch.tensor(labels)
            print(labels.size())

            return imgs, labels


class Features3(Dataset):
    def __init__(self, features, cls, length):
        self.len = int(len(length))
        self.features = features
        self.labels = cls
        self.length = length

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.length[index]


class TrimmedVid(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.img_dir = os.path.join(self.data_dir, 'TrimmedVideos')

        ''' read the data list '''
        if mode == 'train':
            csv_path = os.path.join(self.img_dir, "label/" + "gt_train.csv")
        if mode == 'val':
            csv_path = os.path.join(self.img_dir, "label/" + "gt_valid.csv")
        if mode == 'test':
            csv_path = os.path.join(args.csv_dir)

        self.labels = reader.getVideoList(csv_path)
        self.len = len(self.labels["Video_index"])

        ''' read videos '''
        if mode == 'train':
            self.vid_path = os.path.join(self.img_dir, "video/" + "train")
        if mode == 'val':
            self.vid_path = os.path.join(self.img_dir, "video/" + "valid")
        if mode == 'test':
            self.vid_path = os.path.join(args.data_dir)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        ''' get frames '''
        frames = reader.readShortVideo(
            video_path=self.vid_path,
            video_category=self.labels["Video_category"][idx],
            video_name=self.labels["Video_name"][idx]
        )

        imgs = []

        for i in range(len(frames)):
            imgs.append(self.transform(Image.fromarray(np.uint8(frames[i]))).unsqueeze(0))
        imgs = torch.cat(imgs)

        label = int(self.labels["Action_labels"][idx])

        return imgs, label


class Features(Dataset):
    def __init__(self, frame_avg, cls):
        self.features = frame_avg
        self.labels = cls
        self.len = len(self.features)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class TrimmedVid2(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.img_dir = os.path.join(self.data_dir, 'TrimmedVideos')

        ''' read the data list '''
        if mode == 'train':
            csv_path = os.path.join(self.img_dir, "label/" + "gt_train.csv")
        if mode == 'val':
            csv_path = os.path.join(self.img_dir, "label/" + "gt_valid.csv")
        if mode == 'test':
            csv_path = os.path.join(args.csv_dir)

        self.labels = reader.getVideoList(csv_path)
        self.len = len(self.labels["Video_index"])

        ''' read videos '''
        if mode == 'train':
            self.vid_path = os.path.join(self.img_dir, "video/" + "train")
        if mode == 'val':
            self.vid_path = os.path.join(self.img_dir, "video/" + "valid")
        if mode == 'test':
            self.vid_path = os.path.join(args.data_dir)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        ''' get frames '''
        frames = reader.readShortVideo(
            video_path=self.vid_path,
            video_category=self.labels["Video_category"][idx],
            video_name=self.labels["Video_name"][idx]
        )

        imgs = []

        for i in range(len(frames)):
            imgs.append(self.transform(Image.fromarray(np.uint8(frames[i]))).unsqueeze(0))
        imgs = torch.cat(imgs)

        label = int(self.labels["Action_labels"][idx])
        length = len(frames)

        return imgs, label, length


class Features2(Dataset):
    def __init__(self, features, cls, length):
        self.features = features
        self.labels = cls
        self.length = length
        self.len = int(list(cls.size())[0])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        features_list = []
        helper = 0
        for idx, len in enumerate(self.length):
            #features_list.append(self.features[helper:len + helper].unsqueeze(0))
            features_list.append(self.features[helper:len + helper])
            helper = helper + len

        return features_list[index], self.labels[index], self.length[index]
