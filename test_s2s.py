import os
import torch

import parser
import models
import data_c

import numpy as np
import csv
import torch.nn as nn
from torchvision import models as md
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import rnn

from sklearn.metrics import accuracy_score

def pad_and_sort_batch(DataLoaderBatch):
    """
    DataLoaderBatch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    for idx, data in enumerate(DataLoaderBatch):
       seqs, targs, lengths = data

    video_feat = []
    video_lbl = []
    video_leng = []
    #sample_size = len(seqs)
    sample_size = lengths

    for j in range(0, sample_size, 128):
        video_feat.append(seqs[j:j + 128])
        video_lbl.append(targs[j:j + 128])
        if (j+128) < sample_size:
            video_leng.append(128)
        else:
            video_leng.append(sample_size%128)

    return rnn.pad_sequence(video_feat, batch_first=True), rnn.pad_sequence(video_lbl, batch_first=True), torch.tensor(video_leng)


def collate_fn(DataLoaderBatch):
    imgs = []
    labels = []
    lengths = []

    ''' Reformat Data when loading and add information about length of video '''

    #for i in range(len(DataLoaderBatch)):
    for i in range(1):
        imgs.append(DataLoaderBatch[i][0])
        labels.append(DataLoaderBatch[i][1])
        lengths.append(DataLoaderBatch[i][0].size(0))

    imgs = torch.cat(imgs)
    labels = torch.cat(labels)
    lengths = torch.tensor(lengths)

    return imgs, labels, lengths

def evaluate(model, data_loader):

    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (imgs, gt, len) in enumerate(data_loader):

            for idx, data in enumerate(imgs):
                if len[idx] < 128:
                    data = data[0:len[idx]]
                    gts.append(gt[idx][0:len[idx]])
                else:
                    gts.append(gt[idx])

                data = data.unsqueeze(0)

                if torch.cuda.is_available():
                    data = data.cuda()

                out = model(data)
                pred = torch.max(out, 1)[1]

                pred = pred.cpu().numpy().squeeze()

                preds.append(pred)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    return accuracy_score(gts, preds)


if __name__ == '__main__':
    
    args = parser.arg_parse()

    ''' setup GPU '''
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    ''' load dataset and prepare data loader '''
    test_data = data_c.FullVid(args, mode="test")

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              num_workers=args.workers,
                                              shuffle=False,
                                              collate_fn=collate_fn)

    print('===> prepare pretrained model ...')
    res = md.resnet50(pretrained=True)
    res = nn.Sequential(*list(res.children())[:-1])

    for param in res.parameters():
        param.requires_grad = False

    res.eval()

    if torch.cuda.is_available():
        res.cuda()

    vid_feat = []
    vid_labels = []
    vid_lengths = []

    ''' Calculate feature maps and perform avg pooling for test data '''
    for idx, (imgs, label, len) in enumerate(test_loader):

        ''' move data to gpu '''
        if torch.cuda.is_available():
            imgs = imgs.cuda()

        video_feat = []
        sample = imgs.size(0)

        for j in range(0, sample, 128):
            frames = imgs[j:j + 128]

            if torch.cuda.is_available():
               frames = frames.cuda()

            video_feat.append(res(frames).contiguous().view(frames.size(0), -1).cpu())

        video_feat = torch.cat(video_feat)

        vid_feat.append(video_feat)
        vid_labels.append(label)
        vid_lengths.append(len[0])
        vid_cat = test_data.vid

    ''' prepare data '''
    test_data = data_c.Features3(vid_feat, vid_labels, vid_lengths)

    test_load = torch.utils.data.DataLoader(test_data,
                                            batch_size=1,
                                            num_workers=args.workers,
                                            shuffle=False,
                                            collate_fn=pad_and_sort_batch)

    print('===> prepare model ...')
    model = models.RNN_seq(input_size=2048, hidden_size=1024)

    checkpoint = torch.load(args.resume3)
    #checkpoint = torch.load('RNN_model_best.pth.tar',  map_location='cpu')
    model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    with torch.no_grad():
        preds = []
        gts = []

        for idx, (imgs, gt, len) in enumerate(test_load):

            txt_file = os.path.join(args.save_dir, vid_cat[idx] + '.txt')
            with open(txt_file, "w+", newline="") as txtfile:
                writer = csv.writer(txtfile)

                for idx, data in enumerate(imgs):
                    if len[idx] < 128:
                        data = data[0:len[idx]]
                        #gts.append(gt[idx][0:len[idx]])
                    #else:
                        #gts.append(gt[idx])

                    data = data.unsqueeze(0)

                    if torch.cuda.is_available():
                        data = data.cuda()

                    out = model(data)
                    pred = torch.max(out, 1)[1]

                    for j in range(pred.size(0)):
                        writer.writerow([pred[j].item()])

    #                 pred = pred.cpu().numpy().squeeze()
    #
    #                 preds.append(pred)
    #
    #     gts = np.concatenate(gts)
    #     preds = np.concatenate(preds)
    #
    # acc = accuracy_score(gts, preds)
    # #
    # # print('Testing Accuracy: {}'.format(acc))
