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
    # batch_size = len(DataLoaderBatch)
    batch_split = list(zip(*DataLoaderBatch))

    seqs, targs, lengths = batch_split[0], batch_split[1], batch_split[2]

    return rnn.pad_sequence(seqs, batch_first=True), torch.tensor(targs), torch.tensor(lengths)


def evaluate(model, data_loader):

    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (imgs, gt, len) in enumerate(data_loader):

            if torch.cuda.is_available():
                imgs = imgs.cuda()

            idx = torch.argsort(-len)
            length = len[idx]
            img_packed = rnn.pack_padded_sequence(imgs, lengths=length, batch_first=True)

            pred = model(img_packed, length)
            pred = torch.max(pred, 1)[1]

            #_, pred = torch.max(pred, dim=1)

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()

            if pred.size == 1:
                pred = np.expand_dims(pred, axis=1)
                gt = np.expand_dims(gt, axis=1)

            preds.append(pred)
            gts.append(gt)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    return accuracy_score(gts, preds)


if __name__ == '__main__':
    
    args = parser.arg_parse()

    ''' setup GPU '''
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    ''' load dataset and prepare data loader '''
    test_data = data_c.TrimmedVid2(args, mode="test")

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              num_workers=args.workers,
                                              shuffle=False)

    print('===> prepare pretrained model ...')
    res = md.resnet50(pretrained=True)
    res = nn.Sequential(*list(res.children())[:-1])

    for param in res.parameters():
        param.requires_grad = False

    res.eval()

    if torch.cuda.is_available():
        res.cuda()

    ''' Calculate feature maps and perform avg pooling for test data '''
    for idx, (imgs, label, len) in enumerate(test_loader):

        ''' move data to gpu '''
        if torch.cuda.is_available():
            imgs = imgs.cuda()

        imgs = imgs.squeeze(0)

        out = res(imgs).contiguous().view(imgs.size(0), -1)

        if idx == 0:
            features = out
            label_list = label
            length_list = len
        else:
            features = torch.cat((features, out), dim=0)
            label_list = torch.cat((label_list, label))
            length_list = torch.cat((length_list, len))


    ''' prepare data '''
    test_data = data_c.Features2(features, label_list, length_list)

    test_load = torch.utils.data.DataLoader(test_data,
                                            batch_size=args.train_batch,
                                            num_workers=args.workers,
                                            shuffle=False,
                                            collate_fn=pad_and_sort_batch)

    print('===> prepare model ...')
    model = models.RNN(input_size=2048, hidden_size=1024)

    checkpoint = torch.load(args.resume2)
    #checkpoint = torch.load('RNN_model_best.pth.tar',  map_location='cpu')
    model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    with torch.no_grad():
        preds = []
        gts = []

        txt_file = os.path.join(args.save_dir, 'p2_result.txt')

        with open(txt_file, "w+", newline="") as txtfile:
            writer = csv.writer(txtfile)

            for idx, (imgs, gt, len) in enumerate(test_load):

                if torch.cuda.is_available():
                    imgs = imgs.cuda()

                idx = torch.argsort(-len)
                length = len[idx]
                img_packed = rnn.pack_padded_sequence(imgs, lengths=length, batch_first=True)

                pred = model(img_packed, length)

                pred = torch.max(pred, 1)[1]
                #_, pred = torch.max(pred, dim=1)

                for j in range(pred.size(0)):
                    writer.writerow([pred[j].item()])

                pred = pred.cpu().numpy().squeeze()
                gt = gt.numpy().squeeze()

                if pred.size == 1:
                    pred = np.expand_dims(pred, axis=1)
                    gt = np.expand_dims(gt, axis=1)

                preds.append(pred)
                gts.append(gt)

        gts = np.concatenate(gts)
        preds = np.concatenate(preds)

    acc = accuracy_score(gts, preds)

    print('Testing Accuracy: {}'.format(acc))
