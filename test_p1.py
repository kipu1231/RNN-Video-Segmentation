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

from sklearn.metrics import accuracy_score

def evaluate(model, data_loader):

    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):

            if torch.cuda.is_available():
                imgs = imgs.cuda()

            pred = model(imgs)

            _, pred = torch.max(pred, dim=1)

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
    test_data = data_c.TrimmedVid(args, mode="test")

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              num_workers=args.workers,
                                              shuffle=False)

    print('===> prepare pretrained model ...')
    vgg16 = md.vgg16_bn(pretrained=True)
    vgg16_ft = nn.Sequential(*(list(vgg16.features)))
    vgg16_cls = nn.Sequential(*list(vgg16.classifier.children())[:-1])

    for param in vgg16_ft.parameters():
        param.requires_grad = False

    for param in vgg16_cls.parameters():
        param.requires_grad = False

    vgg16_ft.eval()
    vgg16_cls.eval()

    if torch.cuda.is_available():
        vgg16_ft.cuda()
        vgg16_cls.cuda()

    ''' Calculate feature maps and perform avg pooling for test data '''
    for idx, (imgs, label) in enumerate(test_loader):

        ''' move data to gpu '''
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        imgs = imgs.squeeze(0)

        x = vgg16_ft(imgs).contiguous().view(imgs.size(0), -1)
        out = vgg16_cls(x)

        if idx == 0:
            features = torch.mean(out, dim=0).view(1, 4096)
            label_list = label
        else:
            features = torch.cat((features, torch.mean(out, dim=0).view(1, 4096)), dim=0)
            label_list = torch.cat((label_list, label))

    print('Shapes of test data after Avg pooling')
    print(features.size())
    print(label_list.size())

    ''' prepare model '''
    test_data = data_c.Features(features, label_list)

    test_load = torch.utils.data.DataLoader(test_data,
                                            batch_size=args.train_batch,
                                            num_workers=args.workers,
                                            shuffle=False)

    print('===> prepare model ...')
    model = models.CNN_classifier(args)

    checkpoint = torch.load(args.resume1)
    model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    with torch.no_grad():
        preds = []
        gts = []

        txt_file = os.path.join(args.save_dir, 'p1_valid.txt')

        with open(txt_file, "w+", newline="") as txtfile:
            writer = csv.writer(txtfile)

            for idx, (imgs, gt) in enumerate(test_load):

                if torch.cuda.is_available():
                    imgs = imgs.cuda()

                pred = model(imgs)

                _, pred = torch.max(pred, dim=1)

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
