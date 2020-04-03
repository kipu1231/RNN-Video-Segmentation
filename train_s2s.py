import os
import torch

import parser
import models
import data_c
import test_s2s

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models as md
from torch.nn.utils import rnn

from tensorboardX import SummaryWriter


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


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
    sample_size = len(seqs)

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
    print('size of dataloader')
    print(len(DataLoaderBatch))
    ''' Reformat Data when loading and add information about length of video '''
    for i in range(len(DataLoaderBatch)):
        print(i)
        print(DataLoaderBatch[i][0].size())
        imgs.append(DataLoaderBatch[i][0])
        labels.append(DataLoaderBatch[i][1])
        lengths.append(DataLoaderBatch[i][0].size(0))

    imgs = torch.cat(imgs)
    labels = torch.cat(labels)
    lengths = torch.tensor(lengths)

    return imgs, labels, lengths


if __name__ == '__main__':

    args = parser.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    # ''' load dataset and prepare data loader '''
    # train_data = data_c.FullVid(args, mode="train")
    # val_data = data_c.FullVid(args, mode="val")
    #
    # print('===> prepare dataloader ...')
    # train_loader = torch.utils.data.DataLoader(train_data,
    #                                            batch_size=1,
    #                                            num_workers=args.workers,
    #                                            shuffle=False,
    #                                            collate_fn=collate_fn)
    # val_loader = torch.utils.data.DataLoader(val_data,
    #                                          batch_size=1,
    #                                          num_workers=args.workers,
    #                                          shuffle=False,
    #                                          collate_fn=collate_fn)
    #
    # ''' load model '''
    # print('===> prepare pretrained model ...')
    # res = md.resnet50(pretrained=True)
    # res = nn.Sequential(*list(res.children())[:-1])
    #
    # for param in res.parameters():
    #     param.requires_grad = False
    #
    # res.eval()
    #
    # if torch.cuda.is_available():
    #     res.cuda()
    #
    # print('===> calculate for training data ...')
    #
    # vid_feat = []
    # vid_labels = []
    # vid_lengths = []
    #
    # ''' Calculate feature maps and perform avg pooling for training data '''
    # for idx, (imgs, label, length) in enumerate(val_loader):
    #     print(idx)
    #
    #     ''' move data to gpu '''
    #     if torch.cuda.is_available():
    #         imgs = imgs.cuda()
    #
    #     video_feat = []
    #     sample = imgs.size(0)
    #
    #     for j in range(0, sample, 128):
    #         frames = imgs[j:j + 128]
    #
    #         if torch.cuda.is_available():
    #             frames = frames.cuda()
    #
    #         #print(frames.size())
    #         #print(res(frames).contiguous().view(frames.size(0), -1).cpu().size())
    #         # Shape: 256 x 2048
    #         video_feat.append(res(frames).contiguous().view(frames.size(0), -1).cpu())
    #
    #     video_feat = torch.cat(video_feat)
    #     #print(video_feat.size())
    #
    #     vid_feat.append(video_feat)
    #     vid_labels.append(label)
    #     vid_lengths.append(length[0])
    #
    # torch.save(vid_feat, "features_val.pkl")
    # torch.save(vid_labels, "labels_val.pkl")
    # torch.save(vid_lengths, "length_val.pkl")

    vid_feat = torch.load("./preprocess_p3/features_train_p3.pkl")
    vid_labels = torch.load("./preprocess_p3/labels_train.pkl")
    vid_lengths = torch.load("./preprocess_p3/length_train.pkl")
    vid_feat_val = torch.load("./preprocess_p3/features_val_p3.pkl")
    vid_labels_val = torch.load("./preprocess_p3/labels_val.pkl")
    vid_lengths_val = torch.load("./preprocess_p3/length_val.pkl")

    training_data = data_c.Features3(vid_feat, vid_labels, vid_lengths)
    #validation_data = data_c.Features3(vid_feat, vid_labels, vid_lengths)
    validation_data = data_c.Features3(vid_feat_val, vid_labels_val, vid_lengths_val)

    train_load = torch.utils.data.DataLoader(training_data,
                                             batch_size=1,
                                             num_workers=args.workers,
                                             shuffle=True,
                                             collate_fn=pad_and_sort_batch)

    val_load = torch.utils.data.DataLoader(validation_data,
                                           batch_size=1,
                                           num_workers=args.workers,
                                           shuffle=False,
                                           collate_fn=pad_and_sort_batch)

    ''' load model '''
    print('===> prepare model ...')

    model = models.RNN_seq(input_size=2048, hidden_size=1024)
    #checkpoint = torch.load(args.resume2)
    checkpoint = torch.load('RNN_model_best.pth.tar',  map_location='cpu')
    #checkpoint = torch.load('RNN_model_best.pth_46.tar?dl=1')
    model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        model.cuda()

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_acc = 0

    for epoch in range(1, args.epoch + 1):
        train_info = 'Epoch: [{0}]'.format(epoch)
        print(train_info)

        model.train()
        train_loss = 0.0
        train_acc = 0.0
        iters = 0

        for idx, data in enumerate(train_load):
            train_info = 'Video Number: [{0}]'.format(idx+1)
            #print(train_info)
            iters += 1
            ''' prepare data '''
            imgs, label, length = data

            #print(label.size())
            #print(imgs.size())

            for idx, data in enumerate(imgs):
                train_info = 'Video: [{0}][{1}/{2}]'.format(iters, idx + 1, len(imgs))

                data = data.unsqueeze(0)

                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()

                optimizer.zero_grad()
                out = model(data)

                loss = criterion(out, label[idx])
                loss.backward()
                optimizer.step()

                '''' write out information to tensorboard '''
                writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
                train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

                print(train_info)

        if epoch % args.val_epoch == 0:
            ''' evaluate the model '''
            acc = test_s2s.evaluate(model, val_load)
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(model, os.path.join(args.save_dir, 'model_best.pth.tar'))
                best_acc = acc

        ''' save model '''
        save_model(model, os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))
