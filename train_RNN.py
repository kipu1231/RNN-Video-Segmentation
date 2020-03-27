import os
import torch

import parser
import models
import data_c
import test_RNN

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models as md
from torch.nn.utils import rnn

from tensorboardX import SummaryWriter


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def sort_batch(batch, targets, lengths):
    """
    Sort a minibatch by the length of the sequences with the longest sequences first
    return the sorted batch targes and sequence lengths.
    This way the output can be used by pack_padded_sequences(...)
    """
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    target_tensor = targets[perm_idx]
    return seq_tensor, target_tensor, seq_lengths


def pad_and_sort_batch(DataLoaderBatch):
    """
    DataLoaderBatch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    # batch_size = len(DataLoaderBatch)
    batch_split = list(zip(*DataLoaderBatch))

    seqs, targs, lengths = batch_split[0], batch_split[1], batch_split[2]

    return rnn.pad_sequence(seqs, batch_first=True), torch.tensor(targs), torch.tensor(lengths)


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

    ''' load dataset and prepare data loader '''
    train_data = data_c.TrimmedVid2(args, mode="train")
    val_data = data_c.TrimmedVid2(args, mode="val")

    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=1,
                                               num_workers=args.workers,
                                               shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=1,
                                             num_workers=args.workers,
                                             shuffle=False)

    ''' load model '''
    print('===> prepare pretrained model ...')
    res = md.resnet50(pretrained=True)
    res = nn.Sequential(*list(res.children())[:-1])

    for param in res.parameters():
        param.requires_grad = False

    res.eval()

    if torch.cuda.is_available():
        res.cuda()

    print('===> calculate for training data ...')
    ''' Calculate feature maps and perform avg pooling for training data '''
    for idx, (imgs, label, length) in enumerate(train_loader):
        print(idx)

        ''' move data to gpu '''
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        imgs = imgs.squeeze(0)

        # x = vgg16_ft(imgs).contiguous().view(imgs.size(0), -1)
        # out = vgg16_cls(x)

        out = res(imgs).contiguous().view(imgs.size(0), -1)

        if idx == 0:
            features = out
            label_list = label
            length_list = length
        else:
            features = torch.cat((features, out), dim=0)
            label_list = torch.cat((label_list, label))
            length_list = torch.cat((length_list, length))

    # os.makedirs("./preprocess_vgg/p1/val", exist_ok=True)
    # torch.save(features.cpu(), "features.pkl")
    # torch.save(label_list.cpu(), "labels.pkl")
    # torch.save(length_list.cpu(), "length.pkl")

    print('===> calculate for validation data ...')
    ''' Calculate feature maps and perform avg pooling for validation data '''
    for idx, (imgs, label, length) in enumerate(val_loader):
        print(idx)

        ''' move data to gpu '''
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        imgs = imgs.squeeze(0)

        #x = vgg16_ft(imgs).contiguous().view(imgs.size(0), -1)
        #out = vgg16_cls(x)

        out = res(imgs).contiguous().view(imgs.size(0), -1)

        if idx == 0:
            features_val = out
            label_list_val = label
            length_list_val = length
        else:
            features_val = torch.cat((features_val, out), dim=0)
            label_list_val = torch.cat((label_list_val, label))
            length_list_val = torch.cat((length_list_val, length))

    # os.makedirs("./preprocess_vgg/p1/val", exist_ok=True)
    # torch.save(features_val.cpu(), "features_val_res.pkl")
    # torch.save(label_list_val.cpu(), "labels_val_res.pkl")
    # torch.save(length_list_val.cpu(), "length_val_res.pkl")

    print('===> prepare dataloader ...')
    # features = torch.load("./preprocess_p2/features.pkl")
    # label_list = torch.load("./preprocess_p2/labels.pkl")
    # length_list = torch.load("./preprocess_p2/length.pkl")
    # features_val = torch.load("./preprocess_p2/features_val.pkl")
    # label_list_val = torch.load("./preprocess_p2/labels_val.pkl")
    # length_list_val = torch.load("./preprocess_p2/length_val.pkl")

    # features = torch.load("./preprocess_p2_res/features_res.pkl")
    # label_list = torch.load("./preprocess_p2_res/labels_res.pkl")
    # length_list = torch.load("./preprocess_p2_res/length_res.pkl")
    # features_val = torch.load("./preprocess_p2_res/features_val_res.pkl")
    # label_list_val = torch.load("./preprocess_p2_res/labels_val_res.pkl")
    # length_list_val = torch.load("./preprocess_p2_res/length_val_res.pkl")

    #print(len(features))

    training_data = data_c.Features2(features, label_list, length_list)
    validation_data = data_c.Features2(features_val,label_list_val,length_list_val)

    train_load = torch.utils.data.DataLoader(training_data,
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             shuffle=True,
                                             collate_fn=pad_and_sort_batch)

    val_load = torch.utils.data.DataLoader(validation_data,
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             collate_fn=pad_and_sort_batch)

    ''' load model '''
    print('===> prepare pretrained model ...')
    #model = models.RNN(input_size=4096, hidden_size=1024)
    #model_fc = models.RNN_FC(args, hidden_dim=1)
    model = models.RNN(input_size=2048, hidden_size=1024)

    if torch.cuda.is_available():
        model.cuda()
        #model_fc.cuda()

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer_fc = torch.optim.Adam(model_fc.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_acc = 0

    for epoch in range(1, args.epoch + 1):
        model.train()
        #model_fc.train()
        train_loss = 0.0
        train_acc = 0.0

        for idx, data in enumerate(train_load):
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(train_load))
            iters += 1

            ''' prepare data '''
            imgs, label, length = data
            idx = torch.argsort(-length)
            length = length[idx]

            img_packed = rnn.pack_padded_sequence(imgs, lengths=length, batch_first=True)

            if torch.cuda.is_available():
                img_packed = img_packed.cuda()
                label = label.cuda()

            ''' train model '''
            optimizer.zero_grad()

            out = model(img_packed, length)

            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            '''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)

        if epoch % args.val_epoch == 0:
            ''' evaluate the model '''
            acc = test_RNN.evaluate(model, val_load)
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(model, os.path.join(args.save_dir, 'model_best.pth.tar'))
                best_acc = acc

        ''' save model '''
        save_model(model, os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))
