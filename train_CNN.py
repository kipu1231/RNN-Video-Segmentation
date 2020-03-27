import os
import torch

import parser
import models
import data_c
import test_CNN

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models as md

from tensorboardX import SummaryWriter


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


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

    ''' Code for Preprocessing '''
    # ''' load dataset and prepare data loader '''
    # train_data = data_c.TrimmedVid(args, mode="train")
    # val_data = data_c.TrimmedVid(args, mode="val")
    #
    # print('===> prepare dataloader ...')
    # train_loader = torch.utils.data.DataLoader(train_data,
    #                                            batch_size=1,
    #                                            num_workers=args.workers,
    #                                            shuffle=False)
    # val_loader = torch.utils.data.DataLoader(val_data,
    #                                          batch_size=1,
    #                                          num_workers=args.workers,
    #                                          shuffle=False)
    #
    # ''' load model '''
    # print('===> prepare pretrained model ...')
    # vgg16 = md.vgg16_bn(pretrained=True)
    # vgg16_ft = nn.Sequential(*(list(vgg16.features)))
    # vgg16_cls = nn.Sequential(*list(vgg16.classifier.children())[:-1])
    #
    # for param in vgg16_ft.parameters():
    #     param.requires_grad = False
    #
    # for param in vgg16_cls.parameters():
    #     param.requires_grad = False
    #
    # vgg16_ft.eval()
    # vgg16_cls.eval()
    #
    # if torch.cuda.is_available():
    #     vgg16_ft.cuda()
    #     vgg16_cls.cuda()
    #
    # ''' Use for preprocessing of CNN features with pretrained ResNet'''
    # # res = md.resnet50(pretrained=True)
    # # res = nn.Sequential(*list(res.children())[:-1])
    # #
    # # for param in res.parameters():
    # #     param.requires_grad = False
    # #
    # # res.eval()
    # #
    # # if torch.cuda.is_available():
    # #     res.cuda()
    #
    # ''' Calculate feature maps and perform avg pooling for training data '''
    # for idx, (imgs, label) in enumerate(train_loader):
    #
    #     ''' move data to gpu '''
    #     if torch.cuda.is_available():
    #         imgs = imgs.cuda()
    #     imgs = imgs.squeeze(0)
    #
    #     x = vgg16_ft(imgs).contiguous().view(imgs.size(0), -1)
    #     out = vgg16_cls(x)
    #
    #     #out = res(imgs).contiguous().view(imgs.size(0), -1)
    #
    #     if idx == 0:
    #         features = torch.mean(out, dim=0).view(1, 4096) #use for VGG
    #         #features = torch.mean(out, dim=0).view(1, 2048) #use for ResNet
    #         label_list = label
    #     else:
    #         features = torch.cat((features, torch.mean(out, dim=0).view(1, 4096)), dim=0) #use for VGG
    #         #features = torch.cat((features, torch.mean(out, dim=0).view(1, 2048)), dim=0) #use for ResNet
    #         label_list = torch.cat((label_list, label))
    #
    # os.makedirs("./preprocess_resnet", exist_ok=True)
    # os.makedirs("./preprocess_vgg", exist_ok=True)
    # torch.save(features.cpu(), "features.pkl")
    # torch.save(label_list.cpu(), "labels.pkl")
    #
    # ''' Calculate feature maps and perform avg pooling for validation data '''
    # for idx, (imgs, label) in enumerate(val_loader):
    #
    #     ''' move data to gpu '''
    #     if torch.cuda.is_available():
    #         imgs = imgs.cuda()
    #     imgs = imgs.squeeze(0)
    #
    #     x = vgg16_ft(imgs).contiguous().view(imgs.size(0), -1)
    #     out = vgg16_cls(x)
    #
    #     #out = res(imgs).contiguous().view(imgs.size(0), -1)
    #
    #     if idx == 0:
    #         features_val = torch.mean(out, dim=0).view(1, 4096) #use for VGG
    #         #features_val = torch.mean(out, dim=0).view(1, 2048) #use for ResNet
    #         label_list_val = label
    #     else:
    #         features_val = torch.cat((features_val, torch.mean(out, dim=0).view(1, 4096)), dim=0) #use for VGG
    #         #features_val = torch.cat((features_val, torch.mean(out, dim=0).view(1, 2048)), dim=0) #use for ResNet
    #         label_list_val = torch.cat((label_list_val, label))
    #
    # os.makedirs("./preprocess_resnet", exist_ok=True)
    # os.makedirs("./preprocess_vgg", exist_ok=True)
    # torch.save(features_val.cpu(), "features_val.pkl")
    # torch.save(label_list_val.cpu(), "labels_val.pkl")


    ''' Load for using preprocessing with pretrained VGG '''
    print('===> load preprocessed data with VGG ...')
    features = torch.load("./preprocess/CNN/preprocess_vgg/features.pkl")
    label_list = torch.load("./preprocess/CNN/preprocess_vgg/labels.pkl")
    features_val = torch.load("./preprocess/CNN/preprocess_vgg/features.pkl")
    label_list_val = torch.load("./preprocess/CNN/preprocess_vgg/labels.pkl")

    ''' Print shapes after avg pooling '''
    print('Shapes of training data')
    print(features.size())
    print(label_list.size())
    print('Shapes of val data')
    print(features_val.size())
    print(label_list_val.size())


    print('===> prepare dataloader ...')
    training_data = data_c.Features(features, label_list)
    validation_data = data_c.Features(features_val, label_list_val)

    train_load = torch.utils.data.DataLoader(training_data,
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             shuffle=True)

    val_load = torch.utils.data.DataLoader(validation_data,
                                           batch_size=args.train_batch,
                                           num_workers=args.workers,
                                           shuffle=False)

    print('===> prepare model ...')
    model = models.CNN_classifier(args) 

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
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for idx, (imgs, cls) in enumerate(train_load):
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(train_load))
            iters += 1

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                cls = cls.cuda()

            ''' forward path '''
            #cls = torch.LongTensor(cls)
            output = model(imgs)

            ''' compute loss, backpropagation, update parameters '''
            loss = criterion(output, cls)  # compute loss

            optimizer.zero_grad()  # set grad of all parameters to zero
            loss.backward()  # compute gradient for each parameters
            optimizer.step()  # update parameters

            '''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)

        if epoch % args.val_epoch == 0:
            ''' evaluate the model '''
            acc = test_CNN.evaluate(model, val_load)
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(model, os.path.join(args.save_dir, 'model_best.pth.tar'))
                best_acc = acc

        ''' save model '''
        save_model(model, os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))
