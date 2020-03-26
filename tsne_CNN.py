import torch
from PIL import Image
import parser
import models
import data_c
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    args = parser.arg_parse()

    features = torch.load("preprocess_vgg/features.pkl")
    labels = torch.load("preprocess_vgg/labels.pkl")

    ''' prepare tSNE '''
    tsne = TSNE(n_components=2, init="pca")
    x = np.array([]).reshape(0, 4096)
    y_cls = np.array([], dtype=np.int16).reshape(0, )

    x = np.vstack((x, features.numpy()))
    y_cls = np.concatenate((y_cls, labels.numpy()))

    print(x.shape)
    print(y_cls.shape)

    ''' perform tSNE and get min, max and norm '''
    x_tsne = tsne.fit_transform(x)
    print("Data has the {} before tSNE and the following after tSNE {}".format(x.shape[-1], x_tsne.shape[-1]))
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    X_norm = (x_tsne - x_min) / (x_max - x_min)
    ''' plot results of tSNE '''
    colors = ['lightcoral', 'maroon', 'k', 'grey', 'orange', 'darkslategrey', 'lightskyblue', 'plum', 'yellow', 'sienna', 'green']
    y_cls = y_cls.astype(int)
    class_color = [colors[label] for label in y_cls]

    plt.figure(1, figsize=(8, 8))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=class_color, s=5)
    #plt.legend()
    plt.savefig("./cnn_tsne.png")
    plt.title("tSNE CNN")
    plt.close("all")