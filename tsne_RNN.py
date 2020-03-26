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
from torch.nn.utils import rnn



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
    args.train_batch = 64

    ''' prepare data '''
    features = torch.load("preprocess_p2_res/features_val_res.pkl")
    labels = torch.load("preprocess_p2_res/labels_val_res.pkl")
    length = torch.load("preprocess_p2_res/length_val_res.pkl")

    test_data = data_c.Features2(features, labels, length)

    test_load = torch.utils.data.DataLoader(test_data,
                                            batch_size=args.train_batch,
                                            num_workers=args.workers,
                                            shuffle=False,
                                            collate_fn=pad_and_sort_batch)

    ''' prepare model '''
    model = models.RNN(input_size=2048, hidden_size=1024)

    #checkpoint = torch.load(args.resume2)
    checkpoint = torch.load('RNN_model_best.pth.tar', map_location='cpu')
    model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    embeddings = np.array([]).reshape(0, 1024)
    with torch.no_grad():
        for i, (imgs, label, len) in enumerate(test_load):
            if torch.cuda.is_available():
                imgs = imgs.cuda()

            idx = torch.argsort(-len)
            length = len[idx]
            img_packed = rnn.pack_padded_sequence(imgs, lengths=length, batch_first=True)

            out, _ = model.gru(img_packed)
            out, _ = rnn.pad_packed_sequence(out, batch_first=True)

            outputs = out[torch.arange(out.size(0)), length - 1].cpu().numpy()
            outputs = outputs[idx].reshape(-1, 1024)

            embeddings = np.concatenate((embeddings, outputs))


    ''' prepare tSNE '''
    # tsne = TSNE(n_components=2, n_iter=3000, verbose=1, random_state=seed, n_jobs=20)
    # X = tsne.fit_transform(embeddings)
    # X_min, X_max = X.min(0), X.max(0)
    # X_tsne = (X - X_min) / (X_max - X_min)

    # vis.scatter(X=X_tsne, Y=labels + 1, win="RNN",
    #             opts={
    #                 "legend": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    #                 "markersize": 5,
    #                 "title": "RNN-based features"
    #             })

    print(embeddings.shape)

    tsne = TSNE(n_components=2, init='pca')
    y_cls = np.array([], dtype=np.int16).reshape(0, )
    y_cls = np.concatenate((y_cls, labels.numpy()))

    ''' perform tSNE and get min, max and norm '''
    x_tsne = tsne.fit_transform(embeddings)
    #print("Data has the {} before tSNE and the following after tSNE {}".format(x.shape[-1], x_tsne.shape[-1]))
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    X_norm = (x_tsne - x_min) / (x_max - x_min)
    ''' plot results of tSNE '''
    colors = ['lightcoral', 'maroon', 'k', 'grey', 'orange', 'darkslategrey', 'lightskyblue', 'plum', 'yellow', 'sienna', 'green']
    y_cls = y_cls.astype(int)
    class_color = [colors[label] for label in y_cls]

    plt.figure(1, figsize=(8, 8))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=class_color, s=5)
    plt.savefig("rnn_tsne.png")
    plt.title("tSNE RNN")
    plt.close("all")