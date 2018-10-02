import numpy as np
import pickle
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer import serializers
from chainer.training import extensions
from chainer.cuda import to_cpu, to_gpu, cupy

# from scipy.stats import pearsonr

import argparse
import os
import io



class MLP(chainer.Chain):
    def __init__(self, args):
        self.n_layer = args.layer
        self.n_unit = args.unit
        self.l = []

        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, self.n_unit)
            self.l2 = L.Linear(None, self.n_unit)
            self.l3 = L.Linear(None, self.n_unit)
            self.l4 = L.Linear(None, self.n_unit)
            self.l5 = L.Linear(None, self.n_unit)
            self.l_out = L.Linear(None, 1)
        self.dr = args.dropout_rate

    def __call__(self, x):
       if self.n_layer == 0:
           h_out = self.l_out(x)
           return h_out
       else:
            h1 = F.dropout(F.relu(self.l1(x)), ratio=self.dr)
            if self.n_layer == 1:
                h_out = self.l_out(h1)
                return h_out
            h2 = F.dropout(F.relu(self.l2(h1)), ratio=self.dr)
            if self.n_layer == 2:
                h_out = self.l_out(h2)
                return h_out
            h3 = F.dropout(F.relu(self.l3(h2)), ratio=self.dr)
            if self.n_layer == 3:
                h_out = self.l_out(h3)
                return h_out
            h4 = F.dropout(F.relu(self.l4(h3)), ratio=self.dr)
            if self.n_layer == 4:
                h_out = self.l_out(h4)
                return h_out
            h5 = F.dropout(F.relu(self.l5(h4)), ratio=self.dr)
            if self.n_layer == 5:
                h_out = self.l_out(h5)
                return h_out


def load_npz(npz_path):
    inputs_npz = np.load(npz_path)

    features = inputs_npz['features']
    labels = inputs_npz['labels']

    return features, labels


def concat_npz(args):
    last_features = []
    last_labels = []

    SR_features = []
    for SR in args.SR_models.split('_'):
        npz_path = os.path.join(args.npz_dir, SR + '.npz')
        features, labels = load_npz(npz_path)
        SR_features.append(features)
    last_features = [np.concatenate(temp, axis=0) for temp in zip(*SR_features)]
    last_labels.extend(labels)

    return last_features, last_labels



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_dir', type=str, help='input: npz_dir')
    parser.add_argument('--SR_models', type=str, help='eg. IS_QT_USE')
    parser.add_argument('--DA_versions', type=str, default='2015_2016', help='eg. 2015_2016_2017')
    parser.add_argument('--case', type=str, help='true or lower')

    parser.add_argument("--opt", "-o", type=str, default='Adam', help="Name of Optimizer")
    parser.add_argument("--learning_rate", "-lr", type=str, default=None, help="Learning Rate")
    parser.add_argument("--layer", "-l", type=int, default=2, help="Number of Layer")
    parser.add_argument("--unit", "-u", type=int, default=1024, help="Number of units")
    parser.add_argument("--batchsize", "-b", type=int, default=1024, help="Number of translations in each mini-batch")
    parser.add_argument("--dropout_rate", "-dr", type=float, default=0.0, help="Dropout Rate")
    parser.add_argument("--epoch", "-e", type=int, default=0, help="Number of sweeps over the dataset to train")

    parser.add_argument("--gpu", "-g", type=int, default=-1, help="GPU ID (negative value indicates CPU)")
    parser.add_argument("--resume", "-r", default="", help="Resume the training from snapshot")
    args = parser.parse_args()

    xp = cupy
    model = L.Classifier(MLP(args))
    if args.opt == 'Adam':
        path_model = '../models/Trained_2015_2016_{}/Adam_l{}_u{}_b{}_dr{}.snapshot'.format(args.SR_models, args.layer, args.unit, args.batchsize, args.dropout_rate)
    serializers.load_npz(path_model, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    features, labels = concat_npz(args)

    scores = []
    with chainer.using_config('train', False):
        if args.gpu >= 0:
            for feature in features:
                feature = feature[None, ...]
                feature = to_gpu(feature)
                pred = model.predictor(chainer.Variable(feature.astype(xp.float32)))
                score = to_cpu(pred.data)[0][0]
                scores.append(score)
                print(score)
        else:
            for feature in features:
                feature = feature[None, ...]
                pred = model.predictor(chainer.Variable(feature.astype(np.float32)))
                score = pred.data[0][0]
                scores.append(score)
                print(score)

