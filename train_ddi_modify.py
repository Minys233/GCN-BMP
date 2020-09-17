#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/8/2018 6:54 PM
# @Author  : chinshin
# @FileName: train_ddi.py
import os
import sys
import csv
import copy
import pickle
import random
import chainer
import logging
import numpy as np
from chainer.datasets import split_dataset_random
from chainer import cuda
from chainer import functions as F
from chainer import optimizers
from chainer import training
from chainer import Variable
from chainer.iterators import SerialIterator
from chainer.training import extensions as E
from chainer.training import triggers
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from os.path import dirname, abspath

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_PATH)
from utils import load_csv, index2id
from parsers import CSVFileParserForPair
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry.dataset.parsers import CSVFileParser
from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.training.extensions import ROCAUCEvaluator, PRCAUCEvaluator, PrecisionEvaluator, RecallEvaluator, F1Evaluator, AccuracyEvaluator
from chainer_chemistry.models import MLP, NFP, GGNN, SchNet, WeaveNet, RSGCN, Regressor, Classifier, Cosine

logging.basicConfig(format='%(asctime)s: %(filename)s: %(funcName)s: %(lineno)d: %(message)s', level=logging.INFO)
global_seed = 2018
random.seed(global_seed)


class GraphConvPredictorForPair(chainer.Chain):
    def __init__(self, graph_conv, mlp=None):
        """Initializes the graph convolution predictor.

        Args:
            graph_conv: The graph convolution network required to obtain
                        molecule feature representation.
            mlp: Multi layer perceptron; used as the final fully connected
                 layer. Set it to `None` if no operation is necessary
                 after the `graph_conv` calculation.
        """

        super(GraphConvPredictorForPair, self).__init__()
        with self.init_scope():
            self.graph_conv = graph_conv
            if isinstance(mlp, chainer.Link):
                self.mlp = mlp
        if not isinstance(mlp, chainer.Link):
            self.mlp = mlp

    def __call__(self, atoms_1, adjs_1, atoms_2, adjs_2):
        h1 = self.graph_conv(atoms_1, adjs_1)
        h2 = self.graph_conv(atoms_2, adjs_2)
        if self.mlp.__class__.__name__ == 'MLP':
            h = F.concat((h1, h2), axis=-1)
            h = self.mlp(h)
            return h
        elif self.mlp.__class__.__name__ == 'Cosine':
            h = self.mlp(h1, h2)
            return h
        else:
            ValueError('[ERROR] No methods for similarity prediction')

    def predict(self, atoms_1, adjs_1, atoms_2, adjs_2):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            x = self.__call__(atoms_1, adjs_1, atoms_2, adjs_2)
            return F.sigmoid(x)



# def set_up_predictor(method, n_unit, conv_layers, class_num):
#     """Sets up the graph convolution network  predictor.
#
#     Args:
#         method: Method name. Currently, the supported ones are `nfp`, `ggnn`,
#                 `schnet`, `weavenet` and `rsgcn`.
#         n_unit: Number of hidden units.
#         conv_layers: Number of convolutional layers for the graph convolution
#                      network.
#         class_num: Number of output classes.
#
#     Returns:
#         An instance of the selected predictor.
#     """
#
#     mlp = MLP(out_dim=class_num, hidden_dim=n_unit)
#
#     if method == 'nfp':
#         print('Training an NFP predictor...')
#         nfp = NFP(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers)
#         predictor = GraphConvPredictorForPair(nfp, mlp)
#     elif method == 'ggnn':
#         print('Training a GGNN predictor...')
#         ggnn = GGNN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers)
#         predictor = GraphConvPredictorForPair(ggnn, mlp)
#     elif method == 'schnet':
#         print('Training an SchNet predictor...')
#         schnet = SchNet(out_dim=class_num, hidden_dim=n_unit,
#                         n_layers=conv_layers)
#         predictor = GraphConvPredictorForPair(schnet, None)
#     elif method == 'weavenet':
#         print('Training a WeaveNet predictor...')
#         n_atom = 20
#         n_sub_layer = 1
#         weave_channels = [50] * conv_layers
#
#         weavenet = WeaveNet(weave_channels=weave_channels, hidden_dim=n_unit,
#                             n_sub_layer=n_sub_layer, n_atom=n_atom)
#         predictor = GraphConvPredictorForPair(weavenet, mlp)
#     elif method == 'rsgcn':
#         print('Training an RSGCN predictor...')
#         rsgcn = RSGCN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers)
#         predictor = GraphConvPredictorForPair(rsgcn, mlp)
#     else:
#         raise ValueError('[ERROR] Invalid method: {}'.format(method))
#     return predictor


def set_up_predictor(method, fp_hidden_dim, fp_out_dim, conv_layers, concat_hidden, fp_dropout_rate, net_hidden_dims, class_num, sim_method='mlp'):
    if sim_method == 'mlp':
        mlp = MLP(out_dim=class_num, hidden_dims=net_hidden_dims)
    elif sim_method == 'cosine':
        mlp = Cosine()
    else:
        raise ValueError('[ERROR] Invalid similarity method: {}'.format(method))
    print('Using {} as similarity predictor with hidden_dims {}...'.format(sim_method, net_hidden_dims))

    if method == 'nfp':
        print('Training an NFP predictor...')
        nfp = NFP(out_dim=fp_out_dim, hidden_dim=fp_hidden_dim, n_layers=conv_layers, concat_hidden=concat_hidden)
        predictor = GraphConvPredictorForPair(nfp, mlp)
    elif method == 'ggnn':
        print('Training a GGNN predictor...')
        ggnn = GGNN(out_dim=fp_out_dim, hidden_dim=fp_hidden_dim, n_layers=conv_layers, concat_hidden=concat_hidden, dropout_rate=fp_dropout_rate)
        predictor = GraphConvPredictorForPair(ggnn, mlp)
    elif method == 'schnet':
        print('Training an SchNet predictor...')
        schnet = SchNet(out_dim=class_num, hidden_dim=fp_hidden_dim,
                        n_layers=conv_layers)
        predictor = GraphConvPredictorForPair(schnet, None)
    elif method == 'weavenet':
        print('Training a WeaveNet predictor...')
        n_atom = 20
        n_sub_layer = 1
        weave_channels = [50] * conv_layers

        weavenet = WeaveNet(weave_channels=weave_channels, hidden_dim=fp_hidden_dim,
                            n_sub_layer=n_sub_layer, n_atom=n_atom)
        predictor = GraphConvPredictorForPair(weavenet, mlp)
    elif method == 'rsgcn':
        print('Training an RSGCN predictor...')
        rsgcn = RSGCN(out_dim=fp_out_dim, hidden_dim=fp_hidden_dim, n_layers=conv_layers)
        predictor = GraphConvPredictorForPair(rsgcn, mlp)
    else:
        raise ValueError('[ERROR] Invalid method: {}'.format(method))
    return predictor


def parse_arguments():
    # Lists of supported preprocessing methods/models.
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'ecfp']
    sim_method_list = ['mlp', 'cosine']

    # Set up the argument parser.
    parser = ArgumentParser(description='Classification on ddi dataset')
    parser.add_argument('--datafile', '-d', type=str,
                        default='ddi_train.csv',
                        help='csv file containing the dataset')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        help='method name', default='nfp')
    parser.add_argument('--sim-method', type=str, choices=sim_method_list,
                        help='similarity method', default='mlp')
    parser.add_argument('--label', '-l', nargs='+',
                        default=['label', ],
                        help='target label for classification')
    parser.add_argument('--class-names', type=str,
                        default=['interaction', 'no interactions'],
                        help='class names in classification task')
    parser.add_argument('--conv-layers', '-c', type=int, default=4,
                        help='number of convolution layers')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='batch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='id of gpu to use; negative value means running'
                             'the code on cpu')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='path to save the computed models to')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='learning rate of optimizer')
    parser.add_argument('--weight-decay-rate', type=float, default=0.,
                        help='weight decay rate of optimizer')
    parser.add_argument('--exp-shift-rate', type=float, default=1.,
                        help='exponential shift rate')
    parser.add_argument('--lin-shift-rate', type=float, default=0.,
                        help='linear shift rate')
    parser.add_argument('--unit-num', '-u', type=int, default=16,
                        help='number of units in one layer of the models')
    parser.add_argument('--fp-out-dim', type=int, default=16,
                        help='dimensionality of output of dynamic fingerprint')
    parser.add_argument('--fp-hidden-dim', type=int, default=16,
                        help='dimensionality of hidden units in dynamic fingerprint')
    parser.add_argument('--concat-hidden', type=bool, default=False,
                        help='whether to concatenate the hidden states in all graphconv layers')
    parser.add_argument('--fp-dropout-rate', type=float, default=0.0,
                        help='dropout rate in graph convolutional neural network')
    parser.add_argument('--net-hidden-dims', type=str, default='32,16',
                        help='dimensionality of hidden units in neural network for similarity prediction')
    parser.add_argument('--net-layer-num', type=int, default=2,
                        help='number of layers in neural network for similarity prediction')
    parser.add_argument('--seed', '-s', type=int, default=777,
                        help='random seed value')
    parser.add_argument('--train-data-ratio', '-r', type=float, default=0.8,
                        help='ratio of training data w.r.t the dataset')
    parser.add_argument('--protocol', type=int, default=2,
                        help='pickle protocol version')
    parser.add_argument('--models-filename', type=str, default='classifier.pkl',
                        help='saved models filename')
    parser.add_argument('--resume', type=str, default='',
                        help='path to a trainer snapshot')
    return parser.parse_args()


def main():
    # Parse the arguments.
    args = parse_arguments()

    if args.label:
        labels = args.label
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        raise ValueError('No target label was specified.')
    # Dataset preparation. Postprocessing is required for the regression task.
    def postprocess_label(label_list):
        label_arr = np.asarray(label_list, dtype=np.int32)
        return label_arr

    # Apply a preprocessor to the dataset.
    print('Preprocessing dataset...')
    preprocessor = preprocess_method_dict[args.method]()
    parser = CSVFileParserForPair(preprocessor, postprocess_label=postprocess_label,
                                  labels=labels, smiles_cols=['smiles_1', 'smiles_2'])
    dataset = parser.parse(args.datafile)['dataset']

    # Split the dataset into training and validation.
    train_data_size = int(len(dataset) * args.train_data_ratio)
    train, val = split_dataset_random(dataset, train_data_size, args.seed)

    # Set up the predictor.
    # def set_up_predictor(method, fp_hidden_dim, fp_out_dim, conv_layers, net_hidden_num, class_num, net_layers):
    # predictor = set_up_predictor(args.method, args.unit_num,
    #                              args.conv_layers, class_num)
    if len(args.net_hidden_dims):
        net_hidden_dims = tuple([int(net_hidden_dim) for net_hidden_dim in args.net_hidden_dims.split(',')])
    else:
        net_hidden_dims = ()
    predictor = set_up_predictor(method=args.method,
                                 fp_hidden_dim=args.fp_hidden_dim, fp_out_dim=args.fp_out_dim, conv_layers=args.conv_layers,
                                 concat_hidden=args.concat_hidden, fp_dropout_rate=args.fp_dropout_rate,
                                 net_hidden_dims=net_hidden_dims, class_num=class_num,
                                 sim_method=args.sim_method)

    # Set up the iterator.
    train_iter = SerialIterator(train, args.batchsize)
    val_iter = SerialIterator(val, args.batchsize,
                              repeat=False, shuffle=False)

    metrics_fun = {'accuracy': F.binary_accuracy}
    classifier = Classifier(predictor, lossfun=F.sigmoid_cross_entropy,
                            metrics_fun=metrics_fun, device=args.gpu)

    # Set up the optimizer.
    optimizer = optimizers.Adam(alpha=args.learning_rate, weight_decay_rate=args.weight_decay_rate)
    # optimizer = optimizers.Adam()
    # optimizer = optimizers.SGD(lr=args.learning_rate)
    optimizer.setup(classifier)

    # Set up the updater.
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu,
                                       converter=concat_mols)

    # Set up the trainer.
    print('Training...')
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(E.Evaluator(val_iter, classifier,
                               device=args.gpu, converter=concat_mols))

    train_eval_iter = SerialIterator(train, args.batchsize,
                                       repeat=False, shuffle=False)

    trainer.extend(AccuracyEvaluator(
        train_eval_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='train_acc',
        pos_labels=1, ignore_labels=-1, raise_value_error=False))
    # extension name='validation' is already used by `Evaluator`,
    # instead extension name `val` is used.
    trainer.extend(AccuracyEvaluator(
        val_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='val_acc',
        pos_labels=1, ignore_labels=-1))

    trainer.extend(ROCAUCEvaluator(
        train_eval_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='train_roc',
        pos_labels=1, ignore_labels=-1, raise_value_error=False))
    # extension name='validation' is already used by `Evaluator`,
    # instead extension name `val` is used.
    trainer.extend(ROCAUCEvaluator(
        val_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='val_roc',
        pos_labels=1, ignore_labels=-1))

    trainer.extend(PRCAUCEvaluator(
        train_eval_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='train_prc',
        pos_labels=1, ignore_labels=-1, raise_value_error=False))
    # extension name='validation' is already used by `Evaluator`,
    # instead extension name `val` is used.
    trainer.extend(PRCAUCEvaluator(
        val_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='val_prc',
        pos_labels=1, ignore_labels=-1))

    # trainer.extend(PrecisionEvaluator(
    #     train_eval_iter, classifier, eval_func=predictor,
    #     device=args.gpu, converter=concat_mols, name='train_p',
    #     pos_labels=1, ignore_labels=-1, raise_value_error=False))
    # # extension name='validation' is already used by `Evaluator`,
    # # instead extension name `val` is used.
    # trainer.extend(PrecisionEvaluator(
    #     val_iter, classifier, eval_func=predictor,
    #     device=args.gpu, converter=concat_mols, name='val_p',
    #     pos_labels=1, ignore_labels=-1))
    #
    # trainer.extend(RecallEvaluator(
    #     train_eval_iter, classifier, eval_func=predictor,
    #     device=args.gpu, converter=concat_mols, name='train_r',
    #     pos_labels=1, ignore_labels=-1, raise_value_error=False))
    # # extension name='validation' is already used by `Evaluator`,
    # # instead extension name `val` is used.
    # trainer.extend(RecallEvaluator(
    #     val_iter, classifier, eval_func=predictor,
    #     device=args.gpu, converter=concat_mols, name='val_r',
    #     pos_labels=1, ignore_labels=-1))

    trainer.extend(F1Evaluator(
        train_eval_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='train_f',
        pos_labels=1, ignore_labels=-1, raise_value_error=False))
    # extension name='validation' is already used by `Evaluator`,
    # instead extension name `val` is used.
    trainer.extend(F1Evaluator(
        val_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='val_f',
        pos_labels=1, ignore_labels=-1))

    # apply shift strategy to learning rate every 10 epochs
    # trainer.extend(E.ExponentialShift('alpha', args.exp_shift_rate), trigger=(10, 'epoch'))
    trainer.extend(E.ExponentialShift('alpha', args.exp_shift_rate),
                   trigger=triggers.ManualScheduleTrigger([10, 20, 30, 40, 50], 'epoch'))
    # # observation of learning rate
    trainer.extend(E.observe_lr(), trigger=(1, 'iteration'))

    entries = [
        'epoch',
        'main/loss', 'train_acc/main/accuracy', 'train_roc/main/roc_auc', 'train_prc/main/prc_auc',
        # 'train_p/main/precision', 'train_r/main/recall',
        'train_f/main/f1',
        'validation/main/loss', 'val_acc/main/accuracy', 'val_roc/main/roc_auc', 'val_prc/main/prc_auc',
        # 'val_p/main/precision', 'val_r/main/recall',
        'val_f/main/f1',
        'lr',
        'elapsed_time']
    trainer.extend(E.PrintReport(entries=entries))
    trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(E.LogReport())
    trainer.extend(E.ProgressBar())

    if args.resume:
        resume_path = os.path.join(args.out, args.resume)
        logging.info('Resume training according to snapshot in {}'.format(resume_path))
        chainer.serializers.load_npz(resume_path, trainer)

    trainer.run()

    # Save the regressor's parameters.
    model_path = os.path.join(args.out, args.model_filename)
    print('Saving the trained models to {}...'.format(model_path))
    classifier.save_pickle(model_path, protocol=args.protocol)


if __name__ == '__main__':
    logging.info(ROOT_PATH)

    main()
