#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/8/2018 6:54 PM
# @Author  : chinshin
# @FileName: train_ddi.py

from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import random
import chainer
import logging
import numpy as np
import matplotlib
matplotlib.use('AGG')
from chainer.backends import cuda
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
from chainer import training
from chainer.iterators import SerialIterator
from chainer.training import extensions as E
from chainer.training import triggers
from argparse import ArgumentParser
from os.path import dirname, abspath

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_PATH)
from parsers import CSVFileParserForPair
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.training.extensions import ROCAUCEvaluator, PRCAUCEvaluator, PrecisionEvaluator, RecallEvaluator, F1Evaluator, AccuracyEvaluator
from chainer_chemistry.models import MLP, NFP, SchNet, WeaveNet, RSGCN, Regressor, Classifier, Cosine
# 引入共注意力机制
from models.coattention.alternating_coattention import AlternatingCoattention
from models.coattention.parallel_coattention import ParallelCoattention, CircularParallelCoattention
from models.coattention.vqa_parallel_coattention import VQAParallelCoattention
from models.coattention.PoolingFineCoattention import PoolingFineCoattention
from models.coattention.lt_fine_coattention import LinearTransformFineCoattention
from models.coattention.nie_coattention import NieFineCoattention
from models.coattention.bimpm import BiMPM
from models.coattention.global_coattention import GlobalCoattention
from models.coattention.neural_coattention import NeuralCoattention
# 稳定版本
# from models.chin_ggnn import GGNN
from models.ggnn_dev import GGNN
# from models.ggnn_dev_self_loop import GGNN
from chainer_chemistry.models import NTN, SymMLP, HolE, DistMult

logging.basicConfig(format='%(asctime)s: %(filename)s: %(funcName)s: %(lineno)d: %(message)s', level=logging.INFO)
global_seed = 2018
random.seed(global_seed)


class GraphConvPredictorForPair(chainer.Chain):
    def __init__(self, graph_conv, attn=None, mlp=None, symmetric=None, siamese=True, another_graph_conv=None,
                 use_i_lstm=False, use_s_lstm=False):
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
            if not siamese:
                self.another_graph_conv = another_graph_conv
            if use_s_lstm:
                self.s_lstm_1 = L.NStepLSTM(
                    n_layers=1,
                    in_size=self.graph_conv.out_dim,
                    out_size=self.graph_conv.out_dim,
                    dropout=0.
                )
                self.s_lstm_2 = L.NStepLSTM(
                    n_layers=1,
                    in_size=self.graph_conv.out_dim,
                    out_size=self.graph_conv.out_dim,
                    dropout=0.
                )
            if use_i_lstm:
                self.i_lstm = L.NStepLSTM(
                    n_layers=1,
                    in_size=self.graph_conv.out_dim * 2,
                    out_size=self.graph_conv.out_dim * 2,
                    dropout=0.
                )
            if isinstance(mlp, chainer.Link):
                self.mlp = mlp
            if isinstance(attn, chainer.Link):
                self.attn = attn
        if not isinstance(mlp, chainer.Link):
            self.mlp = mlp
        if not isinstance(attn, chainer.Link):
            self.attn = attn
        self.symmetric = symmetric
        self.siamese = siamese
        self.use_i_lstm = use_i_lstm
        self.use_s_lstm = use_s_lstm

    def __call__(self, atoms_1, adjs_1, atoms_2, adjs_2):
        if self.xp == cuda.cupy:
            atoms_1 = cuda.to_gpu(atoms_1)
            adjs_1 = cuda.to_gpu(adjs_1)
            atoms_2 = cuda.to_gpu(atoms_2)
            adjs_2 = cuda.to_gpu(adjs_2)

        g1 = self.graph_conv(atoms_1, adjs_1)
        atoms_1_first = self.graph_conv.get_atom_array(0)
        atoms_1_last = self.graph_conv.get_atom_array(-1)
        # atoms_1 = atoms_1_last
        atoms_1 = F.concat([atoms_1_first, atoms_1_last], 2)
        if self.use_i_lstm:
            g_1_list = self.graph_conv.get_g_list()
        if self.use_s_lstm:
            g_1_list = self.graph_conv.get_g_list()
            _, _, g_1_s = self.s_lstm_1(None, None, g_1_list)
            g_1_s = g_1_s[-1]
        if not self.siamese:
            g2 = self.another_graph_conv(atoms_2, adjs_2)
        else:
            g2 = self.graph_conv(atoms_2, adjs_2)
        atoms_2_first = self.graph_conv.get_atom_array(0)
        atoms_2_last = self.graph_conv.get_atom_array(-1)
        # atoms_2 = atoms_2_last
        atoms_2 = F.concat([atoms_2_first, atoms_2_last], 2)
        if self.use_i_lstm:
            g_2_list = self.graph_conv.get_g_list()
        if self.use_s_lstm:
            g_2_list = self.graph_conv.get_g_list()
            _, _, g_2_s = self.s_lstm_2(None, None, g_2_list)
            g_2_s = g_2_s[-1]

        if self.use_i_lstm:
            g_c_list = [F.concat((g_1, g_2), axis=1) for g_1, g_2 in zip(g_1_list, g_2_list)]
            _, _, g_middle = self.i_lstm(None, None, g_c_list)
            g_middle = g_middle[-1]

        g1, g2 = self.attn(atoms_1, g1, atoms_2, g2)

        if self.mlp.__class__.__name__ == 'MLP':
            if self.use_i_lstm and not self.use_s_lstm:
                g = F.concat((g1, g_middle, g2), axis=-1)
            if not self.use_i_lstm and self.use_s_lstm:
                g = F.concat((g1, g_1_s, g_2_s, g2), axis=-1)
            if self.use_i_lstm and self.use_s_lstm:
                g = F.concat((g1, g_1_s, g_middle, g_2_s, g2), axis=-1)
            else:
                g = F.concat((g1, g2), axis=-1)
            g = F.concat((g1, g2), axis=-1)
            g = self.mlp(g)
            return g
        elif self.mlp.__class__.__name__ in ['NTN','SymMLP', 'HolE', 'DistMult', 'Cosine']:
            g = self.mlp(g1, g2)
            return g
        else:
            ValueError('[ERROR] No methods for similarity prediction')

    def predict(self, atoms_1, adjs_1, atoms_2, adjs_2):
        if self.symmetric is None:
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                x = self.__call__(atoms_1, adjs_1, atoms_2, adjs_2)
                target = F.sigmoid(x)
                if self.xp == cuda.cupy:
                    target = cuda.to_gpu(target)
                return target
        elif self.symmetric == 'or' or self.symmetric == 'and':
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                x1 = self.__call__(atoms_1, adjs_1, atoms_2, adjs_2)
                target1 = F.sigmoid(x1)
                x2 = self.__call__(atoms_2, adjs_2, atoms_1, adjs_1)
                target2 = F.sigmoid(x2)
                if self.xp == cuda.cupy:
                    target1 = cuda.to_gpu(target1)
                    target2 = cuda.to_gpu(target2)
                if self.symmetric == 'or':
                    target = self.xp.max([target1, target2])
                elif self.symmetric == 'and':
                    target = self.xp.min([target1, target2])
                return target


def set_up_predictor(method,
                     fp_hidden_dim, fp_out_dim, conv_layers, concat_hidden,
                     fp_dropout_rate, fp_batch_normalization,
                     net_hidden_dims, class_num,
                     weight_typing=True, sim_method='mlp', symmetric=None,
                     attn_model=None
                     ):

    sim_method_dict = {
        'mlp': 'multi-layered perceptron',
        'ntn': 'bilinear transform',
        'symmlp': 'symmetric perceptron',
        'hole': 'holographic embedding',
        'dist-mult': 'dist-mult',
    }

    method_dict = {
        'ggnn': 'GGNN',
        'nfp': 'NFP',
    }

    logging.info('Graph Embedding: {}'.format(method_dict.get(method, None)))
    logging.info('Link Prediction: {}'.format(sim_method_dict.get(sim_method, None)))

    lp = None
    if sim_method == 'mlp':
        lp = MLP(out_dim=class_num, hidden_dims=net_hidden_dims)

    elif sim_method == 'ntn':
        ntn_out_dim = 8
        lp = NTN(left_dim=fp_out_dim, right_dim=fp_out_dim, out_dim=class_num,
                  ntn_out_dim=ntn_out_dim, hidden_dims=net_hidden_dims)

    elif sim_method == 'symmlp':
        lp = MLP(out_dim=class_num, hidden_dims=net_hidden_dims)

    elif sim_method == 'hole':
        lp = HolE(out_dim=class_num, hidden_dims=net_hidden_dims)

    elif sim_method == 'dist-mult':
        dm_out_dim = 8
        lp = DistMult(left_dim=fp_out_dim, right_dim=fp_out_dim, out_dim=class_num,
                       dm_out_dim=dm_out_dim, hidden_dims=net_hidden_dims)
    else:
        raise ValueError('[ERROR] Invalid link prediction model: {}'.format(method))

    attn = None
    scorer = 'bilinear'
    if attn_model == 'alter':
        attn_weight_tying = True
        logging.info('Using alternating co-attention')
        if attn_weight_tying:
            logging.info('Weight is tying')
        attn = AlternatingCoattention(hidden_dim=fp_hidden_dim, out_dim=fp_out_dim, head=8, weight_tying=True)
    elif attn_model == 'para':
        attn_weight_tying = True
        logging.info('Using parallel co-attention')
        logging.info('Scorer is {}'.format(scorer))
        if attn_weight_tying:
            logging.info('Weight is tying')
        attn = ParallelCoattention(hidden_dim=fp_hidden_dim, out_dim=fp_out_dim, head=1,
                                   activation=F.tanh, weight_tying=attn_weight_tying)
    elif attn_model == 'circ':
        logging.info('Using circular based parallel co-attention')
        attn = CircularParallelCoattention(hidden_dim=fp_hidden_dim, out_dim=fp_out_dim,
                                           activation=F.tanh)

    elif attn_model == 'vqa':
        logging.info('Using vqa fine-grained co-attention')
        attn = VQAParallelCoattention(hidden_dim=fp_hidden_dim * 2, out_dim=fp_out_dim, head=8)

    elif attn_model == 'pool':
        logging.info('Using pool fine-graind co-attention')
        attn = PoolingFineCoattention(hidden_dim=fp_hidden_dim * 2, out_dim=fp_out_dim)

    elif attn_model == 'lt':
        logging.info('Using lt fine-grained co-attention')
        attn = LinearTransformFineCoattention(hidden_dim=fp_hidden_dim * 2, out_dim=fp_out_dim)

    elif attn_model == 'nie':
        logging.info('Using nie fine-grained co-attention')
        logging.info('Using activation function tanh')
        # multiplied by 2 for concat
        attn = NieFineCoattention(hidden_dim=fp_hidden_dim * 2, out_dim=fp_out_dim, head=8, activation=F.tanh)

    elif attn_model == 'bimpm':
        logging.info('Using bimpm matching strategy')
        attn = BiMPM(hidden_dim=fp_hidden_dim, out_dim=fp_out_dim, head=fp_out_dim,
                     with_max_pool=True, with_att_mean=True, with_att_max=True, aggr=F.sum)

    elif attn_model == 'global':
        logging.info('Using global coattention')
        attn = GlobalCoattention(hidden_dim=fp_hidden_dim, out_dim=fp_out_dim, weight_tying=True)

    elif attn_model == 'neural':
        logging.info('Using neural coattention')
        attn = NeuralCoattention(hidden_dim=fp_hidden_dim, out_dim=fp_out_dim, weight_tying=True)

    else:
        raise ValueError('[ERROR] Invalid Co-Attention Method.')

    siamese = True
    if not siamese:
        encoder_1, encoder_2 = None, None
    else:
        encoder = None
    if method == 'ggnn':
        if not weight_typing:
            logging.info('Weight is not tying')
        if fp_dropout_rate != 0.0:
            logging.info('Forward propagation dropout rate is {:.1f}'.format(fp_dropout_rate))
        if fp_batch_normalization:
            logging.info('Using batch normalization')
        if concat_hidden:
            logging.info('Using concatenation between layers')
        if not siamese:
            encoder_1 = GGNN(out_dim=fp_out_dim, hidden_dim=fp_hidden_dim, n_layers=conv_layers,
                    concat_hidden=concat_hidden, weight_tying=weight_typing)
            encoder_2 = GGNN(out_dim=fp_out_dim, hidden_dim=fp_hidden_dim, n_layers=conv_layers,
                    concat_hidden=concat_hidden, weight_tying=weight_typing)
        else:
            encoder = GGNN(out_dim=fp_out_dim, hidden_dim=fp_hidden_dim, n_layers=conv_layers,
                    concat_hidden=concat_hidden, weight_tying=weight_typing)
    elif method == 'nfp':
        print('Training an NFP predictor...')
        encoder = NFP(out_dim=fp_out_dim, hidden_dim=fp_hidden_dim, n_layers=conv_layers, concat_hidden=concat_hidden)

    else:
        raise ValueError('[ERROR] Invalid graph embedding encoder.')

    if siamese:
        predictor = GraphConvPredictorForPair(
            encoder, attn, lp, symmetric=symmetric, siamese=siamese, use_s_lstm=True, use_i_lstm=True)
    else:
        predictor = GraphConvPredictorForPair(
            encoder_1, attn, lp, symmetric=symmetric, siamese=siamese, another_graph_conv=encoder_2)
    return predictor

def augment_dataset(dataset):
    dataset_tuple = dataset.get_datasets()
    atoms1, adjs1, atoms2, adjs2, labels = dataset_tuple
    new_atoms1 = np.concatenate((atoms1, atoms2), axis=0)
    new_atoms2 = np.concatenate((atoms2, atoms1), axis=0)
    new_adjs1 = np.concatenate((adjs1, adjs2), axis=0)
    new_adjs2 = np.concatenate((adjs2, adjs1), axis=0)
    new_labels = np.concatenate((labels, labels), axis=0)
    new_dataset = NumpyTupleDataset(new_atoms1, new_adjs1, new_atoms2, new_adjs2, new_labels)
    return new_dataset


def parse_arguments():
    # Lists of supported preprocessing methods/models.
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'ecfp']
    sim_method_list = ['mlp', 'cosine', 'ntn', 'symmlp', 'hole', 'dist-mult']
    layer_aggregator_list = ['gru-attn', 'gru', 'lstm-attn', 'lstm', 'attn', 'self-attn', 'concat', 'max-pool']
    attn_list = ['para', 'alter', 'circ', 'vqa', 'pool', 'lt', 'nie', 'bimpm', 'global', 'neural']

    # Set up the argument parser.
    parser = ArgumentParser(description='Classification on ddi dataset')
    parser.add_argument('--datafile', '-d', type=str,
                        default='ddi_train.csv',
                        help='csv file containing the dataset')
    parser.add_argument('--train-datafile', type=str,
                        default='ddi_train.csv',
                        help='csv file containing the train dataset')
    parser.add_argument('--train-pos-neg-ratio', type=float,
                        default=-1.,
                        help='ratio between positive and negative instances')
    parser.add_argument('--valid-datafile', type=str,
                        default='ddi_test.csv',
                        help='csv file containing the test dataset')
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
    parser.add_argument('--exp-shift-strategy', type=int, default=1,
                        help='strategy to adapt the learning rate manually')
    parser.add_argument('--lin-shift-rate', type=float, default=0.,
                        help='linear shift rate')
    parser.add_argument('--unit-num', '-u', type=int, default=16,
                        help='number of units in one layer of the models')
    parser.add_argument('--fp-out-dim', type=int, default=16,
                        help='dimensionality of output of dynamic fingerprint')
    parser.add_argument('--fp-hidden-dim', type=int, default=16,
                        help='dimensionality of hidden units in dynamic fingerprint')
    parser.add_argument('--fp-attention', type=bool, default=False,
                        help='whether to use attention mechanism in dynamic fingerprint')
    parser.add_argument('--update-attention', type=bool, default=False,
                        help='whether to use attention mechasnim in update')
    parser.add_argument('--concat-hidden', type=bool, default=False,
                        help='whether to concatenate the hidden states in all graphconv layers')
    parser.add_argument('--fp-max-degree', type=int, default=6,
                        help='max degrees of neural fingerprint')
    parser.add_argument('--weight-tying', type=str, default=True,
                        help='whether to use the same parameters in all layers(Default: True)')
    parser.add_argument('--attention-tying', type=str, default=True,
                        help='whether to use the same parameter in all attention(Default: True)')
    parser.add_argument('--fp-dropout-rate', type=float, default=0.0,
                        help='dropout rate in graph convolutional neural network')
    parser.add_argument('--fp-bn', type=str, default='False',
                        help='whether to use batch normalization in dynamic fingerprint')

    parser.add_argument('--attn', type=str, default=None, choices=attn_list,
                        help='indicate the type of co-attention')

    parser.add_argument('--net-hidden-dims', type=str, default='32,16',
                        help='dimensionality of hidden units in neural network for similarity prediction')
    parser.add_argument('--net-layer-num', type=int, default=2,
                        help='number of layers in neural network for similarity prediction')
    parser.add_argument('--layer-aggregator', type=str, default='', choices=layer_aggregator_list,
                        help='layer aggregator in dynamic fingerprint (Default: )')
    parser.add_argument('--seed', '-s', type=int, default=777,
                        help='random seed value')
    parser.add_argument('--train-data-ratio', '-r', type=float, default=0.8,
                        help='ratio of training data w.r.t the dataset')
    parser.add_argument('--protocol', type=int, default=2,
                        help='pickle protocol version')
    parser.add_argument('--model-filename', type=str, default='classifier.pkl',
                        help='saved models filename')
    parser.add_argument('--resume', type=str, default='',
                        help='path to a trainer snapshot')
    parser.add_argument('--context', type=str, default='False',
                        help='whether to use context embedding in dynamic fingerprint')
    parser.add_argument('--context-layers', type=int, default=1,
                        help='number of context layers')
    parser.add_argument('--context-dropout', type=float, default=0.,
                        help='dropout rate of context layers')
    parser.add_argument('--message-function', type=str, default='matrix_multiply',
                        help='message function in dynamic fingerprint (default: matrix_multiply)')

    parser.add_argument('--readout-function', type=str, default='graph_level',
                        help='readout function in dynamic fingerprint (default: graph_level)')
    parser.add_argument('--num-timesteps', type=int, default=3,
                        help='number of timesteps in set2vec readout function')
    parser.add_argument('--num-output-hidden-layers', type=int, default=0,
                        help='number of hidden layers in set2vec readout function')
    parser.add_argument('--output-hidden-dim', type=int, default=16,
                        help='number of hidden units in each hidden layer in set2vec readout function')
    parser.add_argument('--output-activation', type=str, choices=['relu'],
                        default='relu', help='activation function used in set2vec readout function')

    parser.add_argument('--multi-gpu', type=str, default='False',
                        help='whether to use multiple GPUs')

    parser.add_argument('--augment', type=str, default='False',
                        help='whether to use data augment')

    parser.add_argument('--max-norm', type=float, default=0.,
                        help='the maximum value of gradient in back propagation')
    parser.add_argument('--l2-rate', type=float, default=0.,
                        help='coefficient for the L2 regularization')
    parser.add_argument('--l1-rate', type=float, default=0.,
                        help='coefficient for the L1 regularization')

    parser.add_argument('--loss-func', type=str, default='cross-entropy',
                        help='loss function training the models')

    parser.add_argument('--symmetric', type=str, default=None,
                        help='how to use symmetric in prediction')
    return parser.parse_args()


def modify_dataset_for_hinge(dataset):
    atoms1, adjs1, atoms2, adjs2, labels = dataset.get_datasets()
    labels_squeezed = np.squeeze(labels, axis=1)
    new_dataset = NumpyTupleDataset(atoms1, adjs1, atoms2, adjs2, labels_squeezed)
    return new_dataset


def main():
    # Parse the arguments.
    args = parse_arguments()
    augment = False if args.augment == 'False' else True
    multi_gpu = False if args.multi_gpu == 'False' else True
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
    logging.info('Preprocess train dataset and test dataset...')
    preprocessor = preprocess_method_dict[args.method]()
    parser = CSVFileParserForPair(preprocessor, postprocess_label=postprocess_label,
                                  labels=labels, smiles_cols=['smiles_1', 'smiles_2'])
    train = parser.parse(args.train_datafile)['dataset']
    test = parser.parse(args.valid_datafile)['dataset']

    if augment:
        logging.info('Utilizing data augmentation in train set')
        train = augment_dataset(train)

    num_train = train.get_datasets()[0].shape[0]
    num_test = test.get_datasets()[0].shape[0]
    logging.info('Train/test split: {}/{}'.format(num_train, num_test))

    if len(args.net_hidden_dims):
        net_hidden_dims = tuple([int(net_hidden_dim) for net_hidden_dim in args.net_hidden_dims.split(',')])
    else:
        net_hidden_dims = ()

    weight_tying = False if args.weight_tying == 'False' else True
    fp_batch_normalization = True if args.fp_bn == 'True' else False

    predictor = set_up_predictor(method=args.method,
                                 fp_hidden_dim=args.fp_hidden_dim, fp_out_dim=args.fp_out_dim,
                                 conv_layers=args.conv_layers, concat_hidden=args.concat_hidden,
                                 fp_dropout_rate=args.fp_dropout_rate, fp_batch_normalization=fp_batch_normalization,
                                 net_hidden_dims=net_hidden_dims, class_num=class_num,
                                 sim_method=args.sim_method, weight_typing=weight_tying,
                                 symmetric=args.symmetric, attn_model=args.attn,
                                 )

    if args.train_pos_neg_ratio != -1.:
        # Set up the iterator.
        train_dataset = train.get_datasets()
        atoms1_train, adjs1_train, atoms2_train, adjs2_train, labels_train = train_dataset
        labels_train = np.squeeze(labels_train)
        train_dataset_arr = np.concatenate([item[:, None] if len(item.shape) == 1 else item for item in list(train_dataset)], axis=1)
        pos_train_dataset_arr = train_dataset_arr[labels_train == 1]
        num_pos_train = pos_train_dataset_arr.shape[0]
        pos_train_indices = np.arange(0, num_pos_train)
        neg_train_dataset_arr = train_dataset_arr[labels_train == 0]
        num_neg_train = neg_train_dataset_arr.shape[0]
        pos_neg_train_ratio = args.train_pos_neg_ratio
        num_pos_train = int(pos_neg_train_ratio * num_neg_train)
        np.random.seed(777)
        np.random.shuffle(pos_train_indices)
        pos_train_indices = pos_train_indices[:num_pos_train]
        pos_train_dataset_arr = pos_train_dataset_arr[pos_train_indices]
        new_train_dataset_arr = np.concatenate((pos_train_dataset_arr, neg_train_dataset_arr), axis=0)
        atoms1_train, adjs1_train = new_train_dataset_arr[:, 0], new_train_dataset_arr[:, 1]
        atoms2_train, adjs2_train = new_train_dataset_arr[:, 2], new_train_dataset_arr[:, 3]
        labels_train = new_train_dataset_arr[:, 4].astype(np.int32)
        labels_train = np.expand_dims(labels_train, axis=1)
        train = NumpyTupleDataset(atoms1_train, adjs1_train, atoms2_train, adjs2_train, labels_train)
        num_train = train.get_datasets()[0].shape[0]
        num_test = test.get_datasets()[0].shape[0]
        logging.info('Train pos-neg ratio is {:.4f}'.format(args.train_pos_neg_ratio))
        logging.info('Train/test number is {}/{}'.format(num_train, num_test))

    # if args.loss_func == 'hinge':
    #     modify_dataset_for_hinge(train)
    # Set up the iterator.
    train_iter = SerialIterator(train, args.batchsize)
    test_iter = SerialIterator(test, args.batchsize,
                              repeat=False, shuffle=False)

    metrics_fun = {'accuracy': F.binary_accuracy}
    loss_func = F.sigmoid_cross_entropy
    if args.loss_func == 'hinge':
        logging.info('Loss function is {}'.format(args.loss_func))
        loss_func = F.hinge
        metrics_fun = {'accuracy': F.accuracy}
    classifier = Classifier(predictor, lossfun=loss_func,
                            metrics_fun=metrics_fun, device=args.gpu)

    # Set up the optimizer.
    optimizer = optimizers.Adam(alpha=args.learning_rate, weight_decay_rate=args.weight_decay_rate)
    # optimizer = optimizers.Adam()
    # optimizer = optimizers.SGD(lr=args.learning_rate)
    optimizer.setup(classifier)
    # add regularization
    if args.max_norm > 0:
        optimizer.add_hook(chainer.optimizer.GradientClipping(threshold=args.max_norm))
    if args.l2_rate > 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.l2_rate))
    if args.l1_rate > 0:
        optimizer.add_hook(chainer.optimizer.Lasso(rate=args.l1_rate))

    # Set up the updater.
    if multi_gpu:
        logging.info('Using multiple GPUs')
        updater = training.ParallelUpdater(train_iter, optimizer, devices={'main': 0, 'second': 1},
                                           converter=concat_mols)
    else:
        logging.info('Using single GPU')
        updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu,
                                           converter=concat_mols)

    # Set up the trainer.
    logging.info('Training...')
    # add stop_trigger parameter
    early_stop = triggers.EarlyStoppingTrigger(monitor='validation/main/loss', patients=10, max_trigger=(500, 'epoch'))
    out = 'output' + '/' + args.out
    trainer = training.Trainer(updater, stop_trigger=early_stop, out=out)

    # trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(E.Evaluator(test_iter, classifier,
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
        test_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='val_acc',
        pos_labels=1, ignore_labels=-1))

    trainer.extend(ROCAUCEvaluator(
        train_eval_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='train_roc',
        pos_labels=1, ignore_labels=-1, raise_value_error=False))
    # extension name='validation' is already used by `Evaluator`,
    # instead extension name `val` is used.
    trainer.extend(ROCAUCEvaluator(
        test_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='val_roc',
        pos_labels=1, ignore_labels=-1))

    trainer.extend(PRCAUCEvaluator(
        train_eval_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='train_prc',
        pos_labels=1, ignore_labels=-1, raise_value_error=False))
    # extension name='validation' is already used by `Evaluator`,
    # instead extension name `val` is used.
    trainer.extend(PRCAUCEvaluator(
        test_iter, classifier, eval_func=predictor,
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
        test_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='val_f',
        pos_labels=1, ignore_labels=-1))

    # apply shift strategy to learning rate every 10 epochs
    # trainer.extend(E.ExponentialShift('alpha', args.exp_shift_rate), trigger=(10, 'epoch'))
    if args.exp_shift_strategy == 1:
        trainer.extend(E.ExponentialShift('alpha', args.exp_shift_rate),
                       trigger=triggers.ManualScheduleTrigger([10, 20, 30, 40, 50, 60], 'epoch'))
    elif args.exp_shift_strategy == 2:
        trainer.extend(E.ExponentialShift('alpha', args.exp_shift_rate),
                       trigger=triggers.ManualScheduleTrigger([5, 10, 15, 20, 25, 30], 'epoch'))
    elif args.exp_shift_strategy == 3:
        trainer.extend(E.ExponentialShift('alpha', args.exp_shift_rate),
                       trigger=triggers.ManualScheduleTrigger([5, 10, 15, 20, 25, 30, 40, 50, 60, 70], 'epoch'))
    else:
        raise ValueError('No such strategy to adapt learning rate')
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
    # change from 10 to 2 on Mar. 1 2019
    trainer.extend(E.snapshot(), trigger=(2, 'epoch'))
    trainer.extend(E.LogReport())
    trainer.extend(E.ProgressBar())
    trainer.extend(E.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(E.PlotReport(['train_acc/main/accuracy', 'val_acc/main/accuracy'], 'epoch', file_name='accuracy.png'))

    if args.resume:
        resume_path = os.path.join(out, args.resume)
        logging.info('Resume training according to snapshot in {}'.format(resume_path))
        chainer.serializers.load_npz(resume_path, trainer)

    trainer.run()

    # Save the regressor's parameters.
    model_path = os.path.join(out, args.model_filename)
    logging.info('Saving the trained models to {}...'.format(model_path))
    classifier.save_pickle(model_path, protocol=args.protocol)


if __name__ == '__main__':
    logging.info(ROOT_PATH)

    main()
