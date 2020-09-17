#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/4/2019 1:12 PM
# @Author  : chinshin
# @FileName: train_ggnn_hole_with_gwm.py
import logging
import os
import pickle

import random
import sys

import chainer
import matplotlib
import numpy as np

matplotlib.use('AGG')
from chainer.backends import cuda
from chainer import functions as F
from chainer import optimizers
from chainer import training
from chainer.iterators import SerialIterator
from chainer.training import extensions as E
from chainer.training import triggers
from argparse import ArgumentParser
from chainer import functions
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.training.extensions import ROCAUCEvaluator, PRCAUCEvaluator, F1Evaluator, AccuracyEvaluator
from chainer_chemistry.models import Classifier

from os.path import dirname, abspath

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_PATH)
from parsers import CSVFileParserForPair
from models import GGNN, GGNN_GWM
from models import NTN, HolE, DistMult, MLP
from setting import *

logging.basicConfig(format='%(asctime)s: %(filename)s: %(funcName)s: %(lineno)d: %(message)s', level=logging.INFO)
random.seed(GLOBAL_SEED)


class GraphConvPredictorForPair(chainer.Chain):
    def __init__(self, graph_conv, mlp=None, symmetric=None):
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
        self.symmetric = symmetric

    def __call__(self, atoms_1, adjs_1, atoms_2, adjs_2):
        if self.xp == cuda.cupy:
            atoms_1 = cuda.to_gpu(atoms_1)
            adjs_1 = cuda.to_gpu(adjs_1)
            atoms_2 = cuda.to_gpu(atoms_2)
            adjs_2 = cuda.to_gpu(adjs_2)

        h1 = self.graph_conv(atoms_1, adjs_1)
        h2 = self.graph_conv(atoms_2, adjs_2)

        if self.mlp.__class__.__name__ == 'MLP':
            h = F.concat((h1, h2), axis=-1)
            h = self.mlp(h)
            return h
        elif self.mlp.__class__.__name__ == 'NTN':
            h = self.mlp(h1, h2)
            return h
        elif self.mlp.__class__.__name__ == 'SymMLP':
            h = self.mlp(h1, h2)
            return h
        elif self.mlp.__class__.__name__ == 'HolE':
            h = self.mlp(h1, h2)
            return h
        elif self.mlp.__class__.__name__ == 'DistMult':
            h = self.mlp(h1, h2)
            return h
        elif self.mlp.__class__.__name__ == 'Cosine':
            h = self.mlp(h1, h2)
            return h
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


class GraphConvPredictorForPairGWM(chainer.Chain):
    def __init__(self, graph_conv, mlp=None, symmetric=None):
        """Initializes the graph convolution predictor.

        Args:
            graph_conv: The graph convolution network required to obtain
                        molecule feature representation.
            mlp: Multi layer perceptron; used as the final fully connected
                 layer. Set it to `None` if no operation is necessary
                 after the `graph_conv` calculation.
        """

        super(GraphConvPredictorForPairGWM, self).__init__()
        with self.init_scope():
            self.graph_conv = graph_conv
            if isinstance(mlp, chainer.Link):
                self.mlp = mlp
        if not isinstance(mlp, chainer.Link):
            self.mlp = mlp
        self.symmetric = symmetric

    def __call__(self, atoms_1, adjs_1, super_nodes_1, atoms_2, adjs_2, super_nodes_2):
        if self.xp == cuda.cupy:
            atoms_1 = cuda.to_gpu(atoms_1)
            adjs_1 = cuda.to_gpu(adjs_1)
            super_nodes_1 = cuda.to_gpu(super_nodes_1)
            atoms_2 = cuda.to_gpu(atoms_2)
            adjs_2 = cuda.to_gpu(adjs_2)
            super_nodes_2 = cuda.to_gpu(super_nodes_2)

        h1 = self.graph_conv(atoms_1, adjs_1, super_nodes_1)
        h2 = self.graph_conv(atoms_2, adjs_2, super_nodes_2)

        if self.mlp.__class__.__name__ == 'MLP':
            h = F.concat((h1, h2), axis=-1)
            h = self.mlp(h)
            return h
        elif self.mlp.__class__.__name__ == 'NTN':
            h = self.mlp(h1, h2)
            return h
        elif self.mlp.__class__.__name__ == 'SymMLP':
            h = self.mlp(h1, h2)
            return h
        elif self.mlp.__class__.__name__ == 'HolE':
            h = self.mlp(h1, h2)
            return h
        elif self.mlp.__class__.__name__ == 'DistMult':
            h = self.mlp(h1, h2)
            return h
        elif self.mlp.__class__.__name__ == 'Cosine':
            h = self.mlp(h1, h2)
            return h
        else:
            ValueError('[ERROR] No methods for similarity prediction')

    def predict(self, atoms_1, adjs_1, super_nodes_1, atoms_2, adjs_2, super_nodes_2):
        if self.symmetric is None:
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                x = self.__call__(atoms_1, adjs_1, super_nodes_1, atoms_2, adjs_2, super_nodes_2)
                target = F.sigmoid(x)
                if self.xp == cuda.cupy:
                    target = cuda.to_gpu(target)
                return target
        elif self.symmetric == 'or' or self.symmetric == 'and':
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                x1 = self.__call__(atoms_1, adjs_1, super_nodes_1, atoms_2, adjs_2, super_nodes_2)
                target1 = F.sigmoid(x1)
                x2 = self.__call__(atoms_2, adjs_2, super_nodes_1, atoms_1, adjs_1, super_nodes_2)
                target2 = F.sigmoid(x2)
                if self.xp == cuda.cupy:
                    target1 = cuda.to_gpu(target1)
                    target2 = cuda.to_gpu(target2)
                if self.symmetric == 'or':
                    target = self.xp.max([target1, target2])
                elif self.symmetric == 'and':
                    target = self.xp.min([target1, target2])
                return target


def set_up_predictor(method, fp_hidden_dim, fp_out_dim, conv_layers, concat_hidden, layer_aggregator,
                     fp_dropout_rate, fp_batch_normalization,
                     net_hidden_dims, class_num,
                     sim_method='mlp', fp_attention=False, weight_typing=True, attention_tying=True,
                     update_attention=False,
                     context=False, context_layers=1, context_dropout=0.,
                     message_function='matrix_multiply', readout_function='graph_level',
                     num_timesteps=3, num_output_hidden_layers=0, output_hidden_dim=16,
                     output_activation=functions.relu,
                     symmetric=None,
                     n_heads=8, dropout_ratio=0.5,
                     ):
    if sim_method == 'mlp':
        logging.info('Using multi-layer perceptron for the learning of composite representation')
        mlp = MLP(out_dim=class_num, hidden_dims=net_hidden_dims)
    elif sim_method == 'ntn':
        logging.info('Using neural tensor network for the learning of composite representation')
        ntn_out_dim = 8
        logging.info('NTN out dim is {}'.format(ntn_out_dim))
        mlp = NTN(left_dim=fp_out_dim, right_dim=fp_out_dim, out_dim=class_num,
                  ntn_out_dim=ntn_out_dim, hidden_dims=net_hidden_dims)
    elif sim_method == 'symmlp':
        logging.info('Using symmetric multi-layer perceptron for the learning of composite representation')
        mlp = MLP(out_dim=class_num, hidden_dims=net_hidden_dims)
    elif sim_method == 'hole':
        logging.info('Using holegraphic embedding for the learning of composite representation')
        mlp = HolE(out_dim=class_num, hidden_dims=net_hidden_dims)
    elif sim_method == 'dist-mult':
        logging.info('Using DistMult embedding for the learning of composite representation')
        dm_out_dim = 8
        mlp = DistMult(left_dim=fp_out_dim, right_dim=fp_out_dim, out_dim=class_num,
                       dm_out_dim=dm_out_dim, hidden_dims=net_hidden_dims)
    else:
        raise ValueError('[ERROR] Invalid similarity method: {}'.format(method))
    logging.info('Using {} as similarity predictor with hidden_dims {}...'.format(sim_method, net_hidden_dims))

    if method == 'ggnn':
        logging.info('Training a GGNN predictor...')
        if fp_attention:
            logging.info('Self-attention mechanism is utilized...')
            if attention_tying:
                logging.info('Self-attention is tying...')
            else:
                logging.info('Self-attention is not tying...')
        if update_attention:
            logging.info('Self-attention mechanism is utilized in update...')
            if attention_tying:
                logging.info('Self-attention is tying...')
            else:
                logging.info('Self-attention is not tying...')
        if not weight_typing:
            logging.info('Weight is not tying...')
        if fp_dropout_rate != 0.0:
            logging.info('Using dropout whose rate is {}...'.format(fp_dropout_rate))
        if fp_batch_normalization:
            logging.info('Using batch normalization in dynamic fingerprint...')
        if concat_hidden:
            logging.info('Incorporating layer aggregation via concatenation after readout...')
        if layer_aggregator:
            logging.info('Incorporating layer aggregation via {} before readout...'.format(layer_aggregator))
        if context:
            logging.info('Context embedding is utilized...')
            logging.info('Number of context layers is {}...'.format(context_layers))
            logging.info('Dropout rate of context layers is {:.2f}'.format(context_dropout))
        logging.info('Message function is {}'.format(message_function))
        logging.info('Readout function is {}'.format(readout_function))
        logging.info('Num_timesteps = {}, num_output_hidden_layers={}, output_hidden_dim={}'.format(
            num_timesteps, num_output_hidden_layers, output_hidden_dim
        ))
        # num_timesteps=3, num_output_hidden_layers=0, output_hidden_dim=16, output_activation=functions.relu,
        ggnn = GGNN(out_dim=fp_out_dim, hidden_dim=fp_hidden_dim, n_layers=conv_layers, concat_hidden=concat_hidden,
                    layer_aggregator=layer_aggregator,
                    dropout_rate=fp_dropout_rate, batch_normalization=fp_batch_normalization,
                    use_attention=fp_attention, weight_tying=weight_typing, attention_tying=attention_tying,
                    context=context, message_function=message_function, readout_function=readout_function,
                    num_timesteps=num_timesteps, num_output_hidden_layers=num_output_hidden_layers,
                    output_hidden_dim=output_hidden_dim, output_activation=output_activation)
        predictor = GraphConvPredictorForPair(ggnn, mlp, symmetric=symmetric)

    elif method == 'ggnn-gwm':
        logging.info('Training an ggnn-gwm predictor...')
        logging.info('N-heads is {}, dropout-ratio is {:.3f}'.format(n_heads, dropout_ratio))
        ggnn_gwm = GGNN_GWM(out_dim=fp_out_dim, hidden_dim=fp_hidden_dim, hidden_dim_super=fp_hidden_dim,
                            n_layers=conv_layers, n_super_feature=300,

                            n_heads=n_heads,
                            dropout_ratio=dropout_ratio,

                            concat_hidden=concat_hidden, weight_tying=weight_typing)
        predictor = GraphConvPredictorForPairGWM(ggnn_gwm, mlp, symmetric=symmetric)
    else:
        raise ValueError('[ERROR] Invalid method: {}'.format(method))
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


def augment_smiles_pairs(smiles_pairs):
    smiles1 = smiles_pairs[:, 0]
    smiles2 = smiles_pairs[:, 1]
    new_smiles1 = np.concatenate((smiles1, smiles2), axis=0)
    new_smiles2 = np.concatenate((smiles2, smiles1), axis=0)
    new_smiles_pairs = np.concatenate(
        (np.expand_dims(new_smiles1, axis=1),
         np.expand_dims(new_smiles2, axis=1)),
        axis=1)
    return new_smiles_pairs


def load_super_nodes(filename='super_nodes.txt'):
    """
    add super nodes for each drug molecule.
    feature vector incoporates original information used in MPNN.
    """

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    filepath = os.path.join(ROOT_PATH, 'examples', 'ddi', filename)
    smiles_list = list()
    atom_feat_list = list()
    with open(filepath, 'r') as reader:
        for line in reader.readlines():
            line = line.strip('\n')
            smiles, feat_str = line.split('\t')[0], line.split('\t')[-1]
            feat = [float(digit) for digit in feat_str.split(' ') if is_number(digit)]
            smiles_list.append(smiles)
            atom_feat_list.append(feat)
    return smiles_list, atom_feat_list


def add_super_nodes(dataset, smiles_pairs):
    dataset_tuple = dataset.get_datasets()
    atoms1, adjs1, atoms2, adjs2, labels = dataset_tuple
    smiles1 = smiles_pairs[:, 0]
    smiles2 = smiles_pairs[:, 1]

    pickle_filename = 'smiles2mol_vec.pkl'
    pickle_filepath = os.path.join(SUPER_NODE_PATH, pickle_filename)
    with open(pickle_filepath, 'r') as reader:
        smiles2atom_feat = pickle.load(reader)
    atom_feat1 = [smiles2atom_feat[smiles] for smiles in smiles1]
    atom_feat1 = np.array(atom_feat1, dtype=np.float32)
    atom_feat2 = [smiles2atom_feat[smiles] for smiles in smiles2]
    atom_feat2 = np.array(atom_feat2, dtype=np.float32)

    new_dataset = NumpyTupleDataset(atoms1, adjs1, atom_feat1, atoms2, adjs2, atom_feat2, labels)
    return new_dataset


def parse_arguments():
    # Lists of supported preprocessing methods/models.
    method_list = ['ggnn', 'ggnn-gwm']
    sim_method_list = ['mlp', 'cosine', 'ntn', 'symmlp', 'hole', 'dist-mult']
    layer_aggregator_list = ['gru-attn', 'gru', 'lstm-attn', 'lstm', 'attn', 'self-attn', 'concat', 'max-pool']

    # Set up the argument parser.
    parser = ArgumentParser(description='Classification on ddi dataset')

    parser.add_argument('--train-datafile', type=str,
                        default='./dataset/interaction/inter_based/ddi_ib_train.csv',
                        help='csv file containing the train dataset')
    parser.add_argument('--valid-datafile', type=str,
                        default='./dataset/interaction/inter_based/ddi_ib_valid.csv',
                        help='csv file containing the valid dataset')
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
    parser.add_argument('--weight-tying', type=str, default=True,
                        help='whether to use the same parameters in all layers(Default: True)')
    parser.add_argument('--attention-tying', type=str, default=True,
                        help='whether to use the same parameter in all attention(Default: True)')
    parser.add_argument('--fp-dropout-rate', type=float, default=0.0,
                        help='dropout rate in graph convolutional neural network')
    parser.add_argument('--fp-bn', type=str, default='False',
                        help='whether to use batch normalization in dynamic fingerprint')
    parser.add_argument('--net-hidden-dims', type=str, default='32,16',
                        help='dimensionality of hidden units in neural network for similarity prediction')
    parser.add_argument('--net-layer-num', type=int, default=2,
                        help='number of layers in neural network for similarity prediction')
    parser.add_argument('--layer-aggregator', type=str, default='', choices=layer_aggregator_list,
                        help='layer aggregator in dynamic fingerprint (Default: )')
    parser.add_argument('--train-data-ratio', '-r', type=float, default=0.8,
                        help='ratio of training data w.r.t the dataset')
    parser.add_argument('--protocol',                                                                                                                                                                                type=int, default=2,
                        help='pickle protocol version')
    parser.add_argument('--models-filename', type=str, default='classifier.pkl',
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

    parser.add_argument('--symmetric', type=str, default=None,
                        help='how to use symmetric in prediction')
    parser.add_argument('--model-filename', type=str, default='classifier.pkl',
                        help='saved models filename')

    parser.add_argument('--n-heads', type=int, default=8, )
    parser.add_argument('--dropout-ratio', type=float, default=0.5)

    return parser.parse_args()



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
    logging.info('Preprocess train dataset and valid dataset...')
    preprocessor = preprocess_method_dict[args.method]()
    parser = CSVFileParserForPair(preprocessor, postprocess_label=postprocess_label,
                                  labels=labels, smiles_cols=['smiles_1', 'smiles_2'])
    train_dict = parser.parse(args.train_datafile, return_smiles_pair_original=True)
    train = train_dict['dataset']
    train_smiles_pairs = train_dict['smiles_pair_original']
    valid_dict = parser.parse(args.valid_datafile, return_smiles_pair_original=True)
    valid = valid_dict['dataset']
    valid_smiles_pairs = valid_dict['smiles_pair_original']

    if augment:
        logging.info('Utilizing data augmentation in train set')
        train = augment_dataset(train)
        if train_smiles_pairs is not None:
            train_smiles_pairs = augment_smiles_pairs(train_smiles_pairs)

    train = add_super_nodes(train, train_smiles_pairs)
    valid = add_super_nodes(valid, valid_smiles_pairs)

    num_train = train.get_datasets()[0].shape[0]
    num_valid = valid.get_datasets()[0].shape[0]
    logging.info('Train/valid split: {}/{}'.format(num_train, num_valid))

    # Set up the predictor.
    if len(args.net_hidden_dims):
        net_hidden_dims = tuple([int(net_hidden_dim) for net_hidden_dim in args.net_hidden_dims.split(',')])
    else:
        net_hidden_dims = ()
    fp_attention = True if args.fp_attention else False
    update_attention = True if args.update_attention else False
    weight_tying = False if args.weight_tying == 'False' else True
    attention_tying = False if args.attention_tying == 'False' else True
    fp_batch_normalization = True if args.fp_bn == 'True' else False
    layer_aggregator = None if args.layer_aggregator == '' else args.layer_aggregator
    context = False if args.context == 'False' else True
    output_activation = functions.relu if args.output_activation == 'relu' else None
    n_heads = args.n_heads
    dropout_ratio = args.dropout_ratio
    predictor = set_up_predictor(method=args.method,
                                 fp_hidden_dim=args.fp_hidden_dim, fp_out_dim=args.fp_out_dim,
                                 conv_layers=args.conv_layers,
                                 concat_hidden=args.concat_hidden, layer_aggregator=layer_aggregator,
                                 fp_dropout_rate=args.fp_dropout_rate, fp_batch_normalization=fp_batch_normalization,
                                 net_hidden_dims=net_hidden_dims, class_num=class_num,
                                 sim_method=args.sim_method, fp_attention=fp_attention, weight_typing=weight_tying,
                                 attention_tying=attention_tying,
                                 update_attention=update_attention,
                                 context=context, context_layers=args.context_layers,
                                 context_dropout=args.context_dropout,
                                 message_function=args.message_function, readout_function=args.readout_function,
                                 num_timesteps=args.num_timesteps,
                                 num_output_hidden_layers=args.num_output_hidden_layers,
                                 output_hidden_dim=args.output_hidden_dim, output_activation=output_activation,
                                 symmetric=args.symmetric,
                                 n_heads=n_heads, dropout_ratio=dropout_ratio
                                 )

    train_iter = SerialIterator(train, args.batchsize)
    valid_iter = SerialIterator(valid, args.batchsize,
                               repeat=False, shuffle=False)

    metrics_fun = {'accuracy': F.binary_accuracy}
    loss_func = F.sigmoid_cross_entropy
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

    trainer.extend(E.Evaluator(valid_iter, classifier,
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
        valid_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='val_acc',
        pos_labels=1, ignore_labels=-1))

    trainer.extend(ROCAUCEvaluator(
        train_eval_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='train_roc',
        pos_labels=1, ignore_labels=-1, raise_value_error=False))
    # extension name='validation' is already used by `Evaluator`,
    # instead extension name `val` is used.
    trainer.extend(ROCAUCEvaluator(
        valid_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='val_roc',
        pos_labels=1, ignore_labels=-1))

    trainer.extend(PRCAUCEvaluator(
        train_eval_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='train_prc',
        pos_labels=1, ignore_labels=-1, raise_value_error=False))
    # extension name='validation' is already used by `Evaluator`,
    # instead extension name `val` is used.
    trainer.extend(PRCAUCEvaluator(
        valid_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='val_prc',
        pos_labels=1, ignore_labels=-1))

    trainer.extend(F1Evaluator(
        train_eval_iter, classifier, eval_func=predictor,
        device=args.gpu, converter=concat_mols, name='train_f',
        pos_labels=1, ignore_labels=-1, raise_value_error=False))
    # extension name='validation' is already used by `Evaluator`,
    # instead extension name `val` is used.
    trainer.extend(F1Evaluator(
        valid_iter, classifier, eval_func=predictor,
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
    trainer.extend(E.snapshot(), trigger=(10, 'epoch'))
    trainer.extend(E.LogReport())
    trainer.extend(E.ProgressBar())
    trainer.extend(E.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(
        E.PlotReport(['train_acc/main/accuracy', 'val_acc/main/accuracy'], 'epoch', file_name='accuracy.png'))

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
