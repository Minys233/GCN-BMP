#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/20/2019 5:19 PM
# @Author  : chinshin
# @FileName: unsupervised_feature.py
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import sys
import random
import chainer
import logging
import numpy as np
import matplotlib
matplotlib.use('AGG')
from argparse import ArgumentParser
from chainer.backends import cuda
from chainer import functions as F
from chainer import optimizers
from chainer import training
from chainer.iterators import SerialIterator
from chainer.training import extensions as E
from chainer.training import triggers
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.training.extensions import ROCAUCEvaluator, PRCAUCEvaluator, F1Evaluator, AccuracyEvaluator
from chainer_chemistry.models import Classifier

from os.path import dirname, abspath
ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_PATH)
from my_utils.parsers import CSVFileParserForPair, Mol2VecParserForPair, MolAutoencoderParserForPair, SSPParserForPair
from setting import *
from models import MLP, NTN, HolE, DistMult

logging.basicConfig(format='%(asctime)s: %(filename)s: %(funcName)s: %(lineno)d: %(message)s', level=logging.INFO)
random.seed(GLOBAL_SEED)

class DDIPredictor(chainer.Chain):

    def __init__(self, lp=None, symmetric=None):
        super(DDIPredictor, self).__init__()
        with self.init_scope():
            if isinstance(lp, chainer.Link):
                self.lp = lp
        if not isinstance(lp, chainer.Link):
            self.lp = lp
        self.symmetric = symmetric

    def __call__(self, feat1, feat2):
        if self.xp == cuda.cupy:
            feat1 = cuda.to_gpu(feat1)
            feat2 = cuda.to_gpu(feat2)

        h1, h2 = feat1, feat2

        if self.lp.__class__.__name__ == 'MLP':
            h = F.concat((h1, h2), axis=-1)
            h = self.lp(h)
            return h
        elif self.lp.__class__.__name__ in ['NTN', 'SymMLP', 'HolE', 'DistMult', 'Cosine']:
            h = self.lp(h1, h2)
            return h
        else:
            ValueError('[ERROR] No methods for similarity prediction')

    def predict(self, feat1, feat2):
        if self.symmetric is None:
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                x = self.__call__(feat1, feat2)
                target = F.sigmoid(x)
                if self.xp == cuda.cupy:
                    target = cuda.to_gpu(target)
                return target
        elif self.symmetric == 'or' or self.symmetric == 'and':
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                x1 = self.__call__(feat1, feat2)
                target1 = F.sigmoid(x1)
                x2 = self.__call__(feat2, feat1)
                target2 = F.sigmoid(x2)
                if self.xp == cuda.cupy:
                    target1 = cuda.to_gpu(target1)
                    target2 = cuda.to_gpu(target2)
                target = None
                if self.symmetric == 'or':
                    target = self.xp.max([target1, target2])
                elif self.symmetric == 'and':
                    target = self.xp.min([target1, target2])
                if target is not None:
                    return target
                else:
                    raise RuntimeError("Target can not be None when returned.")


def set_up_predictor(fp_out_dim, net_hidden_dims, class_num,
                     sim_method='mlp', symmetric=None):

    sim_method_dict = {
        'mlp': 'multi-layered perceptron',
        'ntn': 'bilinear transform',
        'symmlp': 'symmetric perceptron',
        'hole': 'holographic embedding',
        'dist-mult': 'dist-mult',
    }

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
        raise ValueError('[ERROR] Invalid link prediction model: {}'.format(sim_method))

    predictor = DDIPredictor(lp, symmetric=symmetric)

    return predictor


def parse_arguments():
    # Lists of supported preprocessing methods/models.
    feature_list = ['ssp', 'mol2vec', 'molenc', 'mdp']
    sim_method_list = ['mlp', 'cosine', 'ntn', 'symmlp', 'hole', 'dist-mult']

    # Set up the argument parser.
    parser = ArgumentParser(description='Classification on ddi dataset')

    # arguments for dataset
    parser.add_argument('--train-datafile', type=str,
                        default='ddi_train.csv',
                        help='csv file containing the train dataset')
    parser.add_argument('--valid-datafile', type=str,
                        default='ddi_test.csv',
                        help='csv file containing the test dataset')
    parser.add_argument('--label', '-l', nargs='+',
                        default=['label', ],
                        help='target label for classification')
    parser.add_argument('--class-names', type=str,
                        default=['interaction', 'no interactions'],
                        help='class names in classification task')
    parser.add_argument('--augment', type=str, default='False',
                        help='whether to use data augment')

    # parameters for the total model
    parser.add_argument('--feature', '-f', type=str, choices=feature_list,
                        help='feature name', default='mol2vec')
    parser.add_argument('--sim-method', type=str, choices=sim_method_list,
                        help='similarity method', default='hole')
    parser.add_argument('--fp-out-dim', type=int, default=16,
                        help='dimensionality of output of dynamic fingerprint')
    parser.add_argument('--net-hidden-dims', type=str, default='',
                        help='dimensionality of hidden units in neural network for similarity prediction')

    # parameters for regularization
    parser.add_argument('--max-norm', type=float, default=0.,
                        help='the maximum value of gradient in back propagation')
    parser.add_argument('--l2-rate', type=float, default=0.,
                        help='coefficient for the L2 regularization')
    parser.add_argument('--l1-rate', type=float, default=0.,
                        help='coefficient for the L1 regularization')

    parser.add_argument('--symmetric', type=str, default=None,
                        help='how to use symmetric in prediction')

    # parameters for model training
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='batch size')
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

    # parameters for device
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='id of gpu to use; negative value means running'
                             'the code on cpu')

    # parameters for output
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='path to save the computed models to')
    parser.add_argument('--protocol', type=int, default=2,
                        help='pickle protocol version')
    parser.add_argument('--model-filename', type=str, default='classifier.pkl',
                        help='saved models filename')
    parser.add_argument('--resume', type=str, default='',
                        help='path to a trainer snapshot')

    args = parser.parse_args()
    args_dict = args.__dict__
    return args_dict


def augment_dataset(dataset):
    dataset_tuple = dataset.get_datasets()
    feat1, feat2, labels = dataset_tuple
    new_feat1 = np.concatenate((feat1, feat2), axis=0)
    new_feat2 = np.concatenate((feat2, feat1), axis=0)
    new_labels = np.concatenate((labels, labels), axis=0)
    new_dataset = NumpyTupleDataset(new_feat1, new_feat2, new_labels)
    return new_dataset


def main():
    # Parse the arguments.
    args = parse_arguments()
    if args['label']:
        labels = args['label']
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        raise ValueError('No target label was specified.')

    # Dataset preparation. Postprocessing is required for the regression task.
    def postprocess_label(label_list):
        label_arr = np.asarray(label_list, dtype=np.int32)
        return label_arr

    # Apply a preprocessor to the dataset.
    logging.info('Preprocess train dataset and valid dataset...')
    # use `ggnn` for the time being
    preprocessor = preprocess_method_dict['ggnn']()
    # parser = CSVFileParserForPair(preprocessor, postprocess_label=postprocess_label,
    #                               labels=labels, smiles_cols=['smiles_1', 'smiles_2'])
    if args['feature'] == 'molenc':
        parser = MolAutoencoderParserForPair(preprocessor, postprocess_label=postprocess_label,
                                             labels=labels, smiles_cols=['smiles_1', 'smiles_2'])
    if args['feature'] == 'ssp':
        parser = SSPParserForPair(preprocessor, postprocess_label=postprocess_label,
                                  labels=labels, smiles_cols=['smiles_1', 'smiles_2'])
    else:
        parser = Mol2VecParserForPair(preprocessor, postprocess_label=postprocess_label,
                                      labels=labels, smiles_cols=['smiles_1', 'smiles_2'])
    train = parser.parse(args['train_datafile'])['dataset']
    valid = parser.parse(args['valid_datafile'])['dataset']

    if args['augment']:
        logging.info('Utilizing data augmentation in train set')
        train = augment_dataset(train)

    num_train = train.get_datasets()[0].shape[0]
    num_valid = valid.get_datasets()[0].shape[0]
    logging.info('Train/test split: {}/{}'.format(num_train, num_valid))

    if len(args['net_hidden_dims']):
        net_hidden_dims = tuple([int(net_hidden_dim) for net_hidden_dim in args['net_hidden_dims'].split(',')])
    else:
        net_hidden_dims = ()

    predictor = set_up_predictor(fp_out_dim=args['fp_out_dim'], net_hidden_dims=net_hidden_dims, class_num=class_num,
                                 sim_method=args['sim_method'], symmetric=args['symmetric'])

    train_iter = SerialIterator(train, args['batchsize'])
    test_iter = SerialIterator(valid, args['batchsize'],
                              repeat=False, shuffle=False)

    metrics_fun = {'accuracy': F.binary_accuracy}
    classifier = Classifier(predictor, lossfun=F.sigmoid_cross_entropy,
                            metrics_fun=metrics_fun, device=args['gpu'])

    # Set up the optimizer.
    optimizer = optimizers.Adam(alpha=args['learning_rate'], weight_decay_rate=args['weight_decay_rate'])
    # optimizer = optimizers.Adam()
    # optimizer = optimizers.SGD(lr=args.learning_rate)
    optimizer.setup(classifier)
    # add regularization
    if args['max_norm'] > 0:
        optimizer.add_hook(chainer.optimizer.GradientClipping(threshold=args['max_norm']))
    if args['l2_rate'] > 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args['l2_rate']))
    if args['l1_rate'] > 0:
        optimizer.add_hook(chainer.optimizer.Lasso(rate=args['l1_rate']))

    updater = training.StandardUpdater(train_iter, optimizer, device=args['gpu'],
                                       converter=concat_mols)

    # Set up the trainer.
    logging.info('Training...')
    # add stop_trigger parameter
    early_stop = triggers.EarlyStoppingTrigger(monitor='validation/main/loss', patients=10, max_trigger=(500, 'epoch'))
    out = 'output' + '/' + args['out']
    trainer = training.Trainer(updater, stop_trigger=early_stop, out=out)

    trainer.extend(E.Evaluator(test_iter, classifier,
                               device=args['gpu'], converter=concat_mols))

    train_eval_iter = SerialIterator(train, args['batchsize'],
                                       repeat=False, shuffle=False)

    trainer.extend(AccuracyEvaluator(
        train_eval_iter, classifier, eval_func=predictor,
        device=args['gpu'], converter=concat_mols, name='train_acc',
        pos_labels=1, ignore_labels=-1, raise_value_error=False))
    # extension name='validation' is already used by `Evaluator`,
    # instead extension name `val` is used.
    trainer.extend(AccuracyEvaluator(
        test_iter, classifier, eval_func=predictor,
        device=args['gpu'], converter=concat_mols, name='val_acc',
        pos_labels=1, ignore_labels=-1))

    trainer.extend(ROCAUCEvaluator(
        train_eval_iter, classifier, eval_func=predictor,
        device=args['gpu'], converter=concat_mols, name='train_roc',
        pos_labels=1, ignore_labels=-1, raise_value_error=False))
    # extension name='validation' is already used by `Evaluator`,
    # instead extension name `val` is used.
    trainer.extend(ROCAUCEvaluator(
        test_iter, classifier, eval_func=predictor,
        device=args['gpu'], converter=concat_mols, name='val_roc',
        pos_labels=1, ignore_labels=-1))

    trainer.extend(PRCAUCEvaluator(
        train_eval_iter, classifier, eval_func=predictor,
        device=args['gpu'], converter=concat_mols, name='train_prc',
        pos_labels=1, ignore_labels=-1, raise_value_error=False))
    # extension name='validation' is already used by `Evaluator`,
    # instead extension name `val` is used.
    trainer.extend(PRCAUCEvaluator(
        test_iter, classifier, eval_func=predictor,
        device=args['gpu'], converter=concat_mols, name='val_prc',
        pos_labels=1, ignore_labels=-1))

    trainer.extend(F1Evaluator(
        train_eval_iter, classifier, eval_func=predictor,
        device=args['gpu'], converter=concat_mols, name='train_f',
        pos_labels=1, ignore_labels=-1, raise_value_error=False))
    # extension name='validation' is already used by `Evaluator`,
    # instead extension name `val` is used.
    trainer.extend(F1Evaluator(
        test_iter, classifier, eval_func=predictor,
        device=args['gpu'], converter=concat_mols, name='val_f',
        pos_labels=1, ignore_labels=-1))

    # apply shift strategy to learning rate every 10 epochs
    # trainer.extend(E.ExponentialShift('alpha', args.exp_shift_rate), trigger=(10, 'epoch'))
    if args['exp_shift_strategy']== 1:
        trainer.extend(E.ExponentialShift('alpha', args['exp_shift_rate']),
                       trigger=triggers.ManualScheduleTrigger([10, 20, 30, 40, 50, 60], 'epoch'))
    elif args['exp_shift_strategy'] == 2:
        trainer.extend(E.ExponentialShift('alpha', args['exp_shift_rate']),
                       trigger=triggers.ManualScheduleTrigger([5, 10, 15, 20, 25, 30], 'epoch'))
    elif args['exp_shift_strategy'] == 3:
        trainer.extend(E.ExponentialShift('alpha', args['exp_shift_rate']),
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

    if args['resume']:
        resume_path = os.path.join(out, args['resume'])
        logging.info('Resume training according to snapshot in {}'.format(resume_path))
        chainer.serializers.load_npz(resume_path, trainer)

    trainer.run()

    # Save the regressor's parameters.
    model_path = os.path.join(out, args['model_filename'])
    logging.info('Saving the trained models to {}...'.format(model_path))
    classifier.save_pickle(model_path, protocol=args['protocol'])


if __name__ == '__main__':
    main()