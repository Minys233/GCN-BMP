#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/12/2019 2:19 PM
# @Author  : chinshin
# @FileName: eval_coattention.py

from __future__ import unicode_literals
from __future__ import print_function
import os
import sys
import copy
import json
import argparse
import numpy
import pandas as pd
import matplotlib

matplotlib.use('AGG')
import chainer
from chainer.backends import cuda
from chainer import functions as F
from chainer.dataset import iterator as iterator_module
from chainer import link
from sklearn import metrics
from logging import getLogger
from chainer import cuda
from chainer.dataset import convert
from chainer_chemistry.dataset.converters import concat_mols
from os.path import dirname, abspath

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_PATH)
from setting import *
from utils import *
from parsers import CSVFileParserForPair
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict

logging.basicConfig(format='%(asctime)s: %(filename)s: %(funcName)s: %(lineno)d: %(message)s', level=logging.INFO)


class GraphConvPredictorForPair(chainer.Chain):
    def __init__(self, graph_conv, attn=None, mlp=None, symmetric=None):
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
            if isinstance(attn, chainer.Link):
                self.attn = attn
        if not isinstance(mlp, chainer.Link):
            self.mlp = mlp
        if not isinstance(attn, chainer.Link):
            self.attn = attn
        self.symmetric = symmetric

    def __call__(self, atoms_1, adjs_1, atoms_2, adjs_2):
        if self.xp == cuda.cupy:
            atoms_1 = cuda.to_gpu(atoms_1)
            adjs_1 = cuda.to_gpu(adjs_1)
            atoms_2 = cuda.to_gpu(atoms_2)
            adjs_2 = cuda.to_gpu(adjs_2)

        g1 = self.graph_conv(atoms_1, adjs_1)
        atoms_1 = self.graph_conv.get_atom_array()
        g2 = self.graph_conv(atoms_2, adjs_2)
        atoms_2 = self.graph_conv.get_atom_array()

        g1, g2 = self.attn(atoms_1, g1, atoms_2, g2)

        if self.mlp.__class__.__name__ == 'MLP':
            g = F.concat((g1, g2), axis=-1)
            g = self.mlp(g)
            return g
        elif self.mlp.__class__.__name__ == 'NTN':
            g = self.mlp(g1, g2)
            return g
        elif self.mlp.__class__.__name__ == 'SymMLP':
            g = self.mlp(g1, g2)
            return g
        elif self.mlp.__class__.__name__ == 'HolE':
            g = self.mlp(g1, g2)
            return g
        elif self.mlp.__class__.__name__ == 'DistMult':
            g = self.mlp(g1, g2)
            return g
        elif self.mlp.__class__.__name__ == 'Cosine':
            g = self.mlp(g1, g2)
            return g
        else:
            ValueError('[ERROR] No methods for similarity prediction')

    def predict(self, atoms_1, adjs_1, atoms_2, adjs_2):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            if self.xp == cuda.cupy:
                atoms_1 = cuda.to_gpu(atoms_1)
                adjs_1 = cuda.to_gpu(adjs_1)
                atoms_2 = cuda.to_gpu(atoms_2)
                adjs_2 = cuda.to_gpu(adjs_2)

            g1 = self.graph_conv(atoms_1, adjs_1)
            atoms_1 = self.graph_conv.get_atom_array()
            g2 = self.graph_conv(atoms_2, adjs_2)
            atoms_2 = self.graph_conv.get_atom_array()

            g1, g2 = self.attn(atoms_1, g1, atoms_2, g2)

            sim_methods = ['MLP', 'NTN', 'SymMLP', 'HolE', 'DistMult', 'Cosine']
            if self.mlp.__class__.__name__ == 'MLP':
                h = F.concat((g1, g2), axis=-1)
                h = self.mlp(h)
                return h, (g1, g2)
            elif self.mlp.__class__.__name__ in sim_methods:
                h = self.mlp(g1, g2)
                return h, (g1, g2)
            else:
                ValueError('[ERROR] No methods for similarity prediction')


def _to_list(a):
    if isinstance(a, (int, float)):
        return [a, ]
    else:
        # expected to be list or some iterable class
        return a


def _get_1d_numpy_array(v):
    if isinstance(v, chainer.Variable):
        v = v.data
    return cuda.to_cpu(v).ravel()


def _get_numpy_array(v):
    if isinstance(v, chainer.Variable):
        v = v.data
    return cuda.to_cpu(v)


class MyEvaluator(object):

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 mediate_func=None,
                 device=None, eval_hook=None, eval_func=None, name=None,
                 pos_labels=1, ignore_labels=None, raise_value_error=None,
                 logger=None):
        super(MyEvaluator, self).__init__()

        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        if isinstance(target, link.Link):
            target = {'main': target}
        self._targets = target

        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self.eval_func = eval_func
        self.mediate_func = mediate_func

        self.pos_labels = _to_list(pos_labels)
        self.ignore_labels = _to_list(ignore_labels)
        self.raise_value_error = raise_value_error

        self.name = name
        self.logger = logger or getLogger()

        self.metric_dict = {'roc_auc': self.roc_auc_score,
                            'prc_auc': self.prc_auc_score,
                            'accuracy': self.accuracy_score,
                            'f1': self.f1_score}

        self.y_total, self.t_total, self.e1_total, self.e2_total = self.evaluate()

    def compuate(self, metric):

        try:
            return self.metric_dict[metric](self.y_total, self.t_total)
        except KeyError:
            print('No such metric.')

    def generate_representations(self):
        return self.e1_total, self.e2_total

    def generate_y_and_t(self):
        return self.y_total, self.t_total

    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        y_total = []
        t_total = []
        e1_total = []
        e2_total = []
        for batch in it:
            in_arrays = self.converter(batch, self.device)
            with chainer.no_backprop_mode(), chainer.using_config('train',
                                                                  False):
                y, (e1, e2) = eval_func(*in_arrays[:-1])
                # e1_list, e2_list = mediate_func(*in_arrays[:-1])

            t = in_arrays[-1]
            y_data = _get_1d_numpy_array(y)
            t_data = _get_1d_numpy_array(t)
            e1_data = _get_numpy_array(e1)
            e2_data = _get_numpy_array(e2)
            y_total.append(y_data)
            t_total.append(t_data)
            e1_total.append(e1_data)
            e2_total.append(e2_data)

        y_total = numpy.concatenate(y_total).ravel()
        t_total = numpy.concatenate(t_total).ravel()
        e1_total = numpy.concatenate(e1_total)
        e2_total = numpy.concatenate(e2_total)

        return y_total, t_total, e1_total, e2_total

    def roc_auc_score(self, y_total, t_total):
        # --- ignore labels if specified ---
        if self.ignore_labels:
            valid_ind = numpy.in1d(t_total, self.ignore_labels, invert=True)
            y_total = y_total[valid_ind]
            t_total = t_total[valid_ind]

        # --- set positive labels to 1, negative labels to 0 ---
        pos_indices = numpy.in1d(t_total, self.pos_labels)
        t_total = numpy.where(pos_indices, 1, 0)
        try:
            roc_auc = metrics.roc_auc_score(t_total, y_total)
        except ValueError as e:
            # When only one class present in `y_true`, `ValueError` is raised.
            # ROC AUC score is not defined in that case.
            if self.raise_value_error:
                raise e
            else:
                self.logger.warning(
                    'ValueError detected during roc_auc_score calculation. {}'
                        .format(e.args))
                roc_auc = numpy.nan
        return roc_auc

    def prc_auc_score(self, y_total, t_total):
        # --- ignore labels if specified ---
        if self.ignore_labels:
            valid_ind = numpy.in1d(t_total, self.ignore_labels, invert=True)
            y_total = y_total[valid_ind]
            t_total = t_total[valid_ind]

        # --- set positive labels to 1, negative labels to 0 ---
        pos_indices = numpy.in1d(t_total, self.pos_labels)
        t_total = numpy.where(pos_indices, 1, 0)

        if len(numpy.unique(t_total)) != 2:
            if self.raise_value_error:
                raise ValueError("Only one class present in y_true. PRC AUC "
                                 "score is not defined in that case.")
            else:
                return numpy.nan

        precision, recall, _ = metrics.precision_recall_curve(t_total, y_total)
        prc_auc = metrics.auc(recall, precision)
        return prc_auc

    def f1_score(self, y_total, t_total):
        # --- ignore labels if specified ---
        if self.ignore_labels:
            valid_ind = numpy.in1d(t_total, self.ignore_labels, invert=True)
            y_total = y_total[valid_ind]
            t_total = t_total[valid_ind]

        # --- set positive labels to 1, negative labels to 0 ---
        pos_indices = numpy.in1d(t_total, self.pos_labels)
        t_total = numpy.where(pos_indices, 1, 0)

        if len(numpy.unique(t_total)) != 2:
            if self.raise_value_error:
                raise ValueError("Only one class present in y_true. PRC AUC "
                                 "score is not defined in that case.")
            else:
                return numpy.nan

        y_total = numpy.where(y_total > 0, 1, 0)
        y_total = numpy.asarray(y_total, dtype=numpy.int32)
        f1 = metrics.f1_score(y_true=t_total, y_pred=y_total)
        return f1

    def accuracy_score(self, y_total, t_total):
        # --- ignore labels if specified ---
        if self.ignore_labels:
            valid_ind = numpy.in1d(t_total, self.ignore_labels, invert=True)
            y_total = y_total[valid_ind]
            t_total = t_total[valid_ind]

        # --- set positive labels to 1, negative labels to 0 ---
        pos_indices = numpy.in1d(t_total, self.pos_labels)
        t_total = numpy.where(pos_indices, 1, 0)

        if len(numpy.unique(t_total)) != 2:
            if self.raise_value_error:
                raise ValueError("Only one class present in y_true. PRC AUC "
                                 "score is not defined in that case.")
            else:
                return numpy.nan

        y_total = numpy.where(y_total > 0, 1, 0)
        y_total = numpy.asarray(y_total, dtype=numpy.int32)
        accuracy = metrics.accuracy_score(y_true=t_total, y_pred=y_total)
        return accuracy


def parse_arguments():
    parser = argparse.ArgumentParser(description='GGNN-HOLE Evaluation')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--test-datafile', type=str, default='ddi_test.csv',
                        help='The file to be evaluated.')
    parser.add_argument('--label', '-l', nargs='+',
                        default=['label', ],
                        help='target label for classification')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='path to save the computed models to')
    parser.add_argument('--model-filename', type=str, default='classifier.pkl',
                        help='saved models filename')
    parser.add_argument('--snapshot', '-s',
                        default=None, help='The path to a saved snapshot (NPZ)')
    parser.add_argument('--model-name', type=str, default='')
    parser.add_argument('--generate-drug-list', type=str, default='True',
                        help='whether to generate drug list with the entity representation')
    return parser.parse_args()


def sigmoid(x):
    return 1./(1+np.exp(-x))


def add_representations(source_filepath, dst_filepath, e1_list, e2_list, generate_drug_list=True):
    e1_list = check_list(e1_list)
    e1_list = [float_list_2_str(e1) for e1 in e1_list]
    e2_list = check_list(e2_list)
    e2_list = [float_list_2_str(e2) for e2 in e2_list]

    content = pd.read_csv(source_filepath)
    new_content = copy.deepcopy(content)
    new_content.insert(len(new_content.columns) - 1, 'e1', e1_list)
    new_content.insert(len(new_content.columns) - 1, 'e2', e2_list)

    new_content.to_csv(dst_filepath)
    print('Entity Representations have been generated.')

    if generate_drug_list:
        drug_and_representation = dict(zip(new_content['cid_1'], e1_list))
        drug_and_representation.update(dict(zip(new_content['cid_2'], e2_list)))

        output_filepath = os.path.join(os.path.dirname(dst_filepath), 'drug_list_with_representation.csv')
        output_df = pd.DataFrame({'cid': drug_and_representation.keys(), 'entity': drug_and_representation.values()})
        output_df.to_csv(output_filepath)
        print('Drug List with Representations have been generated.')


def add_reprensentations_and_y(source_filepath, dst_filepath, e1_list, e2_list, y_list):

    e1_list = check_list(e1_list)
    e1_list = [float_list_2_str(e1) for e1 in e1_list]
    e2_list = check_list(e2_list)
    e2_list = [float_list_2_str(e2) for e2 in e2_list]
    y_list = check_list(y_list)
    y_list = [sigmoid(y) for y in y_list]

    content = pd.read_csv(source_filepath)
    new_content = copy.deepcopy(content)
    new_content.insert(len(new_content.columns) - 1, 'e1', e1_list)
    new_content.insert(len(new_content.columns) - 1, 'e2', e2_list)
    new_content.insert(len(new_content.columns), 'y', y_list)
    new_content.to_csv(dst_filepath)

    print('Entity Representations have been generated.')


def main():
    args = parse_arguments()
    generate_drug_list = True if args.generate_drug_list == 'True' else False

    if args.label:
        labels = args.label
        # class_num = len(labels) if isinstance(labels, list) else 1
    else:
        raise ValueError('No target label was specified.')

    # Dataset preparation. Postprocessing is required for the regression task.
    def postprocess_label(label_list):
        label_arr = np.asarray(label_list, dtype=np.int32)
        return label_arr

    # Apply a preprocessor to the dataset.
    logging.info('Preprocess test dataset...')
    preprocessor = preprocess_method_dict['ggnn']()
    parser = CSVFileParserForPair(preprocessor, postprocess_label=postprocess_label,
                                  labels=labels, smiles_cols=['smiles_1', 'smiles_2'])
    test_dict = parser.parse(args.test_datafile, return_smiles_pair_original=True)
    test = test_dict['dataset']
    # test_smiles_pairs = test_dict['smiles_pair_original']
    from chainer.iterators import SerialIterator
    test_iter = SerialIterator(test, 32, repeat=False, shuffle=False)

    out = 'output' + '/' + args.out
    model_path = os.path.join(out, args.model_filename)
    # `load_pickle` is static method, call from Class to get an instance
    print('model_path: {}'.format(model_path))
    from chainer_chemistry.models.prediction import Classifier
    model = Classifier.load_pickle(model_path, args.gpu)

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    snapshot_path = os.path.join(out, args.snapshot)
    try:
        chainer.serializers.load_npz(snapshot_path, model)
    except KeyError as e:
        print(e)

    evaluator = MyEvaluator(test_iter, model, converter=concat_mols,
                            device=args.gpu, eval_func=model.predictor.predict,
                            # mediate_func=models.predictor.mediate_output,
                            name='test', ignore_labels=-1)
    e1_total, e2_total = evaluator.generate_representations()
    y_total, t_total = evaluator.generate_y_and_t()

    # print('test_datafile: {}'.format(args.test_datafile))
    test_filename = os.path.basename(args.test_datafile).split('.')[0]
    # print('test_filename: {}'.format(test_filename))
    dst_repre_filename = test_filename + '_e' + '.csv'
    dst_repre_filepath = os.path.join(out, dst_repre_filename)
    add_representations(args.test_datafile, dst_repre_filepath, e1_total, e2_total,
                        generate_drug_list=generate_drug_list)

    dst_filename = test_filename + '_e_y' + '.csv'
    dst_filepath = os.path.join(out, dst_filename)
    add_reprensentations_and_y(args.test_datafile, dst_filepath, e1_total, e2_total, y_total)


    perf_dict = dict()
    for metric in ['roc_auc', 'prc_auc', 'accuracy', 'f1']:
        result = evaluator.compuate(metric=metric)
        perf_dict[metric] = result
        print('{}: {}'.format(metric, result))
    with open(os.path.join(ROOT_PATH, 'eval_result.json'), 'w') as f:
        json.dump(perf_dict, f)


if __name__ == '__main__':
    main()
