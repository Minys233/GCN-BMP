#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/20/2019 5:19 PM
# @Author  : chinshin
# @FileName: mol2vec_based_model.py
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import random
import logging

import chainer
from chainer.backends import cuda
from chainer import functions as F
from chainer_chemistry.models import MLP, NTN, HolE, DistMult

logging.basicConfig(format='%(asctime)s: %(filename)s: %(funcName)s: %(lineno)d: %(message)s', level=logging.INFO)
global_seed = 2018
random.seed(global_seed)


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
