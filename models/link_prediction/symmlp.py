#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/14/2019 4:03 PM
# @Author  : chinshin
# @FileName: symmlp.py

import chainer
from chainer import links


class SymMLP(chainer.Chain):
    def __init__(self, out_dim, hidden_dims=(32, 16), activation=relu):
        super(SymMLP, self).__init__()
        layers = [links.Linear(None, hidden_dim) for hidden_dim in hidden_dims]
        with self.init_scope():
            self.layers = chainer.ChainList(*layers)
            self.l_out = links.Linear(None, out_dim)
        self.activation = activation

    def __call__(self, left_x, right_x):
        h = self.xp.concatenate((left_x + right_x, left_x * right_x), axis=1)
        for l in self.layers:
            h = self.activation(l(h))
        h = self.l_out(h)
        return h