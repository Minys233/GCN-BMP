#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/14/2019 4:03 PM
# @Author  : chinshin
# @FileName: bilinear.py
import chainer
from chainer import links
from chainer.functions import relu


class Bilinear(chainer.Chain):
    def __init__(self, left_dim, right_dim, out_dim, ntn_out_dim=8, hidden_dims=(16,), activation=relu):
        super(Bilinear, self).__init__()

        ntn_layer = links.Bilinear(left_size=left_dim, right_size=right_dim, out_size=ntn_out_dim)
        mlp_layers = [links.Linear(in_size=None, out_size=hidden_dim) for hidden_dim in hidden_dims]

        with self.init_scope():
            self.ntn_layer = ntn_layer
            self.mlp_layers = chainer.ChainList(*mlp_layers)
            self.l_out = links.Linear(in_size=None, out_size=out_dim)

        self.left_dim = left_dim
        self.right_dim = right_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.activation = activation

    def __call__(self, left_x, right_x):
        ntn_output = self.ntn_layer(left_x, right_x)
        h = ntn_output
        for layer in self.mlp_layers:
            h = self.activation(layer(h))
        h = self.l_out(h)
        return h
