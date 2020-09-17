#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/14/2019 4:04 PM
# @Author  : chinshin
# @FileName: dist_mult.py
import numpy
import chainer
from chainer import link
from chainer import links
from chainer import variable
from chainer import initializers
from chainer.functions import relu
from chainer.backends import cuda
from chainer.functions.connection import bilinear


class BilinearDiag(link.Link):

    def __init__(self, left_size, right_size, out_size,
                 initialW=None):
        super(BilinearDiag, self).__init__()
        self.in_sizes = (left_size, right_size)
        assert left_size == right_size

        with self.init_scope():
            shape = (out_size, left_size)
            if isinstance(initialW, (numpy.ndarray, cuda.ndarray)):
                assert initialW.shape == shape
            self.W = variable.Parameter(
                initializers._get_initializer(initialW), shape)


    def __call__(self, e1, e2):
        """Applies the bilinear function to inputs and the internal parameters.

        Args:
            e1 (~chainer.Variable): Left input.
            e2 (~chainer.Variable): Right input.

        Returns:
            ~chainer.Variable: Output variable.

        """
        W_mat = self.diagonal_matrix()
        return bilinear.bilinear(e1, e2, W_mat)

    def diagonal_matrix(self):
        # (out_size, left_size)
        W_data = self.W.data
        # vec: (left_size, )
        # W_mat_data: (out_size, left_size, right_size)
        W_mat_data = self.xp.array([self.xp.diag(vec) for vec in W_data], dtype=self.xp.float32)
        # W_mat_data: (left_size, right_size, out_size)
        W_mat_data = self.xp.transpose(W_mat_data, axes=(1, 2, 0))
        return chainer.as_variable(W_mat_data)

    def zero_grads(self):
        # Left for backward compatibility
        self.zerograds()




class DistMult(chainer.Chain):

    def __init__(self, left_dim, right_dim, out_dim, dm_out_dim=8, hidden_dims=(16, ), activation=relu):
        super(DistMult, self).__init__()
        dm_layer = BilinearDiag(left_size=left_dim, right_size=right_dim, out_size=dm_out_dim)
        mlp_layers = [links.Linear(None, hidden_dim) for hidden_dim in hidden_dims]
        with self.init_scope():
            self.dm_layer = dm_layer
            self.mlp_layers = chainer.ChainList(*mlp_layers)
            self.l_out = links.Linear(None, out_dim)
        self.activation = activation

    def __call__(self, left_x, right_x):
        dm_output = self.dm_layer(left_x, right_x)
        h = dm_output
        for l in self.mlp_layers:
            h = self.activation(l(h))
        h = self.l_out(h)
        return h
