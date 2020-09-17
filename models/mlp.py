#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/2/2019 4:22 PM
# @Author  : chinshin
# @FileName: mlp.py


import chainer
from chainer.functions import relu
from chainer import links
from chainer import functions
import numpy

from chainer.backends import cuda
from chainer.functions.connection import bilinear
from chainer import initializers
from chainer import link
from chainer import variable

class MLP(chainer.Chain):

    """Basic implementation for MLP

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_layers (int): number of layers
        activation (chainer.functions): activation function
    """

    def __init__(self, out_dim, hidden_dims=(32, 16), activation=relu):
        super(MLP, self).__init__()
        layers = [links.Linear(None, hidden_dim) for hidden_dim in hidden_dims]
        with self.init_scope():
            self.layers = chainer.ChainList(*layers)
            self.l_out = links.Linear(None, out_dim)
        self.activation = activation

    def __call__(self, x):
        h = x
        for l in self.layers:
            h = self.activation(l(h))
        h = self.l_out(h)
        return h


class NTN(chainer.Chain):
    def __init__(self, left_dim, right_dim, out_dim, ntn_out_dim=8, hidden_dims=(16, ), activation=relu):
        super(NTN, self).__init__()

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


class HolE(chainer.Chain):
    def __init__(self, out_dim, hidden_dims=(32, 16), activation=relu):
        super(HolE, self).__init__()
        layers = [links.Linear(None, hidden_dim) for hidden_dim in hidden_dims]
        with self.init_scope():
            self.layers = chainer.ChainList(*layers)
            self.l_out = links.Linear(None, out_dim)
        self.activation = activation

    def __call__(self, left_x, right_x):
        h = self.circular_correlation(left_x, right_x)
        for l in self.layers:
            h = self.activation(l(h))
        h = self.l_out(h)
        return h

    def circular_correlation(self, left_x, right_x):
        """
        Computes the circular correlation of two vectors a and b via their fast fourier transforms
        In python code, ifft(np.conj(fft(a)) * fft(b)).real
        :param left_x:
        :param right_x:

        (a - j * b) * (c + j * d) = (ac + bd) + j * (ad - bc)
        :return:
        """
        left_x_real = left_x
        left_x_imag = chainer.as_variable(self.xp.zeros_like(left_x_real, dtype=self.xp.float32))
        left_x_fft_real, left_x_fft_imag = functions.fft((left_x_real, left_x_imag))

        right_x_real = right_x
        right_x_imag = chainer.as_variable(self.xp.zeros_like(right_x_real, dtype=self.xp.float32))
        right_x_fft_real, right_x_fft_imag = functions.fft((right_x_real, right_x_imag))

        prod_fft_real = left_x_fft_real * right_x_fft_real + left_x_fft_imag * right_x_fft_imag
        prod_fft_imag = left_x_fft_real * right_x_fft_imag - left_x_fft_imag * right_x_fft_real

        ifft_real, _ = functions.ifft((prod_fft_real, prod_fft_imag))
        return ifft_real


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



