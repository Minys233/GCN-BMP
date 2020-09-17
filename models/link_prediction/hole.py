#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/14/2019 4:03 PM
# @Author  : chinshin
# @FileName: hole.py
import chainer
from chainer import links
from chainer import functions
from chainer.functions import relu


class HOLE(chainer.Chain):
    def __init__(self, out_dim, hidden_dims=(32, 16), activation=relu):
        super(HOLE, self).__init__()
        hidden_layers = [links.Linear(None, hidden_dim) for hidden_dim in hidden_dims]
        with self.init_scope():
            self.hidden_layers = chainer.ChainList(*hidden_layers)
            self.l_out = links.Linear(None, out_dim)
        self.activation = activation

    def __call__(self, left_x, right_x):
        h = self.circular_correlation(left_x, right_x)
        for l in self.hidden_layers:
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


class HolE(chainer.Chain):
    def __init__(self, out_dim, hidden_dims=(32, 16), activation=relu):
        super(HolE, self).__init__()
        hidden_layers = [links.Linear(None, hidden_dim) for hidden_dim in hidden_dims]
        with self.init_scope():
            self.hidden_layers = chainer.ChainList(*hidden_layers)
            self.l_out = links.Linear(None, out_dim)
        self.activation = activation

    def __call__(self, left_x, right_x):
        h = self.circular_correlation(left_x, right_x)
        for l in self.hidden_layers:
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