#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/5/2019 6:43 PM
# @Author  : chinshin
# @FileName: parallel_coattention.py
import chainer
from chainer import functions as F
from chainer import functions
from chainer import links
from chainer_chemistry.links import GraphLinear

class ParallelCoattention(chainer.Chain):
    def __init__(self, hidden_dim, out_dim, head, activation=functions.tanh, weight_tying=True):
        """
        :param hidden_dim: dimension of atom representation
        :param out_dim: dimension of molecular representation
        :param head: number of heads in attention mechanism
        """
        super(ParallelCoattention, self).__init__()
        n_entities = 1 if weight_tying else 2
        with self.init_scope():
            self.energy_layers = chainer.ChainList(
                *[links.Bilinear(hidden_dim, out_dim, head) for _ in range(n_entities)]
            )

            self.j_layer = GraphLinear(hidden_dim, out_dim)

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.head = head
        self.activation = activation
        self.weight_tying = weight_tying

    def __call__(self, atoms_1, g_1, atoms_2, g_2):
        """
        :param atoms_1: atomic representation of molecule 1, with shape of (mb, N_1, hidden_dim)
        :param g_1: molecular representation of molecule 1, with shape of (mb, out_dim)
        :param atoms_2: atomic representation of molecule 2, with shape of (mb, N_2, hidden_dim)
        :param g_2: molecular representation of molecule 2, with shape of (mb, out_dim)
        :return:
        """
        # compute attention based on molecular representation of entity 2
        attn_1 = self.compute_attention(query=g_2, key=atoms_1, focus=1)
        # attn_1: (mb, N_1, out_dim)
        attn_1 = F.tile(attn_1, reps=(1, 1, self.out_dim))
        # (mb, N_1, out_dim) * (mb, N_1, out_dim) - > (mb, N_1, out_dim)
        z_1 = attn_1 * self.j_layer(atoms_1)
        # compact_1: (mb, out_dim)
        compact_1 = F.sum(z_1, axis=1)

        # compute attention based on attended molecular representation of entity 1
        # attn_2: (mb, N_2, 1)
        attn_2 = self.compute_attention(query=g_1, key=atoms_2, focus=2)
        # attn_2: (mb, N_2, out_dim)
        attn_2 = F.tile(attn_2, reps=(1, 1, self.out_dim))
        # z_2: (mb, N_2, out_dim) * (mb, N_2, out_dim) -> (mb, N_2, out_dim)
        z_2 = attn_2 * self.j_layer(atoms_2)
        # compact_2: (mb, out_dim)
        compact_2 = F.sum(z_2, axis=1)

        return compact_1, compact_2

    def compute_attention(self, query, key, focus):
        """
        :param query: with shape of (mb, out_dim)
        :param key: with shape of (mb, N, hidden_dim)
        :param focus: indicate the focused molecule
        :return: attn: attention weights (mb, N, 1)
        """
        entity_index = 0 if self.weight_tying else focus - 1
        energy_layer = self.energy_layers[entity_index]
        mb, N, hidden_dim = key.shape
        # query: (mb, 1, out_dim)
        query = F.expand_dims(query, axis=1)
        # query: (mb, N, out_dim)
        query = F.tile(query, reps=(1, N, 1))
        query = F.reshape(query, shape=(mb * N, self.out_dim))
        key = F.reshape(key, shape=(mb * N, self.hidden_dim))
        energy = self.activation(energy_layer(key, query))

        # energy = self.scorers[entity_index](query, key)
        energy = F.reshape(energy, shape=(mb, N, self.head))

        return energy


class CircularParallelCoattention(chainer.Chain):
    def __init__(self, hidden_dim, out_dim, activation=functions.tanh, weight_tying=True):
        """
        :param hidden_dim: dimension of atom representation
        :param out_dim: dimension of molecular representation
        """
        super(CircularParallelCoattention, self).__init__()
        # n_entities = 1 if weight_tying else 2
        with self.init_scope():
            # self.energy_layers = chainer.ChainList(
            #     *[links.Bilinear(hidden_dim, out_dim, head) for _ in range(n_entities)]
            # )

            self.j_layer = GraphLinear(hidden_dim, out_dim)

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.head = out_dim
        self.activation = activation
        self.weight_tying = weight_tying

    def __call__(self, atoms_1, g_1, atoms_2, g_2):
        """
        :param atoms_1: atomic representation of molecule 1, with shape of (mb, N_1, hidden_dim)
        :param g_1: molecular representation of molecule 1, with shape of (mb, out_dim)
        :param atoms_2: atomic representation of molecule 2, with shape of (mb, N_2, hidden_dim)
        :param g_2: molecular representation of molecule 2, with shape of (mb, out_dim)
        :return:
        """
        # compute attention based on molecular representation of entity 2
        # atoms_1: (mb, N_1, out_dim)
        atoms_1 = self.j_layer(atoms_1)
        # attn_1: (mb, N_1, out_dim)
        attn_1 = self.compute_attention(query=g_2, key=atoms_1)
        # (mb, N_1, out_dim) * (mb, N_1, out_dim) - > (mb, N_1, out_dim)
        z_1 = attn_1 * atoms_1
        # compact_1: (mb, out_dim)
        compact_1 = F.sum(z_1, axis=1)

        # compute attention based on attended molecular representation of entity 1
        # attn_2: (mb, N_2, 1)
        # atoms_2: (mb, N_2, out_dim)
        atoms_2 = self.j_layer(atoms_2)
        # attn_2: (mb, N_2, out_dim)
        attn_2 = self.compute_attention(query=g_1, key=atoms_2)
        # z_2: (mb, N_2, out_dim) * (mb, N_2, out_dim) -> (mb, N_2, out_dim)
        z_2 = attn_2 * atoms_2
        # compact_2: (mb, out_dim)
        compact_2 = F.sum(z_2, axis=1)

        return compact_1, compact_2

    def compute_attention(self, query, key):
        """
        :param query: with shape of (mb, out_dim)
        :param key: with shape of (mb, N, out_dim)
        :return: attn: attention weights (mb, N, out_dim)
        """
        # entity_index = 0 if self.weight_tying else focus - 1
        # energy_layer = self.energy_layers[entity_index]
        mb, N, _ = key.shape
        # query: (mb, 1, out_dim)
        query = F.expand_dims(query, axis=1)
        # query: (mb, N, out_dim)
        query = F.tile(query, reps=(1, N, 1))
        query = F.reshape(query, shape=(mb * N, self.out_dim))
        key = F.reshape(key, shape=(mb * N, self.out_dim))
        # energy = self.activation(energy_layer(key, query))

        energy = self.activation(self.circular_correlation(key, query))

        energy = F.reshape(energy, shape=(mb, N, self.head))

        return energy

    def circular_correlation(self, left_x, right_x):
        """
        Computes the circular correlation of two vectors a and b via their fast fourier transforms
        In python code, ifft(np.conj(fft(a)) * fft(b)).real
        :param left_x: ()
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



