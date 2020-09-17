#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/12/2019 7:41 PM
# @Author  : chinshin
# @FileName: PoolingFineCoattention.py

import chainer
from chainer import functions as F
from chainer import functions
from chainer import links
from chainer_chemistry.links import GraphLinear

class PoolingFineCoattention(chainer.Chain):
    """
    TODO
    """
    def __init__(self, hidden_dim, out_dim, activation=functions.tanh):
        """
        :param hidden_dim: dimension of atom representation
        :param out_dim: dimension of molecular representation
        :param head: number of heads in attention mechanism
        """
        super(PoolingFineCoattention, self).__init__()
        with self.init_scope():
            self.energy_layer = links.Bilinear(hidden_dim, hidden_dim, 1)
            self.j_layer = GraphLinear(hidden_dim, out_dim)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation = activation

    def __call__(self, atoms_1, g_1, atoms_2, g_2):
        """
        :param atoms_1: atomic representation of molecule 1, with shape of (mb, N_1, hidden_dim)
        :param g_1: molecular representation of molecule 1, with shape of (mb, out_dim)
        :param atoms_2: atomic representation of molecule 2, with shape of (mb, N_2, hidden_dim)
        :param g_2: molecular representation of molecule 2, with shape of (mb, out_dim)
        :return:
        """
        # energy: (mb, N_2, N_1)
        energy = self.compute_attention(query=atoms_2, key=atoms_1)
        # energy_1: (mb, N_1)
        energy_1 = functions.mean(energy, axis=1)
        # attn_1: (mb, N_1)
        attn_1 = functions.softmax(energy_1, axis=1)
        # attn_1: (mb, N_1, out_dim)
        attn_1 = functions.tile(functions.expand_dims(attn_1, axis=-1), reps=(1, 1, self.out_dim))
        # energy_2: (mb, N_2)
        energy_2 = functions.mean(energy, axis=2)
        # attn_2: (mb, N_2)
        attn_2 = functions.softmax(energy_2, axis=1)
        # attn_2: (mb, N_2, out_dim)
        attn_2 = functions.tile(functions.expand_dims(attn_2, axis=-1), reps=(1, 1, self.out_dim))
        # compact_1: (mb, out_dim)
        compact_1 = functions.sum(attn_1 * self.j_layer(atoms_1), axis=1)
        # compact_2: (mb, out_dim)
        compact_2 = functions.sum(attn_2 * self.j_layer(atoms_2), axis=1)
        return compact_1, compact_2

    def compute_attention(self, query, key):
        """
        :param query: with shape of (mb, N_1, hidden_dim)
        :param key: with shape of (mb, N_2, hidden_dim)
        :return: attn: attention weights (mb, N_1, N_2)
        """
        energy_layer = self.energy_layer
        mb, N_1, hidden_dim = query.shape
        N_2 = key.shape[1]
        # query: (mb, N_1, 1, hidden_dim)
        query = functions.expand_dims(query, axis=2)
        # query: (mb, N_1, N_2, hidden_dim)
        query = functions.tile(query, reps=(1, 1, N_2, 1))
        # query: (mb * N_1 * N_2, hidden_dim)
        query = functions.reshape(query, (mb * N_1 * N_2, hidden_dim))
        # key: (mb, 1, N_2, hidden_dim)
        key = functions.expand_dims(key, axis=1)
        # key: (mb, N_1, N_2, hidden_dim)
        key = functions.tile(key, reps=(1, N_1, 1, 1))
        # key: (mb * N_1 * N_2, hidden_dim)
        key = functions.reshape(key, (mb * N_1 * N_2, hidden_dim))
        # energy: (mb * N_1 * N_2, 1)
        energy = self.activation(energy_layer(key, query))
        energy = functions.reshape(energy, (mb, N_1, N_2))
        return energy