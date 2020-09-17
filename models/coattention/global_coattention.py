#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/17/2019 7:42 PM
# @Author  : chinshin
# @FileName: global_coattention.py

import chainer
from chainer import functions as F
from chainer import links as L
from chainer_chemistry.links import GraphLinear

class GlobalCoattention(chainer.Chain):
    def __init__(self, hidden_dim, out_dim,
                 weight_tying=True):
        super(GlobalCoattention, self).__init__()
        n_entities = 1 if weight_tying else 2
        with self.init_scope():
            self.att_layers = chainer.ChainList(
                *[L.Linear(2 * hidden_dim, out_dim) for _ in range(n_entities)]
            )
            self.lt_layer = GraphLinear(hidden_dim, out_dim)

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.weight_tying = weight_tying

    def __call__(self, atoms_1, g_1, atoms_2, g_2):
        """
        :param atoms_1: atomic representation of molecule 1, with shape of (mb, N_1, hidden_dim)
        :param g_1: molecular representation of molecule 1, with shape of (mb, out_dim)
        :param atoms_2: atomic representation of molecule 2, with shape of (mb, N_2, hidden_dim)
        :param g_2: molecular representation of molecule 2, with shape of (mb, out_dim)
        :return:
        """
        # initial_g_1: (mb, hidden_dim)
        initial_g_1 = F.mean(atoms_1, axis=1)
        # initial_g_2: (mb, hidden_dim)
        initial_g_2 = F.mean(atoms_2, axis=1)
        # attn_1: (mb, N_1, out_dim)
        attn_1 = self.compute_attention(query=initial_g_2, key=atoms_1, focus=1)
        # attn_1: (mb, N_1, out_dim) * (mb, N_1, out_dim)
        # compact_1: (mb, N_1, out_dim) -> (mb, out_dim)
        compact_1 = F.sum(attn_1 * self.lt_layer(atoms_1), axis=1)

        # attn_2: (mb, N_2, out_dim)
        attn_2 = self.compute_attention(query=initial_g_1, key=atoms_2, focus=2)
        # attn_2: (mb, N_2, out_dim) * (mb, N_2, out_dim)
        # compact_2: (mb, N_2, out_dim) -> (mb, out_dim)
        compact_2 = F.sum(attn_2 * self.lt_layer(atoms_2), axis=1)
        return compact_1, compact_2

    def compute_attention(self, query, key, focus):
        """
        :param query: with shape of (mb, hidden_dim)
        :param key: with shape of (mb, N, hidden_dim)
        :param focus: indicate the focused molecule
        :return: energy: attention weights (mb, N, out_dim)
        """
        entity_index = 0 if self.weight_tying else focus - 1
        att_layer = self.att_layers[entity_index]
        mb, N, hidden_dim = key.shape
        # query: (mb, 1, hidden_dim)
        query = F.expand_dims(query, axis=1)
        # query: (mb, N, hidden_dim)
        query = F.tile(query, reps=(1, N, 1))
        # query: (mb * N, hidden_dim)
        query = F.reshape(query, shape=(mb * N, hidden_dim))
        # key: (mb * N, hidden_dim)
        key = F.reshape(key, shape=(mb * N, hidden_dim))
        # energy: (mb * N, out_dim)
        energy = F.sigmoid(att_layer(F.concat((key, query), axis=-1)))
        # energy: (mb, N, out_dim)
        energy = F.reshape(energy, shape=(mb, N, self.out_dim))
        return energy