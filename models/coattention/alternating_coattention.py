#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/5/2019 5:42 PM
# @Author  : chinshin
# @FileName: alternating_coattention.py

import chainer
from chainer import functions as F
from chainer_chemistry.links import GraphLinear

class AlternatingCoattention(chainer.Chain):
    def __init__(self, hidden_dim, out_dim, head, weight_tying=False):
        """
        :param hidden_dim: dimension of atom representation
        :param out_dim: dimension of molecular representation
        :param head: number of heads in attention mechanism
        :param weight_tying: indicate whether the weights should be shared between two attention computation
        """
        super(AlternatingCoattention, self).__init__()
        self.n_entities = 1 if weight_tying else 2
        with self.init_scope():
            self.energy_layers_1 = chainer.ChainList(
                *[GraphLinear(hidden_dim + out_dim, head) for _ in range(self.n_entities)]
            )
            self.energy_layers_2 = chainer.ChainList(
                *[GraphLinear(head, 1)]
            )

            self.j_layer = GraphLinear(hidden_dim, out_dim) # shared by different molecules

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.head = head
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
        attn_2 = self.compute_attention(query=compact_1, key=atoms_2, focus=2)
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
        energy_layer_1 = self.energy_layers_1[entity_index]
        energy_layer_2 = self.energy_layers_2[entity_index]
        _, N, hidden_dim = key.shape
        # query: (mb, 1, out_dim)
        query = F.expand_dims(query, axis=1)
        # query: (mb, N, out_dim)
        query = F.tile(query, reps=(1, N, 1))
        # (mb, N, out_dim + hidden_dim) -> (mb, N, head)
        energy = F.tanh(energy_layer_1(F.concat((query, key), axis=2)))
        # (mb, N, head) -> (mb, N, 1)
        energy = energy_layer_2(energy)
        # (mb, N, 1)
        attention = F.softmax(energy, axis=1)
        return attention
