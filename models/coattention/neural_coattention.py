#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/17/2019 8:09 PM
# @Author  : chinshin
# @FileName: neural_coattention.py

import chainer
from chainer import functions as F
from chainer_chemistry.links import GraphLinear

class NeuralCoattention(chainer.Chain):
    def __init__(self, hidden_dim, out_dim,
                 activation=F.relu, weight_tying=True):
        super(NeuralCoattention, self).__init__()
        n_entities = 1 if weight_tying else 2
        with self.init_scope():
            self.att_layers = chainer.ChainList(
                *[GraphLinear(hidden_dim, out_dim) for _ in range(n_entities)]
            )


        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
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
        # initial_g_2: (mb, hidden_dim)
        initial_g_2 = F.mean(atoms_2, axis=1)
        # attn_1: (mb, N_1, 1), doc_1: (mb, N_1, out_dim)
        attn_1, doc_1 = self.compute_attention(query=initial_g_2, key=atoms_1, focus=1)
        # attn_1: (mb, N_1, out_dim)
        attn_1 = F.tile(attn_1, reps=(1, 1, self.out_dim))
        # compact_1: (mb, out_dim)
        compact_1 = F.sum(attn_1 * doc_1, axis=1)

        # initial_g_1: (mb, hidden_dim)
        initial_g_1 = F.mean(atoms_1, axis=1)
        # attn_2: (mb, N_2, 1), doc_2: (mb, N_2, out_dim)
        attn_2, doc_2 = self.compute_attention(query=initial_g_1, key=atoms_2, focus=2)
        # attn_2: (mb, N_2, out_dim)
        attn_2 = F.tile(attn_2, reps=(1, 1, self.out_dim))
        # compact_2: (mb, out_dim)
        compact_2 = F.sum(attn_2 * doc_2, axis=1)

        return compact_1, compact_2

    def compute_attention(self, query, key, focus):
        """
        :param query: with shape of (mb, hidden_dim)
        :param key: with shape of (mb, N, hidden_dim)
        :param focus: indicate the focused molecule
        :return: attn: attention weights (mb, N, 1)
        """
        entity_index = 0 if self.weight_tying else focus - 1
        att_layer = self.att_layers[entity_index]
        # query: (mb, hidden_dim) -> (mb, 1, hidden_dim)
        query = F.expand_dims(query, axis=1)
        # context: (mb, 1, out_dim)
        context = self.activation(att_layer(query))
        # doc: (mb, N, out_dim)
        doc = self.activation(att_layer(key))
        # energy: (mb, N, 1)
        energy = F.sigmoid(F.matmul(doc, F.transpose(context, axes=(0, 2, 1))))
        return energy, doc