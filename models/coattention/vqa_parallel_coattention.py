#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/5/2019 7:40 PM
# @Author  : chinshin
# @FileName: vqa_parallel_coattention.py

import chainer
from chainer import functions as F
from chainer import functions
from chainer import links
from chainer_chemistry.links import GraphLinear

class VQAParallelCoattention(chainer.Chain):
    """
    Fine-grained Co-attention where query is a matrix characterizing atom array contained in a moelcule.

    See also:

    Hierarchical Question-Image Co-Attention for Visual Question Answering


    """
    def __init__(self, hidden_dim, out_dim, head, activation=functions.tanh):
        """
        :param hidden_dim: dimension of atom representation
        :param out_dim: dimension of molecular representation
        :param head: number of heads in attention mechanism
        """
        super(VQAParallelCoattention, self).__init__()
        with self.init_scope():
            self.energy_layer = links.Bilinear(hidden_dim, hidden_dim, 1)
            self.attention_layer_1 = GraphLinear(head, 1, nobias=True)
            self.attention_layer_2 = GraphLinear(head, 1, nobias=True)
            self.lt_layer_1 = GraphLinear(hidden_dim, head, nobias=True)
            self.lt_layer_2 = GraphLinear(hidden_dim, head, nobias=True)
            self.j_layer = GraphLinear(hidden_dim, out_dim)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.head = head
        self.activation = activation

    def __call__(self, atoms_1, g_1, atoms_2, g_2):
        """
        :param atoms_1: atomic representation of molecule 1, with shape of (mb, N_1, hidden_dim)
        :param g_1: molecular representation of molecule 1, with shape of (mb, out_dim)
        :param atoms_2: atomic representation of molecule 2, with shape of (mb, N_2, hidden_dim)
        :param g_2: molecular representation of molecule 2, with shape of (mb, out_dim)
        :return:
        """
        # C: (mb, N_2, N_1)
        C = self.compute_attention(query=atoms_2, key=atoms_1)

        # L_2: (mb, N_2, N_1)
        L_2 = functions.softmax(C, axis=1)
        # L_1: (mb, N_1, N_2)
        L_1 = functions.softmax(functions.transpose(C, (0, 2, 1)), axis=1)

        # lt_atoms_1: (mb, N_1, hidden_dim) -> (mb, N_1, head)
        lt_atoms_1 = self.lt_layer_1(atoms_1)
        # lt_atoms_2: (mb, N_2, hidden_dim) -> (mb, N_2, head)
        lt_atoms_2 = self.lt_layer_2(atoms_2)
        # C: (mb, N_2, N_1), lt_atoms_2: (mb, N_2, head) -> (mb, N_1, head)
        lt_atoms_2_C = functions.matmul(L_1, lt_atoms_2)
        # H_1: (mb, N_1, head)
        H_1 = functions.tanh(functions.add(lt_atoms_1, lt_atoms_2_C))
        # C: (mb, N_2, N_1), lt_atoms_1: (mb, N_1, head)
        # lt_atoms_1_C: (mb, N_2, head)
        lt_atoms_1_C = functions.matmul(L_2, lt_atoms_1)
        H_2 = functions.tanh(functions.add(lt_atoms_2, lt_atoms_1_C))
        # attn_1: (mb, N_1, 1)
        attn_1 = functions.softmax(self.attention_layer_1(H_1))
        # attn_2: (mb, N_2, 1)
        attn_2 = functions.softmax(self.attention_layer_2(H_2))

        compact_1 = functions.sum(functions.tile(attn_1, reps=(1, 1, self.out_dim)) * self.j_layer(atoms_1), axis=1)
        compact_2 = functions.sum(functions.tile(attn_2, reps=(1, 1, self.out_dim)) * self.j_layer(atoms_2), axis=1)
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