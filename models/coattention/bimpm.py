#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/16/2019 6:29 PM
# @Author  : chinshin
# @FileName: bimpm.py

"""
BIMPM model for Sentence Matching
Bilateral Multi-Perspective Matching for Natural Language Sentences
"""

import chainer
from chainer import functions as F
from chainer import initializers
from chainer_chemistry.links import GraphLinear

class BiMPM(chainer.Chain):
    def __init__(self, hidden_dim, out_dim, head,
                 with_max_pool=True, with_att_mean=True, with_att_max=True,
                 aggr=F.sum):
        super(BiMPM, self).__init__()
        num_match = 0
        with self.init_scope():
            if with_max_pool:
                self.max_pooling_W = chainer.Parameter(initializers.HeNormal, shape=(head, hidden_dim))
                num_match += 1
            if with_att_mean:
                self.att_mean_W = chainer.Parameter(initializers.HeNormal, shape=(head, hidden_dim))
                num_match += 1
            if with_att_max:
                self.att_max_W = chainer.Parameter(initializers.HeNormal, shape=(head, hidden_dim))
                num_match += 1
            assert num_match > 0
            # self.out_layer = GraphLinear(num_match * head, out_dim)

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.head = head
        self.with_max_pool = with_max_pool
        self.with_att_mean = with_att_mean
        self.with_att_max = with_att_max
        self.aggr = aggr

    def __call__(self, atoms_1, g1, atoms_2, g2):
        """
        :param atoms_1: with shape of (mb, N_1, hidden_dim)
        :param atoms_2: with shape of (mb, N_2, hidden_dim)
        :return: repre with shape of (mb, out_dim)
        """

        def mp_matching_func(v1, v2, w):
            """
            Implementation of m = f_m(v_1, v_2, W).
            m_k = cosine(W_k \odot v_1, W_k \odot v_2)
            Similar to multi-head attention mechanism
            :param v1: (mb, N_1, hidden_dim)
            :param v2: (mb, N_1, hidden_dim) or (mb, hidden_size)
            :param w: (head, hidden_dim)
            :return: m: (mb, N_1, head)
            """
            mb, N_1, _ = v1.shape
            # w: (hidden_dim, head)
            w = F.transpose(w, axes=(1, 0))
            # w: (1, 1, hidden_dim, head)
            w = F.expand_dims(F.expand_dims(w, axis=0), axis=0)
            # v1: (mb, N_1, hidden_dim, head)
            v1 = F.tile(w, reps=(mb, N_1, 1, 1)) * F.stack([v1] * self.head, axis=3)
            if len(v2.shape) == 3:
                v2 = F.tile(w, reps=(mb, N_1, 1, 1)) * F.stack([v2] * self.head, axis=3)
            else:
                # v2: (mb, hidden_dim) -> (mb, N_1, hidden_dim) -> (mb, N_1, hidden_dim, head)
                v2 = F.tile(w, reps=(mb, N_1, 1, 1)) * F.stack([F.stack([v2] * N_1, axis=1)] * self.head, axis=3)

            # v1/v2: (mb, N_1, hidden_dim, head)
            v1_normed = F.normalize(v1, axis=2)
            v2_normed = F.normalize(v2, axis=2)
            # (mb, N_1, head, head)
            sim = F.matmul(F.transpose(v1_normed, axes=(0, 1, 3, 2)), v2_normed)
            # sim: (mb, N_1, head, head) -> (mb, N_1, head)
            sim = sim[:, :, :, 0]
            return sim 
            
        def mp_matching_func_pairwise(v1, v2, w):
            """
            Implementation of m = f_m(v_1, v_2, W).
            m_k = cosine(W_k \odot v_1, W_k \odot v_2)
            :param v1: (mb, N_1, hidden_dim)
            :param v2: (mb, N_2, hidden_dim)
            :param w: (head, hidden_dim)
            :return: sim: (mb, N_1, N_2, head)
            """
            mb, N_1, _ = v1.shape
            N_2 = v2.shape[1]
            # w: (head, hidden_dim) -> (1, head, hidden_dim) -> (1, head, 1, hidden_dim)
            w = F.expand_dims(F.expand_dims(w, axis=0), axis=2)
            # v1: (mb, head, N_1, hidden_dim)
            v1 = F.tile(w, reps=(mb, 1, N_1, 1)) * F.stack([v1] * self.head, axis=1)
            # v2: (mb, head, N_2, hidden_dim)
            v2 = F.tile(w, reps=(mb, 1, N_2, 1)) * F.stack([v2] * self.head, axis=1)
            # v1: (mb, head, N_1, hidden_dim), normalized on hidden_dim
            v1_normed = F.normalize(v1, axis=3)
            # v2: (mb, head, N_2, hidden_dim), normalized on hidden_dim
            v2_normed = F.normalize(v2, axis=3)
            # sim: (mb, head, N_1, N_2)
            sim = F.matmul(v1_normed, F.transpose(v2_normed, axes=(0, 1, 3, 2)))
            # sim: (mb, N_1, N_2, head)
            sim = F.transpose(sim, axes=(0, 2, 3, 1))
            return sim

        def attention(v1, v2):
            """
            Implementation of cosine-similarity-based attention mechanism
            :param v1: (mb, N_1, hidden_dim)
            :param v2: (mb, N_2, hidden_dim)
            :return: att: (mb, N_1, N_2)
            """
            # (mb, N_1, hidden_dim) -> (mb, N_1, hidden_dim)
            v1_normed = F.normalize(v1, axis=2)
            # (mb, N_2, hidden_dim) -> (mb, N_2, hidden_dim)
            v2_normed = F.normalize(v2, axis=2)
            # (mb, N_1, N_2)
            att = F.matmul(v1_normed, F.transpose(v2_normed, axes=(0, 2, 1)))
            return att

        def div_with_small_value(n, d, eps=1e-4):
            eps_data = self.xp.ones_like(d) * eps
            return n / F.broadcast_to(F.maximum(d, eps_data), shape=n.shape)

        mb, N_1, _ = atoms_1.shape
        N_2 = atoms_2.shape[1]

        mv_atoms1 = list()
        mv_atoms2 = list()

        if self.with_max_pool:
            # 1. Maxpooling-Matching
            # mv_max: (mb, N_1, N_2, head)
            mv_max = mp_matching_func_pairwise(atoms_1, atoms_2, self.max_pooling_W)
            # mv_atoms1_max: (mb, N_1, head)
            mv_atoms1_max = F.max(mv_max, axis=2)
            # mv_atoms2_max: (mb, N_2, head)
            mv_atoms2_max = F.max(mv_max, axis=1)

            mv_atoms1.append(mv_atoms1_max)
            mv_atoms2.append(mv_atoms2_max)

        if self.with_att_mean or self.with_att_max:
            # 2. Attentive-Matching
            # att: (mb, N_1, N_2)
            att = attention(atoms_1, atoms_2)
            # atoms_2: (mb, N_2, hidden_dim) -> (mb, 1, N_2, hidden_dim) -> (mb, N_1, N_2, hidden_dim)
            # att: (mb, N_1, N_2) -> (mb, N_1, N_2, 1) -> (mb, N_1, N_2, hidden_dim)
            # att_atoms2: (mb, N_1, N_2, hidden_dim)
            att_atoms2 = F.tile(F.expand_dims(atoms_2, axis=1), reps=(1, N_1, 1, 1)) * F.tile(F.expand_dims(att, axis=3), reps=(1, 1, 1, self.hidden_dim))

            # atoms_1: (mb, N_1, hidden_dim) -> (mb, N_1, 1, hidden_dim)
            # att: (mb, N_1, N_2) -> (mb, N_1, N_2, 1)
            att_atoms1 = F.tile(F.expand_dims(atoms_1, axis=2), reps=(1, 1, N_2, 1)) * F.tile(F.expand_dims(att, axis=3), reps=(1, 1, 1, self.hidden_dim))

            if self.with_att_mean:
                att_mean_atoms2 = div_with_small_value(F.sum(att_atoms2, axis=2), F.sum(att, axis=2, keepdims=True))
                att_mean_atoms1 = div_with_small_value(F.sum(att_atoms1, axis=1), F.transpose(F.sum(att, axis=1, keepdims=True), (0, 2, 1)))

                # mv_atoms1_att_mean: (mb, N_1, head)
                mv_atoms1_att_mean = mp_matching_func(atoms_1, att_mean_atoms2, self.att_mean_W)
                # mv_atoms2_att_mean: (mb, N_2, head)
                mv_atoms2_att_mean = mp_matching_func(atoms_2, att_mean_atoms1, self.att_mean_W)

                mv_atoms1.append(mv_atoms1_att_mean)
                mv_atoms2.append(mv_atoms2_att_mean)

            if self.with_att_max:
                # 3. Max-Attentive-Matching
                # att_atoms2: (mb, N_1, N_1, hidden_dim) -> (mb, N_1, hidden_dim)
                att_max_atoms2 = F.max(att_atoms2, axis=2)
                # att_atoms1: (mb, N_2, hidden_dim) -> (mb, N_2, hidden_dim)
                att_max_atoms1 = F.max(att_atoms1, axis=1)

                # mv_atoms1_att_max: (mb, N_1, head)
                mv_atoms1_att_max = mp_matching_func(atoms_1, att_max_atoms2, self.att_max_W)
                # mv_atoms2_att_max: (mb, N_2, head)
                mv_atoms2_att_max = mp_matching_func(atoms_2, att_max_atoms1, self.att_max_W)

                mv_atoms1.append(mv_atoms1_att_max)
                mv_atoms2.append(mv_atoms2_att_max)

        # mv_atoms1: (mb, N_1, 3 * head)
        mv_atoms1 = F.concat(mv_atoms1, axis=2)
        # mv_atoms2: (mb, N_2, 3 * head)
        mv_atoms2 = F.concat(mv_atoms2, axis=2)

        # mv_atoms1: (mb, N_1, 3 * head) -> (mb, N_1, out_dim) -> (mb, out_dim)
        mol_1 = self.aggr(mv_atoms1, axis=1)
        # mv_atoms2: (mb, N_2, 3 * head) -> (mb, N_2, out_dim) -> (mb, out_dim)
        mol_2 = self.aggr(mv_atoms2, axis=1)

        return mol_1, mol_2


