#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/22/2019 1:38 PM
# @Author  : chinshin
# @FileName: relgcn.py

from __future__ import unicode_literals

import chainer
from chainer import cuda
from chainer import functions

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.embed_atom_id import EmbedAtomID
from chainer_chemistry.links.connection.graph_linear import GraphLinear
from readout import GGNNReadout
from update import RelGCNUpdate


def rescale_adj(adj):
    xp = cuda.get_array_module(adj)
    num_neighbor = functions.sum(adj, axis=(1, 2))
    base = xp.ones(num_neighbor.shape, dtype=xp.float32)
    cond = num_neighbor.data != 0
    num_neighbor_inv = 1 / functions.where(cond, num_neighbor, base)
    return adj * functions.broadcast_to(
        num_neighbor_inv[:, None, None, :], adj.shape
    )


class RelGCN(chainer.Chain):
    def __init__(self, out_channels=64, num_edge_type=4, ch_list=None,
                 n_atom_types=MAX_ATOMIC_NUM, input_type='int',
                 scale_adj=None):
        super(RelGCN, self).__init__()
        if ch_list is None:
            ch_list = [16, 128, 64]
        with self.init_scope():
            if input_type == 'int':
                self.embed = EmbedAtomID(out_size=ch_list[0], in_size=n_atom_types)

            elif input_type == 'float':
                self.embed = GraphLinear(None, ch_list[0])

            else:
                raise ValueError("[ERROR] Unexpected value input type={}".format(input_type))

            self.rgcn_convs = chainer.ChainList(*[
                RelGCNUpdate(ch_list[i], ch_list[i+1], num_edge_type)
                for i in range(len(ch_list)-1)
            ])

            self.rgcn_readout = GGNNReadout(
                out_dim=out_channels, hidden_dim=ch_list[-1],
                nobias=True, activation=functions.tanh
            )

        self.input_type = input_type
        self.scale_adj = scale_adj

    def __call__(self, h, adj):
        if h.dtype == self.xp.int32:
            assert self.input_type == 'int'
        else:
            assert self.input_type == 'float'

        h = self.embed(h)
        if self.scale_adj:
            adj = rescale_adj(adj)
        for rgcn_conv in self.rgcn_convs:
            h = functions.tanh(rgcn_conv(h, adj))
        h = self.rgcn_readout(h)
        return h