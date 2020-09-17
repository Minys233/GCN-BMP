#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/22/2019 1:21 PM
# @Author  : chinshin
# @FileName: relgcn_update.py

import chainer
from chainer import functions
from chainer_chemistry.links.connection.graph_linear import GraphLinear


class RelGCNUpdate(chainer.Chain):
    def __init__(self, in_channels, out_channels, num_edge_type=4):
        super(RelGCNUpdate, self).__init__()
        with self.init_scope():
            self.graph_linear_self = GraphLinear(in_channels, out_channels)
            self.graph_linear_edge = GraphLinear(
                in_channels, out_channels * num_edge_type
            )
        self.num_edge_type = num_edge_type
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __call__(self, h, adj):
        mb, node, ch = h.shape

        # self connection
        hs = self.graph_linear_self(h)

        # information aggregated from neighbor nodes.
        # m: (mb, node, out_channels * num_edge_type)
        m = self.graph_linear_edge(h)
        m = functions.reshape(
            m, shape=(mb, node, self.out_channels, self.num_edge_type)
        )
        # m: (mb, node, out_channels, num_edge_type) -> (mb, num_edge_type, out_channels, num_edge_type)
        m = functions.transpose(m, axes=(0, 3, 1, 2))
        # adj: (mb, num_edge_type, node, node)
        # m: (mb, num_edge_type, node, ch)
        m = functions.matmul(adj, m)
        # summation along the edget type dimension
        # m: (mb, node, ch)
        m = functions.sum(m, axis=1)
        return hs + m

