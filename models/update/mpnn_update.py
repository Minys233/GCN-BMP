#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/22/2019 2:18 PM
# @Author  : chinshin
# @FileName: mpnn_update.py
from __future__ import unicode_literals
import chainer
from chainer import links
from chainer import functions

class MPNNUpdate(chainer.Chain):
    def __init__(self, hidden_dim=16, nn=None):
        super(MPNNUpdate, self).__init__()
        with self.init_scope():
            self.message_layer = EdgeNet(out_channels=hidden_dim, nn=nn)
            self.update_layer = links.GRU(2 * hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.nn = nn

    def __call__(self, h, adj):
        mb, node, ch = h.shape
        # h: (mb, node, 2 * ch)
        h = self.message_layer(h, adj)
        # h: (mb * node, 2 * ch)
        h = functions.reshape(h, shape=(mb * node, 2 * ch))
        # h: (mb * node, ch)
        h = self.update_layer(h)
        # h: (mb, node, ch)
        h = functions.reshape(h, shape=(mb, node, ch))
        return h

    def reset_state(self):
        self.update_layer.reset_state()


class EdgeNet(chainer.Chain):
    def __init__(self, out_channels, nn=None):
        super(EdgeNet, self).__init__()
        if nn is None:
            nn = MLP(out_dim=out_channels, hidden_dim=16)
        with self.init_scope():
            if isinstance(nn, chainer.Link):
                self.nn_layer_in = nn
                self.nn_layer_out = nn
        if not isinstance(nn, chainer.Link):
            raise ValueError('nn {} must be chainer.Link'.format(nn))
        self.out_channels = out_channels

    def __call__(self, h, adj):
        mb, node, ch = h.shape
        if ch != self.out_channels:
            raise ValueError('out channels must be equal to dimension'
                             'of feature vector associated to each atom, '
                             '{}, but it was set to {}'.format(ch, self.out_channels))

        # adj: (mb, edge_type, node, node)
        edge_type = adj.shape[1]
        adj_in = adj
        adj_out = functions.transpose(adj_in, axes=(0, 1, 3, 2))

        # expand edge vector to matrix
        # adj_in: (mb*node*node, edge_type)
        adj_in = functions.reshape(adj_in, (-1, edge_type))
        # adj_in: (mb*node*node, out_channels*out_channels)
        adj_in = self.nn_layer_in(adj_in)
        # adj_in: (mb, node, node, out_channels, out_channels)
        adj_in = functions.reshape(
            adj_in, shape=(mb, node, node, self.out_channels, self.out_channels))
        # adj_in: (mb, node, out_channels, node, out_channels)
        adj_in = functions.transpose(
            adj_in, axes=(0, 1, 3, 2, 4)
        )
        # adj_in: (mb, node*out_channels, node*out_channels)
        adj_in = functions.reshape(
            adj_in, shape=(mb, node * self.out_channels, node * self.out_channels)
        )

        adj_out = functions.reshape(adj_out, (-1, edge_type))
        adj_out = self.nn_layer_out(adj_out)
        adj_out = functions.reshape(
            adj_out, shape=(mb, node, node, self.out_channels, self.out_channels)
        )
        adj_out = functions.transpose(
            adj_out, axes=(0, 1, 3, 2, 4)
        )
        adj_out = functions.reshape(
            adj_out, shape=(mb, node * self.out_channels, node * self.out_channels)
        )

        h = functions.reshape(h, (mb, node * ch, 1))
        # message_in: (mb, node*ch, 1)
        message_in = functions.matmul(adj_in, h)
        # message_in: (mb, node, ch)
        message_in = functions.reshape(message_in, shape=(mb, node, ch))
        # message_out: (mb, node*ch, 1)
        message_out = functions.matmul(adj_out, h)
        # message_out: (mb, node, ch)
        message_out = functions.reshape(message_out, shape=(mb, node, ch))
        # message: (mb, node, ch*2)
        message = functions.concat([message_in, message_out], axis=2)
        return message


class MLP(chainer.Chain):
    def __init__(self, out_dim, hidden_dim):
        super(MLP, self).__init__()
        with self.init_scope():
            self.linear1 = links.Linear(None, hidden_dim)
            self.linear2 = links.Linear(None, out_dim**2)
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

    def __call__(self, x):
        h = functions.relu(self.linear1(x))
        return self.linear2(h)

