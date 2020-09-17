#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/28/2019 3:53 PM
# @Author  : chinshin
# @FileName: ggnn_dev_jk_gru.py

from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import chainer
from chainer import cuda
from chainer import functions
from chainer import links
from chainer.backends import cuda
import chainer_chemistry
from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links import GraphLinear


def select_aggr(layer_aggr, n_layers=None, in_size=None, out_size=None, dropout=0.0):
    assert layer_aggr is not None
    if layer_aggr == 'concat':
        aggr = ConcatAggregator()
    elif layer_aggr == 'max':
        aggr = MaxAggregator()
    elif layer_aggr == 'mean':
        aggr = AvgAggregator()
    elif layer_aggr == 'gru':
        assert n_layers is not None and \
            in_size is not None and out_size is not None
        aggr = GRUAggregator(n_layers, in_size, out_size, dropout)
    elif layer_aggr == 'bigru':
        assert n_layers is not None and \
            in_size is not None and out_size is not None
        aggr = BiGRUAggregator(n_layers, in_size, out_size, dropout)
    elif layer_aggr == 'attn':
        assert n_layers is not None
        aggr = AttnAggregator()
    else:
        raise ValueError('No such layer aggr named {}'.format(layer_aggr))
    return aggr


class GGNN(chainer.Chain):

    NUM_EDGE_TYPE = 4

    def __init__(self, out_dim, hidden_dim=16,
                 n_layers=4, n_atom_types=MAX_ATOMIC_NUM, concat_hidden=False,
                 dropout_rate=0.0, layer_aggr=None,
                 batch_normalization=False,
                 weight_tying=True, update_tying=True,
                 ):
        super(GGNN, self).__init__()
        n_readout_layer = n_layers if concat_hidden else 1
        n_message_layer = 1 if weight_tying else n_layers
        n_update_layer = 1 if update_tying else n_layers
        self.n_readout_layer = n_readout_layer
        self.n_message_layer = n_message_layer
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.concat_hidden = concat_hidden
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.weight_tying = weight_tying
        self.update_tying = update_tying
        self.layer_aggr = layer_aggr

        with self.init_scope():
            # Update
            self.embed = EmbedAtomID(out_size=hidden_dim, in_size=n_atom_types)

            self.message_layers = chainer.ChainList(
                *[GraphLinear(hidden_dim, self.NUM_EDGE_TYPE * hidden_dim)
                  for _ in range(n_message_layer)]
            )

            self.update_layer = chainer.ChainList(
                *[links.Linear(2 * hidden_dim, hidden_dim)
                  for _ in range(n_update_layer)]
            )
            # self.update_layer = links.GRU(2 * hidden_dim, hidden_dim)

            # Layer Aggregation
            self.aggr = select_aggr(layer_aggr, 1, hidden_dim, hidden_dim)

            # Readout
            self.i_layers = chainer.ChainList(
                *[GraphLinear(2 * hidden_dim, out_dim)
                  for _ in range(n_readout_layer)]
            )
            self.j_layers = chainer.ChainList(
                *[GraphLinear(hidden_dim, out_dim)
                  for _ in range(n_readout_layer)]
            )

    def update(self, h, adj, step=0):
        # --- Message & Update part ---
        # (minibatch, atom, ch)
        mb, atom, ch = h.shape
        out_ch = ch
        message_layer_index = 0 if self.weight_tying else step
        update_layer_index = 0 if self.update_tying else step
        # m: (minibatch, atom, ch) -> (minibatch, atom, edge_type * ch) -> (minibatch, atom, ch, edge_type)
        m = functions.reshape(self.message_layers[message_layer_index](h),
                              (mb, atom, out_ch, self.NUM_EDGE_TYPE))

        # m: (minibatch, atom, ch, edge_type)
        # Transpose
        # m: (minibatch, edge_type, atom, ch)
        m = functions.transpose(m, (0, 3, 1, 2))

        # adj: (minibatch * edge_type, atom, atom)
        adj = functions.reshape(adj, (mb * self.NUM_EDGE_TYPE, atom, atom))
        # m: (minibatch * edge_type, atom, ch)
        m = functions.reshape(m, (mb * self.NUM_EDGE_TYPE, atom, out_ch))

        # (mb * edge_type, atom, atom) * (mb * edge_type, atom, out_ch)
        m = chainer_chemistry.functions.matmul(adj, m)

        # (minibatch * edge_type, atom, out_ch) -> (minibatch, edge_type, atom, out_ch)
        m = functions.reshape(m, (mb, self.NUM_EDGE_TYPE, atom, out_ch))
        # Take sum
        m = functions.sum(m, axis=1)
        # (minibatch, atom, out_ch)

        # --- Update part ---
        # Contraction
        h = functions.reshape(h, (mb * atom, ch))

        # Contraction
        m = functions.reshape(m, (mb * atom, ch))

        # input for GRU: (mb * atom, 2 * ch) -> (mb * atom, ch)
        # out_h = self.update_layer(functions.concat((h, m), axis=1))
        #
        out_h = functions.relu(self.update_layer[update_layer_index](functions.concat((h, m), axis=1)))
        # Expansion: (mb * atom, ch) -> (mb, atom, ch)
        out_h = functions.reshape(out_h, (mb, atom, ch))
        return out_h

    def readout(self, h, h0, step=0):
        # --- Readout part ---
        index = step if self.concat_hidden else 0
        # h, h0: (minibatch, atom, ch)
        g = functions.sigmoid(
            self.i_layers[index](functions.concat((h, h0), axis=2))) \
            * self.j_layers[index](h)
        g = functions.sum(g, axis=1)  # sum along atom's axis
        return g

    def __call__(self, atom_array, adj):
        # reset state
        # self.update_layer.reset_state()
        # [layer.reset_state() for layer in self.update_layer]
        if atom_array.dtype == self.xp.int32:
            h = self.embed(atom_array)  # (minibatch, max_num_atoms)
        else:
            h = atom_array
        h0 = functions.copy(h, cuda.get_device_from_array(h.data).id)
        g_list = []
        h_list = []
        for step in range(self.n_layers):
            h = self.update(h, adj, step)

            if self.dropout_rate != 0.0:
                h = functions.dropout(h, ratio=self.dropout_rate)

            if self.concat_hidden:
                g = self.readout(h, h0, step)
                g_list.append(g)

            if self.layer_aggr:
                h_list.append(h)

        if self.concat_hidden:
            return functions.concat(g_list, axis=1)
        elif self.layer_aggr:
            output = self.aggr(h_list)

            return self.readout(output, h0, 0)
        else:
            g = self.readout(h, h0, 0)
            return g


class ConcatAggregator(chainer.Chain):
    def __init__(self):
        super(ConcatAggregator, self).__init__()

    def __call__(self, h_list):
        # h_list: list of elem with shape of (mb, node, ch)
        # (mb, node, ch * n_conv_layers)
        # [mb, atoms, n_layers * hidden_dim]
        h = functions.concat(h_list, axis=-1)
        return h


class MaxAggregator(chainer.Chain):
    def __init__(self):
        super(MaxAggregator, self).__init__()

    def __call__(self, h_list):
        # hs: (mb, node, ch, n_conv_layers)
        h_list = [functions.expand_dims(h, axis=-2) for h in h_list]
        # (mb, atoms, n_layers, hidden_dim)
        concat_h = functions.concat(h_list, axis=-2)
        # (mb, atoms, n_layers, hidden_dim) -> (mb, atoms, hidden_dim)
        h = functions.max(concat_h, axis=-2)
        return h


class AvgAggregator(chainer.Chain):
    def __init__(self):
        super(AvgAggregator, self).__init__()

    def __call__(self, h_list):
        # hs: (mb, node, ch, n_conv_layers)
        h_list = [functions.expand_dims(h, axis=-2) for h in h_list]
        # (mb, atoms, n_layers, hidden_dim)
        concat_h = functions.concat(h_list, axis=-2)
        # (mb, atoms, n_layers, hidden_dim) -> (mb, atoms, hidden_dim)
        h = functions.mean(concat_h, axis=-2)
        return h


class GRUAggregator(chainer.Chain):
    def __init__(self, n_layers, in_size, out_size, dropout=0.0):
        super(GRUAggregator, self).__init__()
        with self.init_scope():
            self.gru_layer = links.NStepGRU(n_layers, in_size, out_size, dropout)

        self.n_layers = n_layers
        self.in_size = in_size
        self.out_size = out_size
        self.dropout = dropout

    def __call__(self, h_list):
        h_list = [functions.expand_dims(h, axis=-2) for h in h_list]
        concat_h = functions.concat(h_list, axis=-2)
        mb, atoms, n_layers, hidden_dim = concat_h.shape
        # concat_h: (n_layers, mb, atoms, hidden_dim)
        concat_h = functions.transpose(concat_h, axes=(2, 0, 1, 3))
        seq_h = functions.reshape(concat_h, shape=(n_layers, mb * atoms, hidden_dim))
        seq_h_list = list(seq_h)
        _, seq_out_list = self.gru_layer(None, seq_h_list)
        # [n_layers, mb * atoms, hidden_dim]
        seq_out_arr = functions.concat([functions.expand_dims(seq, axis=0) for seq in seq_out_list], axis=0)
        # [mb * atoms, hidden_dim]
        seq_out_forward = seq_out_arr[-1, :, :hidden_dim]
        # [mb * atoms, 2 * hidden_dim] -> [mb, atoms, 2 * hidden_dim]
        seq_out_arr = functions.reshape(seq_out_forward, shape=(mb, atoms, hidden_dim))
        # [mb, atoms, 2 * hidden_dim]
        h = seq_out_arr

        return h


class BiGRUAggregator(chainer.Chain):
    def __init__(self, n_layers, in_size, out_size, dropout):
        super(BiGRUAggregator, self).__init__()
        with self.init_scope():
            self.bigru_layer = links.NStepBiGRU(n_layers, in_size, out_size, dropout)
            self.out_layer = GraphLinear(2 * out_size, out_size)
        self.n_layers = n_layers
        self.in_size = in_size
        self.out_size = out_size
        self.dropout = dropout

    def __call__(self, h_list):
        h_list = [functions.expand_dims(h, axis=-2) for h in h_list]
        concat_h = functions.concat(h_list, axis=-2)
        mb, atoms, n_layers, hidden_dim = concat_h.shape
        # concat_h: (n_layers, mb, atoms, hidden_dim)
        concat_h = functions.transpose(concat_h, axes=(2, 0, 1, 3))
        seq_h = functions.reshape(concat_h, shape=(n_layers, mb * atoms, hidden_dim))
        seq_h_list = list(seq_h)
        _, seq_out_list = self.bigru_layer(None, seq_h_list)
        # [n_layers, mb * atoms, hidden_dim]
        seq_out_arr = functions.concat([functions.expand_dims(seq, axis=0) for seq in seq_out_list], axis=0)
        # [mb * atoms, hidden_dim]
        seq_out_forward = seq_out_arr[-1, :, :hidden_dim]
        # [mb * atoms, hidden_dim]
        seq_out_backward = seq_out_arr[0, :, hidden_dim:]
        # [mb * atoms, 2 * hidden_dim]
        seq_out_arr = functions.concat([seq_out_forward, seq_out_backward], axis=-1)
        # [mb * atoms, 2 * hidden_dim] -> [mb, atoms, 2 * hidden_dim]
        seq_out_arr = functions.reshape(seq_out_arr, shape=(mb, atoms, 2 * hidden_dim))
        # [mb, atoms, 2 * hidden_dim]
        h = seq_out_arr
        h = self.out_layer(h)
        return h


class AttnAggregator(chainer.Chain):
    """
    query: the final graph convolution layer representation
    """
    def __init__(self):
        super(AttnAggregator, self).__init__()

    def __call__(self, h_list):
        """
        :param h_list: list of h, h with shape of (mb, node, ch)
        :return:
        """
        h_list = [functions.expand_dims(h, axis=-2) for h in h_list]
        concat_h = functions.concat(h_list, axis=-2)
        # concat_h: (n_layers, mb, atoms, hidden_dim)
        concat_h = functions.transpose(concat_h, axes=(2, 0, 1, 3))
        n_layers, mb, atoms, hidden_dim = concat_h.shape
        query_final = h_list[-1]
        energy_final = self.compute_attention(query_final, key=concat_h)
        query_first = h_list[0]
        energy_first = self.compute_attention(query_first, key=concat_h)
        energy = energy_final + energy_first
        # (n_layers, 1)
        coefs = functions.softmax(energy, axis=0)
        coefs = functions.broadcast_to(
            functions.reshape(coefs, shape=(n_layers, 1, 1, 1)),
            shape=(n_layers, mb, atoms, hidden_dim))
        # (n_layers, ) * (n_layers, mb, atoms, hidden_dim)
        prod = coefs * concat_h
        # (mb, atoms, hidden_dim)
        compact = functions.sum(prod, axis=0)
        return compact

    def compute_attention(self, query, key):
        """
        :param query: with shape of (mb, atom, ch)
        :param key: with shape of (n_layers, mb, atom, ch)
        :return: coefs: with shape of (n_layers, 1)
        """
        n_layes, mb, atom, hidden_dim = key.shape
        # query: (1, mb, atom, ch)
        query = functions.expand_dims(query, axis=0)
        # query: (1, mb * atom * hidden_dim)
        query = functions.reshape(query, shape=(1, mb * atom * hidden_dim))
        key = functions.reshape(key, shape=(n_layes, mb * atom * hidden_dim))
        # (n_layes, 1)
        energy = functions.matmul(key, query, transb=True)
        return energy