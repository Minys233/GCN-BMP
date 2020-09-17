#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/28/2019 3:53 PM
# @Author  : chinshin
# @FileName: ggnn_dev_gate.py

from __future__ import unicode_literals
from __future__ import print_function
import chainer
from chainer import cuda
from chainer import functions
from chainer import links
from chainer.backends import cuda
import chainer_chemistry
from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links import GraphLinear


class GGNN(chainer.Chain):

    NUM_EDGE_TYPE = 4

    def __init__(self, out_dim, hidden_dim=16,
                 n_layers=4, n_atom_types=MAX_ATOMIC_NUM, concat_hidden=False,
                 dropout_rate=0.0,
                 batch_normalization=False,
                 weight_tying=True, update_tying=True,
                 ):
        super(GGNN, self).__init__()
        n_readout_layer = n_layers if concat_hidden else 1
        n_message_layer = 1 if weight_tying else n_layers
        n_update_layer = 1 if update_tying else n_layers
        self.n_readout_layer = n_readout_layer
        self.n_message_layer = n_message_layer
        self.n_update_layer = n_update_layer
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.concat_hidden = concat_hidden
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.weight_tying = weight_tying
        self.update_tying = update_tying

        with self.init_scope():
            # Update
            self.embed = EmbedAtomID(out_size=hidden_dim, in_size=n_atom_types)

            self.message_layers = chainer.ChainList(
                *[GraphLinear(hidden_dim, self.NUM_EDGE_TYPE * hidden_dim)
                  for _ in range(n_message_layer)]
            )

            # update layer implemented by GRU
            # self.update_layer = links.GRU(2 * hidden_dim, hidden_dim)
            # update layer implemented by simple gating mechanism
            # self.gate_layer = links.Linear(2 * hidden_dim, hidden_dim)
            self.gate_layer = chainer.ChainList(
                *[links.Linear(2 * hidden_dim, hidden_dim)
                  for _ in range(n_update_layer)]
            )
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
        # gating mechanism:
        # h_v^{t+1} = \alpha * m_v^{t+1} + (1 - \alpha) * h_v^t
        # \alpha = \sigma(W[h_v^t; m_v^{t+1}]+b)
        alpha = functions.sigmoid(self.gate_layer[update_layer_index](functions.concat((h, m), axis=1)))
        out_h = (1 - alpha) * h + alpha * m
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
        if atom_array.dtype == self.xp.int32:
            h = self.embed(atom_array)  # (minibatch, max_num_atoms)
        else:
            h = atom_array
        h0 = functions.copy(h, cuda.get_device_from_array(h.data).id)
        g_list = []
        for step in range(self.n_layers):
            h = self.update(h, adj, step)

            if self.dropout_rate != 0.0:
                h = functions.dropout(h, ratio=self.dropout_rate)

            if self.concat_hidden:
                g = self.readout(h, h0, step)
                g_list.append(g)

        if self.concat_hidden:
            return functions.concat(g_list, axis=1)
        else:
            g = self.readout(h, h0, 0)
            return g