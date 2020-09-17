# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: chinshin
# date: 2019/3/31
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

    """
    将GGNN中的GRU修改为fuse gate
    参考文献：Natural Language Inference Over Interaction Space
    """
    NUM_EDGE_TYPE = 4

    def __init__(self, out_dim, hidden_dim=16,
                 n_layers=4, n_atom_types=MAX_ATOMIC_NUM, concat_hidden=False,
                 dropout_rate=0.0,
                 batch_normalization=False,
                 weight_tying=True,
                 ):
        super(GGNN, self).__init__()
        n_readout_layer = n_layers if concat_hidden else 1
        n_message_layer = 1 if weight_tying else n_layers
        self.n_readout_layer = n_readout_layer
        self.n_message_layer = n_message_layer
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.concat_hidden = concat_hidden
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.weight_tying = weight_tying

        with self.init_scope():
            # Update
            self.embed = EmbedAtomID(out_size=hidden_dim, in_size=n_atom_types)
            self.embed_linear = GraphLinear(in_size=66, out_size=hidden_dim)

            self.message_layers = chainer.ChainList(
                *[GraphLinear(hidden_dim, self.NUM_EDGE_TYPE * hidden_dim)
                  for _ in range(n_message_layer)]
            )

            self.update_layer = links.GRU(2 * hidden_dim, hidden_dim)
            self.update_layer1 = links.Linear(2 * hidden_dim, hidden_dim)
            self.update_layer2 = links.Linear(2 * hidden_dim, hidden_dim)
            self.update_layer3 = links.Linear(2 * hidden_dim, hidden_dim)

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

        def fuse(h, m, dropout_rate=0.0):
            """
            :param h: (mb * atom, ch)
            :param m: (mb * atom, ch)
            :param dropout_rate: float
            :return: out_h: (mb * atom, ch)
            """
            # input: (mb * atom, ch)
            input = functions.concat((h, m), axis=1)
            # z: (mb * atom, ch)
            z = functions.tanh(self.update_layer1(input))
            # r: (mb * atom, ch)
            r = functions.sigmoid(self.update_layer2(input))
            # f: (mb * atom, ch)
            f = functions.sigmoid(self.update_layer3(input))
            out_h = functions.dropout(r * h, dropout_rate) + f * z
            return out_h

        # --- Message & Update part ---
        # (minibatch, atom, ch)
        mb, atom, ch = h.shape
        out_ch = ch
        message_layer_index = 0 if self.weight_tying else step
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
        out_h = fuse(h, m, 0.05)

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
        self.update_layer.reset_state()

        if atom_array.dtype == self.xp.int32:
            h = self.embed(atom_array)  # (minibatch, max_num_atoms)
        else:
            h = self.embed_linear(atom_array)

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
