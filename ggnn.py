#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/2/2019 4:05 PM
# @Author  : chinshin
# @FileName: ggnn.py
import numpy
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


class GGNN(chainer.Chain):
    """Gated Graph Neural Networks (GGNN)

    See: Li, Y., Tarlow, D., Brockschmidt, M., & Zemel, R. (2015).\
        Gated graph sequence neural networks. \
        `arXiv:1511.05493 <https://arxiv.org/abs/1511.05493>`_

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_layers (int): number of layers
        n_atom_types (int): number of types of atoms
        concat_hidden (bool): If set to True, readout is executed in each layer
            and the result is concatenated
        weight_tying (bool): enable weight_tying or not

    """
    NUM_EDGE_TYPE = 4

    def __init__(self, out_dim, hidden_dim=16,
                 n_layers=4, n_atom_types=MAX_ATOMIC_NUM, concat_hidden=False,
                 layer_aggregator=None,
                 dropout_rate=0.0,
                 batch_normalization=False,
                 weight_tying=True,
                 use_attention=False,
                 update_attention=False,
                 attention_tying=True,
                 context=False, context_layers=1, context_dropout=0.,
                 message_function='matrix_multiply',
                 edge_hidden_dim=16,
                 readout_function='graph_level', num_timesteps=3,
                 num_output_hidden_layers=0, output_hidden_dim=16, output_activation=functions.relu,
                 ):
        super(GGNN, self).__init__()
        n_readout_layer = n_layers if concat_hidden else 1
        n_message_layer = 1 if weight_tying else n_layers
        n_attention_layer = 1 if attention_tying else n_layers
        self.n_readout_layer = n_readout_layer
        self.n_message_layer = n_message_layer
        self.n_attention_layer = n_attention_layer
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.concat_hidden = concat_hidden
        self.layer_aggregator = layer_aggregator
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.weight_tying = weight_tying
        self.use_attention = use_attention
        self.update_attention = update_attention
        self.attention_tying = attention_tying
        self.context = context
        self.context_layers = context_layers
        self.context_dropout = context_dropout
        self.message_functinon = message_function
        self.edge_hidden_dim = edge_hidden_dim
        self.readout_function = readout_function
        self.num_timesteps = num_timesteps
        self.num_output_hidden_layers = num_output_hidden_layers
        self.output_hidden_dim = output_hidden_dim
        self.output_activation = output_activation

        with self.init_scope():
            # Update
            self.embed = EmbedAtomID(out_size=hidden_dim, in_size=n_atom_types)

            self.message_layers = chainer.ChainList(
                *[GraphLinear(hidden_dim, self.NUM_EDGE_TYPE * hidden_dim)
                  for _ in range(n_message_layer)]
            )

            if self.message_functinon == 'edge_network':
                del self.message_layers
                self.message_layers = chainer.ChainList(
                    *[EdgeNetwork(in_dim=self.NUM_EDGE_TYPE, hidden_dim=self.edge_hidden_dim, node_dim=self.hidden_dim)
                      for _ in range(n_message_layer)]
                )

            if self.context:
                self.context_bilstm = links.NStepBiLSTM(
                    n_layers=self.context_layers, in_size=self.hidden_dim, out_size=self.hidden_dim / 2,
                    dropout=context_dropout)

            # self-attention layer
            if use_attention or update_attention:
                # these commented layers are written for GAT impelmented by TensorFlow.
                # self.linear_transform_layer = chainer.ChainList(
                #     *[links.ConvolutionND(1, in_channels=hidden_dim, out_channels=hidden_dim, ksize=1, nobias=True)
                #         for _ in range(n_attention_layer)]
                # )
                # self.conv1d_layer_1 = chainer.ChainList(
                #     *[links.ConvolutionND(1, in_channels=hidden_dim, out_channels=1, ksize=1)
                #         for _ in range(n_attention_layer)]
                # )
                # self.conv1d_layer_2 = chainer.ChainList(
                #     *[links.ConvolutionND(1, in_channels=hidden_dim, out_channels=1, ksize=1)
                #       for _ in range(n_attention_layer)]
                # )
                self.linear_transform_layer = chainer.ChainList(
                    *[links.Linear(in_size=hidden_dim, out_size=hidden_dim, nobias=True)
                      for _ in range(n_attention_layer)]
                )
                self.neural_network_layer = chainer.ChainList(
                    *[links.Linear(in_size=2 * self.hidden_dim, out_size=1, nobias=True)
                      for _ in range(n_attention_layer)]
                )

            # batch normalization
            if batch_normalization:
                self.batch_normalization_layer = links.BatchNormalization(size=hidden_dim)

            self.update_layer = links.GRU(2 * hidden_dim, hidden_dim)
            # Readout
            self.i_layers = chainer.ChainList(
                *[GraphLinear(2 * hidden_dim, out_dim)
                  for _ in range(n_readout_layer)]
            )
            self.j_layers = chainer.ChainList(
                *[GraphLinear(hidden_dim, out_dim)
                  for _ in range(n_readout_layer)]
            )

            if self.readout_function == 'set2vec':
                del self.i_layers, self.j_layers
                # def __init__(self, node_dim, output_dim, num_timesteps=3, inner_prod='default',
                #   num_output_hidden_layers=0, output_hidden_dim=16, activation=chainer.functions.relu):
                self.readout_layer = chainer.ChainList(
                    *[Set2Vec(node_dim=self.hidden_dim * 2, output_dim=out_dim, num_timesteps=num_timesteps,
                              num_output_hidden_layers=num_output_hidden_layers, output_hidden_dim=output_hidden_dim,
                              activation=output_activation)
                      for _ in range(n_readout_layer)]
                )

            if self.layer_aggregator:
                self.construct_layer_aggregator()

                if self.layer_aggregator == 'gru-attn' or 'gru':
                    self.bigru_layer = links.NStepBiGRU(n_layers=1, in_size=self.hidden_dim, out_size=self.hidden_dim,
                                                        dropout=0.)
                if self.layer_aggregator == 'lstm-attn' or 'lstm':
                    self.bilstm_layer = links.NStepBiLSTM(n_layers=1, in_size=self.hidden_dim, out_size=self.hidden_dim,
                                                          dropout=0.)
                if self.layer_aggregator == 'gru-attn' or 'lstm-attn' or 'attn':
                    self.attn_dense_layer = links.Linear(in_size=self.n_layers, out_size=self.n_layers)
                if self.layer_aggregator == 'self-attn':
                    self.attn_linear_layer = links.Linear(in_size=self.n_layers, out_size=self.n_layers)

    def construct_layer_aggregator(self):
        """
        self.layer_aggregator: can be as follows,
        concat: concatenation of hidden state of different layers for each node
        max-pool: element-wise max-pooling of hidden state of different layers for each node
        attn: attention mechanism implemented by a single-layered neural network
        lstm-attn:
        gru_attn:
        lstm:
        gru:
        """
        if self.layer_aggregator == 'concat':
            input_dim = self.n_layers * self.hidden_dim
            del self.i_layers, self.j_layers
            self.i_layers = chainer.ChainList(
                *[GraphLinear(in_size=2 * input_dim, out_size=self.out_dim)
                  for _ in range(self.n_readout_layer)]
            )
            self.j_layers = chainer.ChainList(
                *[GraphLinear(in_size=input_dim, out_size=self.out_dim)
                  for _ in range(self.n_readout_layer)]
            )
        elif self.layer_aggregator == 'max-pool':
            input_dim = self.hidden_dim
            if input_dim == self.hidden_dim:
                return

        elif self.layer_aggregator == 'attn':
            input_dim = self.hidden_dim
            if input_dim == self.hidden_dim:
                return

        elif self.layer_aggregator == 'lstm-attn' or 'gru-attn':
            input_dim = 2 * self.hidden_dim
            if input_dim == self.hidden_dim:
                return
            else:
                del self.i_layers, self.j_layers
                self.i_layers = chainer.ChainList(
                    *[GraphLinear(in_size=2 * input_dim, out_size=self.out_dim)
                      for _ in range(self.n_readout_layer)]
                )
                self.j_layers = chainer.ChainList(
                    *[GraphLinear(in_size=input_dim, out_size=self.out_dim)
                      for _ in range(self.n_readout_layer)]
                )

    def update(self, h, adj, step=0):
        # --- Message & Update part ---
        # (minibatch, atom, ch)
        mb, atom, ch = h.shape
        out_ch = ch
        message_layer_index = 0 if self.weight_tying else step
        # m: (minibatch, atom, ch) -> (minibatch, atom, edge_type * ch) -> (minibatch, atom, ch, edge_type)
        if self.message_functinon == 'matrix_multiply':
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

        elif self.message_functinon == 'edge_network':
            # (minibatch, atom, out_ch)
            m = functions.reshape(self.message_layers[message_layer_index](h, adj),
                                  (mb, atom, out_ch))
        else:
            raise ValueError('There is no such message function named {}'.format(self.message_functinon))

        # --- Update part ---
        # Contraction
        h = functions.reshape(h, (mb * atom, ch))

        # Contraction
        m = functions.reshape(m, (mb * atom, ch))

        # input for GRU: (mb * atom, 2 * ch) -> (mb * atom, ch)
        out_h = self.update_layer(functions.concat((h, m), axis=1))
        # Expansion: (mb * atom, ch) -> (mb, atom, ch)
        out_h = functions.reshape(out_h, (mb, atom, ch))
        return out_h

    def update_with_attention(self, h, adj, step=0):
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

        # masked self-attention mechanism
        attention_layer_index = 0 if self.attention_tying else step
        h = functions.reshape(m, shape=(mb * self.NUM_EDGE_TYPE * atom, out_ch))
        h = self.linear_transform_layer[attention_layer_index](h)
        h = functions.reshape(h, shape=(mb * self.NUM_EDGE_TYPE, atom, 2 * self.hidden_dim))
        # (minibatch * edge_type, atom, atom, hidden_dim)
        a_input = functions.concat(
            [functions.tile(h, reps=(1, 1, atom)).reshape(mb * self.NUM_EDGE_TYPE, atom * atom, -1),
             functions.tile(h, reps=(1, atom, 1))], axis=-1).reshape(mb * self.NUM_EDGE_TYPE, atom, atom,
                                                                     2 * self.hidden_dim)
        # (minibatch * edge_type, atom, atom)
        e = functions.leaky_relu(
            functions.reshape(functions.squeeze(self.neural_network_layer[attention_layer_index](a_input), axis=-1),
                              shape=(mb * self.NUM_EDGE_TYPE, atom, atom)))
        # (minibatch * edge_type, atom, atom)
        zero_vec = -9e15 * self.xp.ones_like(e, dtype=self.xp.float32)
        # (minibatch * edge_type, atom, atom)
        attention = functions.where(adj > 0, e, zero_vec)
        # (minibatch * edge_type, atom, atom)
        attention = functions.softmax(attention, axis=2)

        # modify adj using attention score
        # adj: (minibatch * edge_type, atom, atom) * attention: (minibatch * edge_type, atom, atom)
        adj *= attention

        # message construction part: unweighted summation
        # change unweighted summation into weighted summation by incoprating the masked self-attention mechanism
        # established by Graph Attention Networks.
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

        out_h = self.update_layer(functions.concat((h, m), axis=1))
        # Expansion
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

    def self_attention(self, h, adj, step):
        attention_layer_index = 0 if self.attention_tying else step
        mask = np.sum(adj, axis=1)
        mask[mask == 0] = - 10000
        # [mb, atoms, ch] -> [mb, ch, atoms]
        mb, atoms, ch = h.shape
        h = functions.transpose(h, axes=(0, 2, 1))
        h = self.linear_transform_layer[attention_layer_index](h)
        # [mb, 1, atoms]
        f_1 = self.conv1d_layer_1[attention_layer_index](h)
        # [mb, 1, atoms] -> [mb, atoms, 1]
        f_1 = functions.transpose(f_1, axes=(0, 2, 1))
        # [mb, atoms, 1] -> [mb, atoms, atoms]
        f_1 = functions.tile(f_1, reps=(1, 1, atoms))
        # [mb, 1, atoms]
        f_2 = self.conv1d_layer_2[attention_layer_index](h)
        # [mb, 1, atoms] -> [mb, atoms, atoms]
        f_2 = functions.tile(f_2, reps=(1, atoms, 1))
        logits = f_1 + f_2
        # logits *= mask
        # [mb, atoms, atoms]
        coefs = functions.softmax(functions.leaky_relu(logits))
        coefs = functions.transpose(coefs, axes=(0, 2, 1))
        # [mb, ch, atoms] -> [mb, atoms, ch]
        h = functions.transpose(h, axes=(0, 2, 1))

        h = functions.dropout(h, ratio=self.dropout_rate) if self.dropout_rate != 0.0 else h
        # [mb, atoms, atoms] * [mb, atoms, ch]
        vals = functions.matmul(coefs, h)

        h = functions.elu(vals)
        return h

    def masked_self_attention(self, input, adj, step):
        adj = np.sum(adj, axis=1)
        # [mb, atoms, ch]
        mb, atoms, ch = input.shape
        attention_layer_index = 0 if self.attention_tying else step
        # [mb, atoms, hidden_dim]
        h = functions.reshape(input, shape=(mb * atoms, ch))
        h = self.linear_transform_layer[attention_layer_index](h)
        h = functions.reshape(h, shape=(mb, atoms, -1))
        # [mb, atoms, atoms, 2 * hidden_dim]
        a_input = functions.concat([functions.tile(h, reps=(1, 1, atoms)).reshape(mb, atoms * atoms, -1),
                                    functions.tile(h, reps=(1, atoms, 1))], axis=-1).reshape(mb, atoms, atoms,
                                                                                             2 * self.hidden_dim)
        a_input = functions.reshape(a_input, shape=(mb * atoms * atoms, 2 * self.hidden_dim))
        # [mb * atoms * atoms, 2 * hidden_dim] => [mb * atoms * atoms, 1] => [mb, atoms * atoms]
        e = functions.leaky_relu(
            functions.reshape(functions.squeeze(self.neural_network_layer[attention_layer_index](a_input), axis=-1),
                              shape=(mb, atoms, atoms)))

        # [mb, atoms, atoms]
        zero_vec = -9e15 * self.xp.ones_like(e, dtype=self.xp.float32)
        # [mb, atoms, atoms]
        attention = functions.where(adj > 0, e, zero_vec)

        # [mb, atoms, atoms]
        attention = functions.softmax(attention, axis=2)
        # [mb, atoms, atoms] * [mb, atoms, hidden_dim] => [mb, atoms, hidden_dim]
        h_prime = functions.matmul(attention, h)
        h_prime = functions.elu(h_prime)
        return h_prime

    def layer_aggregation(self, h_list, h0):
        """
        :param h_list: list of h (mb, atoms, hidden_dim), len is n_layers
        :param h0:
        :return:
        """
        read_out_g = None

        if self.layer_aggregator == 'concat':
            # [mb, atoms, n_layers * hidden_dim]
            h = functions.concat(h_list, axis=-1)
            # [mb, atoms, n_layers * hidden_dim]
            concat_h0 = functions.concat([h0] * self.n_layers, axis=-1)
            concat_g = self.readout(h, concat_h0, step=0)
            # (mb, atoms, n_layers * hidden_dim)
            read_out_g = concat_g

        elif self.layer_aggregator == 'max-pool':
            h_list = [functions.expand_dims(h, axis=-2) for h in h_list]
            # (mb, atoms, n_layers, hidden_dim)
            concat_h = functions.concat(h_list, axis=-2)
            # (mb, atoms, n_layers, hidden_dim) -> (mb, atoms, hidden_dim)
            h = functions.max(concat_h, axis=-2)
            max_g = self.readout(h, h0, step=0)
            # (mb, atoms, hidden_dim)
            read_out_g = max_g

        elif self.layer_aggregator == 'lstm':
            h_list = [functions.expand_dims(h, axis=-2) for h in h_list]
            concat_h = functions.concat(h_list, axis=-2)
            mb, atoms, n_layers, hidden_dim = concat_h.shape
            seq_h = functions.reshape(concat_h, shape=(mb * atoms, n_layers, hidden_dim))
            seq_h_list = list(seq_h)
            _, _, seq_out_list = self.bilstm_layer(None, None, seq_h_list)
            # [mb * atoms, n_layers, hidden_dim]
            seq_out_arr = np.array(seq_out_list, dtype=np.float32)
            # [mb * atoms, hidden_dim]
            seq_out_forward = seq_out_arr[:, -1, :hidden_dim]
            # [mb * atoms, hidden_dim]
            seq_out_backward = seq_out_arr[:, 0, hidden_dim:]
            # [mb * atoms, 2 * hidden_dim]
            seq_out_arr = functions.concat([seq_out_forward, seq_out_backward], axis=-1)
            # [mb * atoms, 2 * hidden_dim] -> [mb, atoms, 2 * hidden_dim]
            seq_out_arr = functions.reshape(seq_out_arr, shape=(mb, atoms, 2 * hidden_dim))
            # [mb, atoms, 2 * hidden_dims]
            h = seq_out_arr
            # [mb, atoms, 2 * hidden_dim]
            lstm_h0 = functions.concat([h0, h0], axis=-1)
            lstm_g = self.readout(h, lstm_h0, step=0)
            read_out_g = lstm_g

        elif self.layer_aggregator == 'gru':
            h_list = [functions.expand_dims(h, axis=-2) for h in h_list]
            concat_h = functions.concat(h_list, axis=-2)
            mb, atoms, n_layers, hidden_dim = concat_h.shape
            seq_h = functions.reshape(concat_h, shape=(mb * atoms, n_layers, hidden_dim))
            seq_h_list = list(seq_h)
            _, seq_out_list = self.bigru_layer(None, seq_h_list)
            # [mb * atoms, n_layers, hidden_dim]
            seq_out_arr = functions.concat([functions.expand_dims(seq, axis=0) for seq in seq_out_list], axis=0)
            # [mb * atoms, hidden_dim]
            seq_out_forward = seq_out_arr[:, -1, :hidden_dim]
            # [mb * atoms, hidden_dim]
            seq_out_backward = seq_out_arr[:, 0, hidden_dim:]
            # [mb * atoms, 2 * hidden_dim]
            seq_out_arr = functions.concat([seq_out_forward, seq_out_backward], axis=-1)
            # [mb * atoms, 2 * hidden_dim] -> [mb, atoms, 2 * hidden_dim]
            seq_out_arr = functions.reshape(seq_out_arr, shape=(mb, atoms, 2 * hidden_dim))
            # [mb, atoms, 2 * hidden_dim]
            h = seq_out_arr
            # [mb, atoms, 2 * hidden_dim]
            gru_h0 = functions.concat([h0, h0], axis=-1)
            gru_g = self.readout(h, gru_h0, step=0)
            read_out_g = gru_g

        elif self.layer_aggregator == 'lstm-attn':
            h_list = [functions.expand_dims(h, axis=-2) for h in h_list]
            concat_h = functions.concat(h_list, axis=-2)
            mb, atoms, n_layers, hidden_dim = concat_h.shape
            seq_h = functions.reshape(concat_h, shape=(mb * atoms, n_layers, hidden_dim))
            seq_h_list = list(seq_h)
            hy, cy, seq_out_list = self.bilstm_layer(None, None, seq_h_list)

            # attention mechanism implemented by a single-layered feedforward neural network
            # (mb * atoms, n_layers, 2 * hidden_dim)
            seq_out = functions.concat([functions.expand_dims(elem, axis=0) for elem in seq_out_list], axis=0)
            # (mb * atoms, n_layers, 2 * hidden_dim) -> (mb * atoms, 2 * hidden_dim, n_layers)
            a = functions.transpose(seq_out, axes=(0, 2, 1))
            # (mb * atoms, 2 * hidden_dim, n_layers) -> (mb * atoms * 2 * hidden_dim, n_layers)
            a = functions.reshape(a, shape=(mb * atoms * 2 * hidden_dim, n_layers))
            attn = self.attn_dense_layer(a)
            attn_prob = functions.softmax(attn, axis=1)
            # (mb * atoms * 2 * hidden_dim, n_layers) -> (mb * atoms, 2 * hidden_dim, n_layers)
            attn_prob = functions.reshape(attn_prob, shape=(mb * atoms, 2 * hidden_dim, n_layers))
            # (mb * atoms, 2 * hidden_dim, n_layers) -> (mb * atoms, n_layers, 2 * hidden_dim)
            attn_prob = functions.transpose(attn_prob, axes=(0, 2, 1))
            # seq_out && attn_prob, equal to assign different weights to different layers
            output_attention_mul = seq_out * attn_prob
            # (mb * atoms, n_layers, 2 * hidden_dim) -> (mb * atoms, 2 * hidden_dim)
            output_attention_mul = functions.sum(output_attention_mul, axis=1)
            # (mb * atoms, 2 * hidden_dim) -> (mb, atoms, 2 * hidden_dim)
            output_attention_mul = functions.reshape(output_attention_mul, shape=(mb, atoms, 2 * hidden_dim))
            h = output_attention_mul
            # (mb, atoms, hidden_dim) -> (mb, atoms, 2 * hidden_dim)
            attn_h0 = functions.concat([h0, h0], axis=-1)
            # (mb, atoms, 2 * hidden_dim) -> (mb, 2 * hidden_dim)
            attn_g = self.readout(h, attn_h0, step=0)
            read_out_g = attn_g

        elif self.layer_aggregator == 'gru-attn':
            h_list = [functions.expand_dims(h, axis=-2) for h in h_list]
            concat_h = functions.concat(h_list, axis=-2)
            mb, atoms, n_layers, hidden_dim = concat_h.shape
            seq_h = functions.reshape(concat_h, shape=(mb * atoms, n_layers, hidden_dim))
            seq_h_list = list(seq_h)
            # hy, cy, seq_out_list = self.bilstm_layer(None, None, seq_h_list)
            _, seq_out_list = self.bigru_layer(None, seq_h_list)

            # attention mechanism implemented by a single-layered feedforward neural network
            # (mb * atoms, n_layers, 2 * hidden_dim)
            seq_out = functions.concat([functions.expand_dims(elem, axis=0) for elem in seq_out_list], axis=0)
            # (mb * atoms, n_layers, 2 * hidden_dim) -> (mb * atoms, 2 * hidden_dim, n_layers)
            a = functions.transpose(seq_out, axes=(0, 2, 1))
            # (mb * atoms, 2 * hidden_dim, n_layers) -> (mb * atoms * 2 * hidden_dim, n_layers)
            a = functions.reshape(a, shape=(mb * atoms * 2 * hidden_dim, n_layers))
            attn = self.attn_dense_layer(a)
            attn_prob = functions.softmax(attn, axis=1)
            # (mb * atoms * 2 * hidden_dim, n_layers) -> (mb * atoms, 2 * hidden_dim, n_layers)
            attn_prob = functions.reshape(attn_prob, shape=(mb * atoms, 2 * hidden_dim, n_layers))
            # (mb * atoms, 2 * hidden_dim, n_layers) -> (mb * atoms, n_layers, 2 * hidden_dim)
            attn_prob = functions.transpose(attn_prob, axes=(0, 2, 1))
            # seq_out && attn_prob, equal to assign different weights to different layers
            output_attention_mul = seq_out * attn_prob
            # (mb * atoms, n_layers, 2 * hidden_dim) -> (mb * atoms, 2 * hidden_dim)
            output_attention_mul = functions.sum(output_attention_mul, axis=1)
            # (mb * atoms, 2 * hidden_dim) -> (mb, atoms, 2 * hidden_dim)
            output_attention_mul = functions.reshape(output_attention_mul, shape=(mb, atoms, 2 * hidden_dim))
            h = output_attention_mul
            # (mb, atoms, hidden_dim) -> (mb, atoms, 2 * hidden_dim)
            attn_h0 = functions.concat([h0, h0], axis=-1)
            # (mb, atoms, 2 * hidden_dim) -> (mb, 2 * hidden_dim)
            attn_g = self.readout(h, attn_h0, step=0)
            read_out_g = attn_g

        elif self.layer_aggregator == 'attn':
            h_list = [functions.expand_dims(h, axis=-2) for h in h_list]
            # [mb, atoms, n_layers, hidden_dim]
            concat_h = functions.concat(h_list, axis=-2)
            mb, atoms, n_layers, hidden_dim = concat_h.shape
            # [mb, atoms, n_layers, hidden_dim] -> [mb, atoms, hidden_dim, n_layers]
            a = functions.transpose(concat_h, axes=(0, 1, 3, 2))
            # [mb * atoms * hidden_dim, n_layers]
            a = functions.reshape(a, shape=(mb * atoms * hidden_dim, n_layers))
            attn = self.attn_dense_layer(a)
            attn_prob = functions.softmax(attn, axis=1)
            # [mb * atoms * hidden_dim, n_layers] -> [mb, atoms, hidden_dim, n_layers]
            attn_prob = functions.reshape(attn_prob, shape=(mb, atoms, hidden_dim, n_layers))
            # [mb, atoms, n_layers, hidden_dim]
            attn_prob = functions.transpose(attn_prob, axes=(0, 1, 3, 2))
            output_attention_mul = concat_h * attn_prob
            # [mb, atoms, n_layers, hidden_dim] -> [mb, atoms, hidden_dim]
            output_attention_mul = functions.sum(output_attention_mul, axis=2)
            h = output_attention_mul
            attn_g = self.readout(h, h0, step=0)
            read_out_g = attn_g

        elif self.layer_aggregator == 'self-attn':
            h_list = [functions.expand_dims(h, axis=-2) for h in h_list]
            # [mb, atmos, n_layers, hidden_dim]
            concat_h = functions.concat(h_list, axis=-2)
            mb, atoms, n_layers, hidden_dim = concat_h.shape

        return read_out_g

    # def encode_layer_outputs(self):
    #     return self.encode_layer_outputs

    def __call__(self, atom_array, adj):
        """Forward propagation

        Args:
            atom_array (numpy.ndarray): minibatch of molecular which is
                represented with atom IDs (representing C, O, S, ...)
                `atom_array[mol_index, atom_index]` represents `mol_index`-th
                molecule's `atom_index`-th atomic number
            adj (numpy.ndarray): minibatch of adjancency matrix with edge-type
                information with shape of (mb, n_layers, atoms, atoms)

        Returns:
            ~chainer.Variable: minibatch of fingerprint
        """
        # reset state
        self.update_layer.reset_state()
        if atom_array.dtype == self.xp.int32:
            # if self.xp == cuda.cupy:
            #     atom_array = cuda.to_gpu(atom_array)
            h = self.embed(atom_array)  # (minibatch, max_num_atoms)
        else:
            h = atom_array
        # context embedding
        if self.context:
            h_list = [sub_h for sub_h in h]
            _, _, context_h_list = self.context_bilstm(None, None, h_list)
            context_h = functions.concat([functions.expand_dims(sub_h, axis=0) for sub_h in context_h_list], axis=0)
            h = context_h
        h0 = functions.copy(h, cuda.get_device_from_array(h.data).id)
        g_list = []
        h_list = []
        # self.encode_layer_outputs = []
        for step in range(self.n_layers):
            if self.update_attention:
                h = self.update_with_attention(h, adj, step)
            else:
                h = self.update(h, adj, step)

            if self.use_attention:
                # h = self.self_attention(h, adj, step)
                h = self.masked_self_attention(h, adj, step)

            if self.dropout_rate != 0.0:
                h = functions.dropout(h, ratio=self.dropout_rate)

            if self.concat_hidden:
                if self.readout_function == 'set2vec':
                    h_input = functions.concat((h, h0), axis=2) if h0 is not None else h
                    g = self.readout_layer[step](h_input)
                else:
                    g = self.readout(h, h0, step)
                g_list.append(g)

            # preparation for layer aggregation
            if self.layer_aggregator:
                h_list.append(h)

            # self.encode_layer_outputs.append(h)

        if self.layer_aggregator:
            return self.layer_aggregation(h_list, h0)

        if self.concat_hidden:
            return functions.concat(g_list, axis=1)
        else:
            if self.readout_function == 'set2vec':
                h_input = functions.concat((h, h0), axis=2) if h0 is not None else h
                g = self.readout_layer[0](h_input)
            else:
                g = self.readout(h, h0, 0)
            return g


class EdgeNetwork(chainer.Chain):
    def __init__(self, in_dim, hidden_dim, node_dim, n_hidden_layers=0, activation=functions.relu, dropout=0.0):
        # in_dim should be layer_dim
        # hidden_dim is edge_hidden_dim
        # out_dim should be node_dim ** 2
        super(EdgeNetwork, self).__init__()
        with self.init_scope():
            first_linear_layer = links.Linear(in_size=in_dim, out_size=hidden_dim)
            rest_linear_layers = [links.Linear(in_size=hidden_dim, out_size=hidden_dim) for _ in
                                  range(n_hidden_layers - 1)]
            hidden_layers = [first_linear_layer] + rest_linear_layers
            self.hidden_layers = chainer.ChainList(
                *hidden_layers
            )
            if n_hidden_layers == 0:
                self.output_layer = links.Linear(in_size=in_dim, out_size=node_dim ** 2)
            else:
                self.output_layer = links.Linear(in_size=hidden_dim, out_size=node_dim ** 2)
            self.bias_add_layer = links.Bias(axis=1)

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.out_dim = self.node_dim ** 2
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.dropout = dropout

    def __call__(self, h, adj):
        # h: (mb, atoms, hidden_dim)
        mb, num_edge_type, atoms, _ = adj.shape
        # (mb, atoms, atoms, num_edge_type)
        adj_in_one_hot = functions.transpose(adj, axes=(0, 2, 3, 1))
        adj_reshape_in = functions.reshape(adj_in_one_hot, shape=(mb * atoms * atoms, num_edge_type))
        adj_nn = adj_reshape_in
        for i in range(self.n_hidden_layers):
            # layer_dim = num_edge_type
            adj_nn = self.hidden_layers[i](adj_nn)
            adj_nn = self.activation(adj_nn)
            if self.dropout != 0.0:
                adj_nn = functions.dropout(adj_nn, ratio=self.dropout)

        # (mb * atoms * atoms, out_dim) = (mb * atoms * atoms, node_dim * node_dim)
        adj_output = self.output_layer(adj_nn)

        adj_tmp = functions.reshape(adj_output, shape=(mb, atoms, atoms, self.node_dim, self.node_dim))
        a = functions.reshape(
            functions.transpose(adj_tmp, axes=(0, 1, 3, 2, 4)),
            shape=(-1, atoms * self.node_dim, atoms * self.node_dim)
        )

        # a: (mb, atoms * hidden_dim, atoms * hidden_dim)
        # flat: (mb, atoms * hidden_dim, 1)
        h_flat = functions.reshape(h, shape=(mb, atoms * self.node_dim, 1))
        a_mul = functions.reshape(
            functions.matmul(a, h_flat), shape=(mb * atoms, self.node_dim)
        )

        message_bias = self.xp.zeros(shape=(self.node_dim,), dtype=self.xp.float32)
        message_bias = chainer.Variable(data=message_bias, name='message_bias')
        a_t = functions.bias(a_mul, message_bias, axis=1)
        messages = functions.reshape(a_t, shape=(mb, atoms, self.node_dim))

        return messages


class Set2Vec(chainer.Chain):
    def __init__(self, node_dim, output_dim, num_timesteps=3, inner_prod='default',
                 num_output_hidden_layers=0, output_hidden_dim=16, activation=chainer.functions.relu):
        super(Set2Vec, self).__init__()
        with self.init_scope():
            self.lstm_block = LSTMWithoutInput(node_dim=node_dim, attention_m=True)
            self.ff_block = FeedForward(node_dim=node_dim * 2, output_dim=output_dim,
                                        num_hidden_layers=num_output_hidden_layers,
                                        hidden_dim=output_hidden_dim,
                                        activation=activation)

        self.node_dim = node_dim
        self.output_dim = output_dim
        self.num_timesteps = num_timesteps
        self.inner_prod = inner_prod
        self.num_output_hidden_layers = num_output_hidden_layers
        self.output_hidden_dim = output_hidden_dim
        self.activation = activation

    def __call__(self, h):
        # h: (mb, atoms, node_dim)
        mb, atoms, node_dim = h.shape
        # (mb, atoms, 1, node_dim)
        h_exp = functions.expand_dims(h, axis=2)
        attention_w1_data = self.xp.zeros(shape=(1, 1, node_dim, node_dim), dtype=self.xp.float32)
        init = chainer.initializers.GlorotUniform(dtype=self.xp.float32)
        init(attention_w1_data)
        attention_w1 = chainer.Variable(data=attention_w1_data, name='attention_w1')
        # (mb, node_dim, atoms, 1)
        h_exp_transposed = functions.transpose(h_exp, axes=(0, 3, 1, 2))
        # (node_dim, node_dim, 1, 1)
        attention_w1_transposed = functions.transpose(attention_w1, axes=(2, 3, 0, 1))
        # (mb, node_dim, atoms, 1)
        embedded_nodes = functions.convolution_2d(h_exp_transposed, attention_w1_transposed, stride=(1, 1))
        # (mb, atoms, 1, node_dim)
        embedded_nodes = functions.transpose(embedded_nodes, axes=(0, 2, 3, 1))

        # (mb, 2 * node_dim)
        _, _, m = self.set2vec(embedded_nodes, self.num_timesteps, inner_prod=self.inner_prod,
                               name="output_LSTMLoopAtt")

        # feedforward NN
        # (mb, output_dim)
        output = self.ff_block(m)

        return output

    def set2vec(self, input_set, num_timesteps, mprev=None, cprev=None, inner_prod='default', name='lstm'):
        batch_size = input_set.shape[0]
        node_dim = input_set.shape[3]
        assert self.node_dim == node_dim

        if mprev is None:
            mprev = chainer.Variable(self.xp.zeros(shape=(batch_size, node_dim), dtype=self.xp.float32))
        # (batch_size, node_dim * 2)
        mprev = functions.concat(
            [mprev, chainer.Variable(self.xp.zeros(shape=(batch_size, node_dim), dtype=self.xp.float32))], axis=1)
        if cprev is None:
            cprev = chainer.Variable(self.xp.zeros(shape=(batch_size, node_dim), dtype=self.xp.float32))

        logit_att = []
        attention_w2_data = self.xp.zeros(shape=(node_dim, node_dim), dtype=self.xp.float32)
        init = chainer.initializers.GlorotUniform(dtype=self.xp.float32)
        init(attention_w2_data)
        attention_w2 = chainer.Variable(data=attention_w2_data, name=name + "att_W_2")
        attention_v_data = self.xp.zeros(shape=(node_dim, 1), dtype=self.xp.float32)
        attention_v = chainer.Variable(data=attention_v_data, name=name + "att_V")

        for i in range(num_timesteps):
            m, c = self.lstm_block(mprev, cprev)
            query = functions.matmul(m, attention_w2)
            query = functions.reshape(query, shape=(-1, 1, 1, node_dim))
            if inner_prod == 'default':
                energies = functions.reshape(
                    functions.matmul(
                        functions.reshape(functions.tanh(
                            functions.add(functions.broadcast_to(query, shape=input_set.shape), input_set)),
                            shape=(-1, node_dim)),
                        attention_v
                    ), shape=(batch_size, -1)
                )
            elif inner_prod == 'dot':
                att_mem_reshape = functions.reshape(input_set, shape=(batch_size, -1, node_dim))
                query = functions.reshape(query, shape=(-1, node_dim, 1))
                energies = functions.reshape(functions.matmul(att_mem_reshape, query), shape=(batch_size, -1))
            else:
                raise ValueError("Invalid inner_prod type: {}".format(inner_prod))

            att = functions.softmax(energies)

            att = functions.reshape(att, shape=(batch_size, -1, 1, 1))

            read = functions.sum(functions.broadcast_to(att, shape=input_set.shape) * input_set, axis=(1, 2))
            m = functions.concat([m, read], axis=1)

            logit_att.append(m)
            mprev = m
            cprev = c

        return logit_att, c, m


class LSTMWithoutInput(chainer.Chain):
    def __init__(self, node_dim, attention_m=False):
        super(LSTMWithoutInput, self).__init__()

        m_nodes = node_dim
        if attention_m:
            m_nodes = 2 * node_dim

        with self.init_scope():
            self.input_gate = links.Linear(in_size=m_nodes, out_size=node_dim)
            self.forget_gate = links.Linear(in_size=m_nodes, out_size=node_dim)
            self.cell_gate = links.Linear(in_size=m_nodes, out_size=node_dim)
            self.output_gate = links.Linear(in_size=m_nodes, out_size=node_dim)

        self.node_dim = node_dim
        self.attention_m = attention_m

    def __call__(self, mprev, cprev):
        i_g = functions.sigmoid(self.input_gate(mprev))
        f_g = functions.sigmoid(self.forget_gate(mprev))
        cprime = functions.sigmoid(self.cell_gate(mprev))
        c = f_g * cprev + i_g * functions.tanh(cprime)
        o_g = functions.sigmoid(self.output_gate(mprev))
        m = o_g * functions.tanh(c)
        return m, c


class FeedForward(chainer.Chain):
    def __init__(self, node_dim, output_dim, num_hidden_layers=0, hidden_dim=16, dropout=None,
                 activation=functions.tanh):
        super(FeedForward, self).__init__()
        with self.init_scope():
            first_layer = links.Linear(in_size=node_dim, out_size=hidden_dim)
            rest_layers = [links.Linear(in_size=hidden_dim, out_size=hidden_dim) for _ in range(num_hidden_layers - 1)]
            linear_layers = [first_layer] + rest_layers
            self.hidden_layers = chainer.ChainList(
                *linear_layers
            )
            if num_hidden_layers == 0:
                self.output_layer = links.Linear(in_size=node_dim, out_size=output_dim)
            else:
                self.output_layer = links.Linear(in_size=hidden_dim, out_size=output_dim)

        self.node_dim = node_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.activation = activation

    def __call__(self, input_variable):
        h_nn = input_variable
        for i in range(self.num_hidden_layers):
            h_nn = self.activation(self.hidden_layers[i](h_nn))

            if self.dropout is not None:
                h_nn = functions.dropout(h_nn, ratio=self.dropout)

        # h_nn: (mb, 64)
        nn_output = self.output_layer(h_nn)
        return nn_output
