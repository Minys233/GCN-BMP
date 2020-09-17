import chainer
from chainer import cuda
from chainer import functions

import chainer_chemistry
from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links import EmbedAtomID, GraphLinear

class GGNNReadout(chainer.Chain):
    """GGNN submodule for readout part.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector associated to
            each atom
        nobias (bool): If ``True``, then this function does not use
            the bias
        activation (~chainer.Function or ~chainer.FunctionNode):
            activate function for node representation
            `functions.tanh` was suggested in original paper.
        activation_agg (~chainer.Function or ~chainer.FunctionNode):
            activate function for aggregation
            `functions.tanh` was suggested in original paper.
    """

    def __init__(self, out_dim, hidden_dim=16, nobias=False,
                 activation=functions.identity,
                 activation_agg=functions.identity):
        super(GGNNReadout, self).__init__()
        with self.init_scope():
            self.i_layer = GraphLinear(None, out_dim, nobias=nobias)
            self.j_layer = GraphLinear(None, out_dim, nobias=nobias)
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.nobias = nobias
        self.activation = activation
        self.activation_agg = activation_agg

    def __call__(self, h, h0=None, is_real_node=None):
        # --- Readout part ---
        # h, h0: (minibatch, node, ch)
        # is_real_node: (minibatch, node)
        h1 = functions.concat((h, h0), axis=2) if h0 is not None else h

        g1 = functions.sigmoid(self.i_layer(h1))
        g2 = self.activation(self.j_layer(h1))
        g = g1 * g2
        if is_real_node is not None:
            # mask virtual node feature to be 0
            mask = self.xp.broadcast_to(
                is_real_node[:, :, None], g.shape)
            g = g * mask
        # sum along node axis
        g = self.activation_agg(functions.sum(g, axis=1))
        return g


class GINUpdate(chainer.Chain):
    r"""GIN submodule for update part.

    Simplest implementation of Graph Isomorphism Network (GIN):
    2-layered MLP + ReLU
    no learnble epsilon

    Batch Normalization is not implemetned. instead we use droout

    # TODO: implement Batch Normalization
    # TODO: use GraphMLP instead of GraphLinears

    See: Xu, Hu, Leskovec, and Jegelka, \
        "How powerful are graph neural networks?", in ICLR 2019.

    Args:
        hidden_dim (int): dimension of feature vector associated to
            each atom
        dropout_ratio (float): ratio of dropout, insted of bach normlization
    """

    def __init__(self, hidden_dim=16, dropout_ratio=0.5):
        super(GINUpdate, self).__init__()
        with self.init_scope():
            # two Linear + RELU
            self.linear_g1 = GraphLinear(hidden_dim, hidden_dim)
            self.linear_g2 = GraphLinear(hidden_dim, hidden_dim)
        # end with
        self.dropout_ratio = dropout_ratio
    # end-def

    def __call__(self, h, adj):
        """
        Describing a layer.

        Args:
            h (numpy.ndarray): minibatch by num_nodes by hidden_dim
                numpy array. local node hidden states
            adj (numpy.ndarray): minibatch by num_nodes by num_nodes 1/0 array.
                Adjacency matrices over several bond types

        Returns:
            updated h

        """

        # (minibatch, atom, ch)
        mb, atom, ch = h.shape

        # --- Message part ---
        # adj (mb, atom, atom)
        # fv   (minibatch, atom, ch)
        adj = functions.sum(adj, axis=1)
        fv = chainer_chemistry.functions.matmul(adj, h)
        assert (fv.shape == (mb, atom, ch))

        # sum myself
        sum_h = fv + h
        assert (sum_h.shape == (mb, atom, ch))

        # apply MLP
        new_h = functions.relu(self.linear_g1(sum_h))
        if self.dropout_ratio > 0.0:
            new_h = functions.relu(
                functions.dropout(
                    self.linear_g2(new_h), ratio=self.dropout_ratio))
        else:
            new_h = functions.relu(self.linear_g2(new_h))

        # done???
        return new_h


class GIN(chainer.Chain):
    """
    Simplest implementation of Graph Isomorphism Network (GIN)

    See: Xu, Hu, Leskovec, and Jegelka, \
    "How powerful are graph neural networks?", in ICLR 2019.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (default=16): dimension of hidden vectors
            associated to each atom
        n_layers (default=4): number of layers
        n_atom_types: number of atoms
        dropout_ratio (default=0.5); if > 0.0, perform dropout
        concat_hidden (default=False): If set to True, readout is executed in
            each layer and the result is concatenated
        weight_tying (default=True): enable weight_tying for all units


    """
    NUM_EDGE_TYPE = 4

    def __init__(self, out_dim, hidden_dim=16,
                 n_layers=4, n_atom_types=MAX_ATOMIC_NUM,
                 dropout_ratio=0.5,
                 concat_hidden=False,
                 weight_tying=True,
                 activation=functions.identity):
        super(GIN, self).__init__()

        n_message_layer = 1 if weight_tying else n_layers
        n_readout_layer = n_layers if concat_hidden else 1
        with self.init_scope():
            # embedding
            self.embed = EmbedAtomID(out_size=hidden_dim, in_size=n_atom_types)

            # two non-linear MLP part
            self.update_layers = chainer.ChainList(*[GINUpdate(
                hidden_dim=hidden_dim, dropout_ratio=dropout_ratio)
                for _ in range(n_message_layer)])

            # Readout
            self.readout_layers = chainer.ChainList(*[GGNNReadout(
                out_dim=out_dim, hidden_dim=hidden_dim,
                activation=activation, activation_agg=activation)
                for _ in range(n_readout_layer)])
        # end with

        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_message_layers = n_message_layer
        self.n_readout_layer = n_readout_layer
        self.dropout_ratio = dropout_ratio
        self.concat_hidden = concat_hidden
        self.weight_tying = weight_tying

    def __call__(self, atom_array, adj, is_real_node=None):
        """
        Describe the whole forwar path

        Args:
            atom_array (numpy.ndarray): mol-minibatch by node numpy.ndarray,
                minibatch of molecular which is represented with atom IDs
                (representing C, O, S, ...) atom_array[m, i] = a represents
                m-th molecule's i-th node is value a (atomic number)
            adj (numpy.ndarray): mol-minibatch by relation-types by node by
                node numpy.ndarray,
                minibatch of multple relational adjancency matrix with
                edge-type information adj[i, j] = b represents
                m-th molecule's  edge from node i to node j has value b
            is_real_node:

        Returns:
            numpy.ndarray: final molecule representation
        """

        if atom_array.dtype == self.xp.int32:
            h = self.embed(atom_array)  # (minibatch, max_num_atoms)
        else:
            h = atom_array

        h0 = functions.copy(h, cuda.get_device_from_array(h.data).id)

        g_list = []
        for step in range(self.n_message_layers):
            message_layer_index = 0 if self.weight_tying else step
            h = self.update_layers[message_layer_index](h, adj)
            if self.concat_hidden:
                g = self.readout_layers[step](h, h0, is_real_node)
                g_list.append(g)

        if self.concat_hidden:
            return functions.concat(g_list, axis=1)
        else:
            g = self.readout_layers[0](h, h0, is_real_node)
            return g
