from chainer_chemistry.dataset.preprocessors.common import construct_adj_matrix
from chainer_chemistry.dataset.preprocessors.common \
    import construct_atomic_number_array
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms
from chainer_chemistry.dataset.preprocessors.common \
    import construct_atomic_number_array, construct_discrete_edge_matrix, MolFeatureExtractionError
from chainer_chemistry.dataset.preprocessors.mol_preprocessor \
    import MolPreprocessor

import numpy as np
from rdkit import Chem


degrees = [0, 1, 2, 3, 4, 5, 6]
allowable_set = degrees

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return map(lambda s: x == s, allowable_set)


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return map(lambda s: x == s, allowable_set)


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                           'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                           'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',  # H?
                                           'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                           'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), allowable_set) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), allowable_set) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), allowable_set) +
                    [atom.GetIsAromatic()])


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])


class MolGraph(object):
    def __init__(self):
        self.nodes = {}  # dict of lists of nodes, keyed by node type

    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))

    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i: [] for i in degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)

        self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes['atom']])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n: i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]


class Node(object):
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']

    def __init__(self, ntype, features, rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix

    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)

    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]


class DrugFPPreprocessor(MolPreprocessor):

    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=False,
                 kekulize=False):
        super(DrugFPPreprocessor, self).__init__(
            add_Hs=add_Hs, kekulize=kekulize)
        if max_atoms >= 0 and out_size >= 0 and max_atoms > out_size:
            raise ValueError('max_atoms {} must be less or equal to '
                             'out_size {}'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size

    def get_input_features(self, mol):
        """get input features

        Args:
            mol (Mol):

        Returns: (atom, feature_dim)

        """
        # type_check_num_atoms(mol, self.max_atoms)
        # atom_array = construct_atomic_number_array(mol, out_size=self.out_size)
        # adj_array = construct_adj_matrix(mol, out_size=self.out_size)
        # return atom_array, adj_array
        graph = MolGraph()
        atoms_by_rd_idx = {}
        for atom in mol.GetAtoms():
            new_atom_node = graph.new_node('atom', features=atom_features(atom), rdkit_ix=atom.GetIdx())
            atoms_by_rd_idx[atom.GetIdx()] = new_atom_node

        for bond in mol.GetBonds():
            atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
            atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
            new_bond_node = graph.new_node('bond', features=bond_features(bond))
            new_bond_node.add_neighbors((atom1_node, atom2_node))
            atom1_node.add_neighbors((atom2_node,))

        mol_node = graph.new_node('molecule')
        mol_node.add_neighbors(graph.nodes['atom'])

        atom_array = graph.feature_array('atom')
        atom_array = atom_array.astype(dtype=np.float32)
        adj_array = construct_discrete_edge_matrix(mol, out_size=self.out_size)
        return atom_array, adj_array
