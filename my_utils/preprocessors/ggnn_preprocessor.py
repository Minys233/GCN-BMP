#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/18/2019 1:38 PM
# @Author  : chinshin
# @FileName: ggnn_preprocessor.py
from __future__ import unicode_literals
from collections import defaultdict
import numpy as np
from rdkit import Chem
from chainer_chemistry.dataset.preprocessors.common \
    import construct_atomic_number_array, construct_discrete_edge_matrix, MolFeatureExtractionError
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms
from chainer_chemistry.dataset.preprocessors.mol_preprocessor \
    import MolPreprocessor


class MyGGNNPreprocessor(MolPreprocessor):
    """GGNN Preprocessor

    Args:
        max_atoms (int): Max number of atoms for each molecule, if the
            number of atoms is more than this value, this data is simply
            ignored.
            Setting negative value indicates no limit for max atoms.
        out_size (int): It specifies the size of array returned by
            `get_input_features`.
            If the number of atoms in the molecule is less than this value,
            the returned arrays is padded to have fixed size.
            Setting negative value indicates do not pad returned array.
        add_Hs (bool): If True, implicit Hs are added.
        kekulize (bool): If True, Kekulizes the molecule.

    """
    def construct_dict(self):
        # global dict for index-based encoding
        self.atom_dict = defaultdict(lambda: len(self.atom_dict))
        self.bond_dict = defaultdict(lambda: len(self.bond_dict))
        self.fingerprint_dict = defaultdict(lambda: len(self.fingerprint_dict))
        self.edge_dict = defaultdict(lambda: len(self.edge_dict))

    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=False,
                 kekulize=False, radius=0):
        super(MyGGNNPreprocessor, self).__init__(
            add_Hs=add_Hs, kekulize=kekulize)
        if 0 <= max_atoms < out_size and out_size >= 0:
            raise ValueError('max_atoms {} must be less or equal to '
                             'out_size {}'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size
        self.radius = radius

        self.construct_dict()

    # def get_input_features(self, mol):
    #     """
    #     get input features
    #     Input feature contains: atom array and adjacency matrix
    #     Args:
    #         mol (Mol):
    #
    #     Returns:
    #
    #     """
    #     type_check_num_atoms(mol, self.max_atoms)
    #     atom_array = construct_atomic_number_array(mol, out_size=self.out_size)
    #     adj_array = construct_discrete_edge_matrix(mol, out_size=self.out_size)
    #     return atom_array, adj_array

    def get_input_features(self, mol):

        type_check_num_atoms(mol, self.max_atoms)

        atoms = self.create_atoms(mol)
        i_jbond_dict = self.create_ijbonddict(mol)

        subgraph_array = self.extract_subgraph(atoms, i_jbond_dict, self.radius)
        adj_array = construct_discrete_edge_matrix(mol)

        return subgraph_array, adj_array

    def create_atoms(self, mol):
        """
        Create a list of atom IDs considering the aromaticity
        :param mol: rdkit.Chem.Mol object
        :return:
        """
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        for a in mol.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], "aromatic")
        atoms = [self.atom_dict[a] for a in atoms]
        return np.array(atoms, dtype=np.int32)

    def create_ijbonddict(self, mol):
        """
        Create a dictionary, which each key is a node ID
        and each value is the tuples of its neighboring node
        and bond IDs.
        :param mol: rdkit.Chem.Mol object
        :return: i_jbond_dict
        """
        i_jbond_dict = defaultdict(lambda: [])
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = self.bond_dict[str(b.GetBondType())]
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        return i_jbond_dict

    def extract_subgraph(self, atoms, i_jbond_dict, radius):
        """
        Extract the r-radius subgraphs from a molecular graph using WL algorithm
        :param atoms:
        :param i_jbond_dict:
        :param radius:
        :return:
        """
        if (len(atoms) == 1) or (radius == 0):
            fingerprints = [self.fingerprint_dict[a] for a in atoms]

        else:
            nodes = atoms
            i_jedge_dict = i_jbond_dict

            for _ in range(radius):

                """Update each node ID considering its neighboring nodes and edges
                (i.e., r-radius subgraphs or fingerprints)."""
                fingerprints = []
                for i, j_edge in i_jedge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    fingerprints.append(self.fingerprint_dict[fingerprint])
                nodes = fingerprints

                """Also update each edge ID considering two nodes
                on its both sides."""
                _i_jedge_dict = defaultdict(lambda: [])
                for i, j_edge in i_jedge_dict.items():
                    for j, edge in j_edge:
                        both_side = tuple(sorted((nodes[i], nodes[j])))
                        edge = self.edge_dict[(both_side, edge)]
                        _i_jedge_dict[i].append((j, edge))
                i_jedge_dict = _i_jedge_dict

        return np.array(fingerprints, dtype=np.int32)

    @staticmethod
    def create_adjacency(mol):
        """
        :param mol: rdkit.Chem.Mol object
        :return:
        """
        adjacency = Chem.GetAdjacencyMatrix(mol)
        return np.array(adjacency, dtype=np.int32)


