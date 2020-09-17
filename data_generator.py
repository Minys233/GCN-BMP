#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/20/2019 6:47 PM
# @Author  : chinshin
# @FileName: data_generator.py
"""
generate data under different splitters

splitters are as follows:
    drug-based
        random split:
        scaffold split
    interaction-based:
        random split
        random split with different scales


dataset:
    input:
        add_zinc_smiles.txt: contains a list of drugs whose total number is 548.
        drug_drug_matrix: contains known interactions between differnet drugs.
    output:
        ddi_train.csv: train dataset containing multiple DDIs.
        ddi_test.csv: test datast containing multiple DDIs.
        index_train.txt: optional for drug-based splitter
        index_test.txt: optional for drug-based splitter
"""

import os
import csv
import sys
import copy
import random
import logging
import argparse
import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict
from rdkit.Chem import MolFromSmiles
from os.path import abspath, dirname
from sklearn.model_selection import train_test_split
from chainer_chemistry.dataset.splitters import ScaffoldSplitter, RandomSplitter, StratifiedSplitter
from deepchem.feat import WeaveFeaturizer

logging.basicConfig(format='%(asctime)s: %(filename)s: %(funcName)s: %(lineno)d: %(message)s', level=logging.INFO)
ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_PATH)
global_seed = 2018
random.seed(global_seed)

from utils import load_csv, index2id, is_number
from setting import *


def add_super_nodes(graph_distance=True, explicit_H=False,
                    use_chirality=False):
    """
    add super nodes for each drug molecule.
    feature vector incoporates original information used in MPNN.
    """
    drug_list_filename = 'drug_list.csv'
    drug_list_filepath = os.path.join(DRUG_LIST_PATH, drug_list_filename)
    df = pd.read_csv(drug_list_filepath)
    smiles_list = df['smiles']

    mol_list = [MolFromSmiles(smiles) for smiles in smiles_list]
    featurizer = WeaveFeaturizer(graph_distance=graph_distance,
                                 explicit_H=explicit_H, use_chirality=use_chirality)
    weave_mol_list = featurizer.featurize(mol_list)
    atom_feat_list = [mol.get_atom_features().sum(axis=0) for mol in weave_mol_list]

    mean_atom_feat_list = [mol.get_atom_features().mean(axis=0) for mol in weave_mol_list]
    max_atom_feat_list = [mol.get_atom_features().max(axis=0) for mol in weave_mol_list]
    atom_feat_list = np.concatenate(
        (atom_feat_list, mean_atom_feat_list, max_atom_feat_list), axis=1)
    atom_feat_list = np.concatenate(
        (atom_feat_list, np.zeros((NUM_DRUGS, 244 - 75 * 3))), axis=1
    )

    smiles2atom_feat = dict()
    for smiles, atom_feat in zip(smiles_list, atom_feat_list):
        smiles2atom_feat[smiles] = atom_feat

    filename = 'super_nodes.pkl'
    filepath = os.path.join(SUPER_NODE_PATH, filename)
    with open(filepath, 'w') as writer:
        pickle.dump(smiles2atom_feat, writer)

    print('Super nodes have been generated in path {}'.format(filepath))
    # return smiles2atom_feat


def add_super_nodes2(n_super_features=244, mean=0.0, scale=1.0):
    MAX_ATOMIC_NUM = 117
    w = np.random.normal(loc=mean, scale=scale, size=(MAX_ATOMIC_NUM, n_super_features))

    drug_list_filename = 'drug_list.csv'
    drug_list_filepath = os.path.join(DRUG_LIST_PATH, drug_list_filename)
    df = pd.read_csv(drug_list_filepath)
    smiles_list = df['smiles']

    mol_list = [MolFromSmiles(smiles) for smiles in smiles_list]
    super_node_list = list()
    for mol in mol_list:
        atomic_num_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        atomic_embedding_list = [w[atomic_num - 1] for atomic_num in atomic_num_list]
        super_node_feat = np.array(atomic_embedding_list, dtype=np.float32).mean(axis=0)
        super_node_list.append(super_node_feat)

    smiles2super_node = dict()
    for smiles, super_node_feat in zip(smiles_list, super_node_list):
        smiles2super_node[smiles] = super_node_feat

    filename = 'super_nodes_random_embedding.pkl'
    filepath = os.path.join(SUPER_NODE_PATH, filename)
    with open(filepath, 'w') as writer:
        pickle.dump(smiles2super_node, writer)

    print('Super nodes have been generated in path {}'.format(filepath))


def load_super_nodes(filename='super_nodes.txt'):
    """
    add super nodes for each drug molecule.
    feature vector incoporates original information used in MPNN.
    """
    filepath = os.path.join(ROOT_PATH, 'examples', 'ddi', filename)
    smiles_list = list()
    atom_feat_list = list()
    with open(filepath, 'r') as reader:
        for line in reader.readlines():
            line = line.strip('\n')
            smiles, feat_str = line.split('\t')[0], line.split('\t')[-1]
            feat = [float(digit) for digit in feat_str.split(' ') if is_number(digit)]
            smiles_list.append(smiles_list)
            atom_feat_list.append(feat)
    return smiles_list, atom_feat_list


def generate_drug_list():
    filename = 'drug_list_copy.csv'
    filepath = os.path.join(DRUG_LIST_PATH, filename)
    df = pd.read_csv(filepath)

    data = list()
    for row_id, row_series in df.iterrows():
        row_dict = dict(row_series)
        row_dict.pop('Unnamed: 0')
        if MolFromSmiles(row_dict['smiles']) is not None:
            data.append(row_dict)

    new_filename = 'drug_list.csv'
    new_filepath = os.path.join(DRUG_LIST_PATH, new_filename)
    new_df = pd.DataFrame(data=data)
    new_df.to_csv(new_filepath)

    new_df = pd.read_csv(new_filepath)
    assert sum([MolFromSmiles(smiles) is None for smiles in new_df['smiles']]) == 0


def validate_drug_list():
    filename = 'drug_list.csv'
    filepath = os.path.join(DRUG_LIST_PATH, filename)
    df = pd.read_csv(filepath)
    assert sum([MolFromSmiles(smiles) is None for smiles in df['smiles']]) == 0


class Splitter(object):

    def __init__(self):
        drug_list_filepath = os.path.join(DRUG_LIST_PATH, 'drug_list.csv')
        self.drug_list_df = pd.read_csv(drug_list_filepath)
        drug_drug_matrix_filename = 'drug_drug_matrix.csv'
        drug_drug_matrix_filepath = os.path.join(GROUND_TRUTH_PATH, drug_drug_matrix_filename)
        self.drug_drug_matrix_df = pd.read_csv(drug_drug_matrix_filepath)

    def __generate_data_and_labels(self):
        drug_list_df = self.drug_list_df
        drug_drug_matrix_df = self.drug_drug_matrix_df

        pairs = list()
        labels = list()
        columns = drug_drug_matrix_df.columns.values[1:]
        columns_from_drug_list = drug_list_df['cid'].values
        assert len(columns) == NUM_DRUGS
        assert list(columns) == list(columns_from_drug_list)
        for row_id, row_series in drug_drug_matrix_df.iterrows():
            row_cid = columns[row_id]
            for col_id, col_cid in enumerate(columns):
                if col_id > row_id:
                    pairs.append((row_cid, col_cid))
                    labels.append(int(row_series.loc[col_cid]))
        pairs = np.array(pairs)
        labels = np.array(labels)
        assert len(pairs) == NUM_INTERACTIONS
        assert len(labels) == NUM_INTERACTIONS
        return pairs, labels

    def stat(self, pairs, labels):
        num_pos = np.sum(labels == 1)
        num_neg = np.sum(labels == 0)
        ratio = float(num_pos) / float(num_neg)
        print('pos: {}, neg: {}, ratio: {}'.format(num_pos, num_neg, ratio))

    def __write_to_disk(self, split_and_dataset):

        for split, item in split_and_dataset.iteritems():
            dataset, filepath = item
            pairs, labels = dataset

            df = self.drug_list_df
            cid2drugbank_id = dict(zip(df['cid'], df['drugbank_id']))
            cid2smiles = dict(zip(df['cid'], df['smiles']))

            data = list()
            for pair, label in zip(pairs, labels):
                cid_1, cid_2 = pair
                dbid_1 = cid2drugbank_id[cid_1]
                dbid_2 = cid2drugbank_id[cid_2]
                smiles_1 = cid2smiles[cid_1]
                smiles_2 = cid2smiles[cid_2]
                item = OrderedDict({
                    'cid_1': cid_1,
                    'cid_2': cid_2,
                    'drugbank_id_1': dbid_1,
                    'drugbank_id_2': dbid_2,
                    'smiles_1': smiles_1,
                    'smiles_2': smiles_2,
                    'label': label
                })
                data.append(item)

            df_train = pd.DataFrame(data=data)
            df_train.to_csv(filepath)

            print('{} dataset generated.'.format(split[0].upper() + split[1:], format))

    def random_split_based_drug(self, train_filepath, valid_filepath, test_filepath,
                                frac_train=0.8, frac_valid=0.1, frac_test=0.1):
        """
        Splitter based on distinct drug.
        Corresponding splitter about DDI are based on the splitted drugs.
        """
        pairs, labels = self.__generate_data_and_labels()
        drug_list_df = copy.deepcopy(self.drug_list_df)
        cid_arr = drug_list_df['cid'].values

        ss = RandomSplitter()
        train_inds, valid_inds, test_inds = ss.train_valid_test_split(dataset=drug_list_df,
                                                                      frac_train=frac_train, frac_valid=frac_valid,
                                                                      frac_test=frac_test,
                                                                      return_index=True, seed=GLOBAL_SEED)
        assert len(train_inds) + len(valid_inds) + len(test_inds) == NUM_DRUGS
        cids_train, cids_valid, cids_test = cid_arr[train_inds], cid_arr[valid_inds], cid_arr[test_inds]

        train_inds_filepath = os.path.join(
            os.path.dirname(train_filepath),
            os.path.basename(train_filepath).split('.')[0] + '_ind' + '.csv')
        valid_inds_filepath = os.path.join(
            os.path.dirname(valid_filepath),
            os.path.basename(valid_filepath).split('.')[0] + '_ind' + '.csv')
        test_inds_filepath = os.path.join(
            os.path.dirname(test_filepath),
            os.path.basename(test_filepath).split('.')[0] + '_ind' + '.csv')

        # row selection
        df_train = drug_list_df.loc[train_inds]
        del df_train['Unnamed: 0']
        df_valid = drug_list_df.loc[valid_inds]
        del df_valid['Unnamed: 0']
        df_test = drug_list_df.loc[test_inds]
        del df_test['Unnamed: 0']

        df_train.to_csv(train_inds_filepath)
        df_valid.to_csv(valid_inds_filepath)
        df_test.to_csv(test_inds_filepath)

        pairs_train, labels_train = list(), list()
        pairs_valid, labels_valid = list(), list()
        pairs_test, labels_test = list(), list()
        for pair, label in zip(pairs, labels):
            cid_1, cid_2 = pair
            # train dataset generation
            if cid_1 in cids_train and cid_2 in cids_train:
                pairs_train.append((cid_1, cid_2))
                labels_train.append(label)
            # valid dataset generation
            elif (cid_1 in cids_train and cid_2 in cids_valid) or \
                    (cid_1 in cids_valid and cid_2 in cids_train):
                pairs_valid.append((cid_1, cid_2))
                labels_valid.append(label)
            # test dataset generation
            elif (cid_1 in cids_train and cid_2 in cids_test) or \
                    (cid_1 in cids_test and cid_2 in cids_train):
                pairs_test.append((cid_1, cid_2))
                labels_test.append(label)

        pairs_train = np.array(pairs_train)
        labels_train = np.array(labels_train)
        pairs_valid = np.array(pairs_valid)
        labels_valid = np.array(labels_valid)
        pairs_test = np.array(pairs_test)
        labels_test = np.array(labels_test)

        # statistics
        # train: num_total, num_pos, num_neg
        # test: num_total, num_pos, num_neg
        num_total_train = labels_train.shape[0]
        num_pos_train = np.sum(labels_train == 1)
        num_neg_train = np.sum(labels_train == 0)
        num_total_valid = labels_valid.shape[0]
        num_pos_valid = np.sum(labels_valid == 1)
        num_neg_valid = np.sum(labels_valid == 0)
        num_total_test = labels_test.shape[0]
        num_pos_test = np.sum(labels_test == 1)
        num_neg_test = np.sum(labels_test == 0)
        print('Statistics: ')
        print('Train# total: {}, pos: {}, neg: {}'.format(num_total_train, num_pos_train, num_neg_train))
        print('Valid# total: {}, pos: {}, neg: {}'.format(num_total_valid, num_pos_valid, num_neg_valid))
        print('Test # total: {}, pos: {}, neg: {}'.format(num_total_test, num_pos_test, num_neg_test))

        split_and_dataset = {
            'train': [(pairs_train, labels_train), train_filepath],
            'valid': [(pairs_valid, labels_valid), valid_filepath],
            'test': [(pairs_test, labels_test), test_filepath],
        }
        # write train dataset and test dataset into disk
        self.__write_to_disk(split_and_dataset)

    def scaffold_split_based_drug(self, train_filepath, valid_filepath, test_filepath,
                                  frac_train=0.8, frac_valid=0.1, frac_test=0.1):
        """
        We need to delete two drugs:
        CID000004609, DB00526
        [H][N]1([H])[C@@H]2CCCC[C@H]2[N]([H])([H])[Pt]11OC(=O)C(=O)O1
        CID000060754, DB00225
        O=C1[O-][Gd+3]234567[O]=C(C[N]2(CC[N]3(CC([O-]4)=O)CC[N]5(CC(=[O]6)NC)CC(=O)[O-]7)C1)NC
        """
        pairs, labels = self.__generate_data_and_labels()

        drug_list_df = copy.deepcopy(self.drug_list_df)

        db_ids_removed = ['DB00526', 'DB00225']
        drug_list_df.drop(
            drug_list_df.loc[drug_list_df['drugbank_id'].isin(db_ids_removed)].index)

        assert sum([db_id in drug_list_df['drugbank_id'] for db_id in db_ids_removed]) == 0

        cid_arr = drug_list_df['cid'].values
        smiles_arr = drug_list_df['smiles'].values
        ss = ScaffoldSplitter()

        train_inds, valid_inds, test_inds = ss.train_valid_test_split(drug_list_df, smiles_arr,
                                                                      frac_train=frac_train, frac_valid=frac_valid,
                                                                      frac_test=frac_test,
                                                                      return_index=True)
        assert len(train_inds) + len(valid_inds) + len(test_inds) == NUM_DRUGS
        cids_train, cids_valid, cids_test = cid_arr[train_inds], cid_arr[valid_inds], cid_arr[test_inds]

        train_inds_filepath = os.path.join(
            os.path.dirname(train_filepath),
            os.path.basename(train_filepath).split('.')[0] + '_ind' + '.csv')
        valid_inds_filepath = os.path.join(
            os.path.dirname(valid_filepath),
            os.path.basename(valid_filepath).split('.')[0] + '_ind' + '.csv')
        test_inds_filepath = os.path.join(
            os.path.dirname(test_filepath),
            os.path.basename(test_filepath).split('.')[0] + '_ind' + '.csv')

        df_train = drug_list_df.loc[train_inds]
        # del df_train['Unnamed: 0']
        df_valid = drug_list_df.loc[valid_inds]
        # del df_valid['Unnamed: 0']
        df_test = drug_list_df.loc[test_inds]
        # del df_valid['Unnamed: 0']

        df_train.to_csv(train_inds_filepath, index=False)
        df_valid.to_csv(valid_inds_filepath, index=False)
        df_test.to_csv(test_inds_filepath)

        pairs_train, labels_train = list(), list()
        pairs_valid, labels_valid = list(), list()
        pairs_test, labels_test = list(), list()
        for pair, label in zip(pairs, labels):
            cid_1, cid_2 = pair
            # train dataset generation
            if cid_1 in cids_train and cid_2 in cids_train:
                pairs_train.append((cid_1, cid_2))
                labels_train.append(label)
            # valid dataset generation
            elif (cid_1 in cids_train and cid_2 in cids_valid) or \
                    (cid_2 in cids_train and cid_1 in cids_valid):
                pairs_valid.append((cid_1, cid_2))
                labels_valid.append(label)
            # test dataset generation
            elif (cid_1 in cids_train and cid_2 in cids_test) or \
                    (cid_1 in cids_test and cid_2 in cids_train):
                pairs_test.append((cid_1, cid_2))
                labels_test.append(label)

        pairs_train = np.array(pairs_train)
        labels_train = np.array(labels_train)
        pairs_valid = np.array(pairs_valid)
        labels_valid = np.array(labels_valid)
        pairs_test = np.array(pairs_test)
        labels_test = np.array(labels_test)

        # statistics
        # train: num_total, num_pos, num_neg
        # test: num_total, num_pos, num_neg
        num_total_train = labels_train.shape[0]
        num_pos_train = np.sum(labels_train == 1)
        num_neg_train = np.sum(labels_train == 0)
        num_total_valid = labels_valid.shape[0]
        num_pos_valid = np.sum(labels_valid == 1)
        num_neg_valid = np.sum(labels_valid == 0)
        num_total_test = labels_test.shape[0]
        num_pos_test = np.sum(labels_test == 1)
        num_neg_test = np.sum(labels_test == 0)
        print('Statistics: ')
        print('Train# total: {}, pos: {}, neg: {}'.format(num_total_train, num_pos_train, num_neg_train))
        print('Valid# total: {}, pos: {}, neg: {}'.format(num_total_valid, num_pos_valid, num_neg_valid))
        print('Test # total: {}, pos: {}, neg: {}'.format(num_total_test, num_pos_test, num_neg_test))

        split_and_dataset = {
            'train': [(pairs_train, labels_train), train_filepath],
            'valid': [(pairs_valid, labels_valid), valid_filepath],
            'test': [(pairs_test, labels_test), test_filepath],
        }
        # write train dataset and test dataset into disk
        self.__write_to_disk(split_and_dataset)

    def random_split_based_interaction(self, train_filepath, valid_filepath, test_filepath,
                                       frac_train=0.8, frac_valid=0.1, frac_test=0.1):

        pairs, labels = self.__generate_data_and_labels()

        ss = StratifiedSplitter()
        train_inds, valid_inds, test_inds = ss.train_valid_test_split(
            dataset=pairs, labels=labels,
            frac_train=frac_train, frac_valid=frac_valid, frac_test=frac_test,
            return_index=True, seed=GLOBAL_SEED)
        assert len(train_inds) + len(valid_inds) + len(test_inds) == NUM_INTERACTIONS

        pairs_train, pairs_valid, pairs_test = pairs[train_inds], pairs[valid_inds], pairs[test_inds]
        labels_train, labels_valid, labels_test = labels[train_inds], labels[valid_inds], labels[test_inds]

        ratio_train = (float(np.sum(labels_train == 1)) / float(np.sum(labels_train == 0)))
        ratio_valid = (float(np.sum(labels_valid == 1)) / float(np.sum(labels_valid == 0)))
        ratio_test = (float(np.sum(labels_test == 1)) / float(np.sum(labels_test == 0)))
        ratio = (float(np.sum(labels == 1))) / float(np.sum(labels == 0))

        assert int(100 * ratio) == int(100 * ratio_train) == int(100 * ratio_valid) == int(100 * ratio_test)

        # statistics
        # train: num_total, num_pos, num_neg
        # test: num_total, num_pos, num_neg
        num_total_train = labels_train.shape[0]
        num_pos_train = np.sum(labels_train == 1)
        num_neg_train = np.sum(labels_train == 0)
        num_total_valid = labels_valid.shape[0]
        num_pos_valid = np.sum(labels_valid == 1)
        num_neg_valid = np.sum(labels_valid == 0)
        num_total_test = labels_test.shape[0]
        num_pos_test = np.sum(labels_test == 1)
        num_neg_test = np.sum(labels_test == 0)
        print('Statistics: ')
        print('Train# total: {}, pos: {}, neg: {}'.format(num_total_train, num_pos_train, num_neg_train))
        print('Valid# total: {}, pos: {}, neg: {}'.format(num_total_valid, num_pos_valid, num_neg_valid))
        print('Test # total: {}, pos: {}, neg: {}'.format(num_total_test, num_pos_test, num_neg_test))

        split_and_dataset = {
            'train': [(pairs_train, labels_train), train_filepath],
            'valid': [(pairs_valid, labels_valid), valid_filepath],
            'test': [(pairs_test, labels_test), test_filepath],
        }
        # write train dataset and test dataset into disk.
        self.__write_to_disk(split_and_dataset)

    def random_split_based_interaction_equal(self, train_filepath, valid_filepath, test_filepath,
                                            frac_train=0.8, frac_valid=0.1, frac_test=0.1):
        pairs, labels = self.__generate_data_and_labels()

        ss = StratifiedSplitter()
        train_inds, valid_inds, test_inds = ss.train_valid_test_split(
            dataset=pairs, labels=labels,
            frac_train=frac_train, frac_valid=frac_valid, frac_test=frac_test,
            return_index=True)
        assert len(train_inds) + len(valid_inds) + len(test_inds) == NUM_INTERACTIONS

        pairs_train, pairs_valid, pairs_test = pairs[train_inds], pairs[valid_inds], pairs[test_inds]
        labels_train, labels_valid, labels_test = labels[train_inds], labels[valid_inds], labels[test_inds]

        ratio_train = (float(np.sum(labels_train == 1)) / float(np.sum(labels_train == 0)))
        ratio_valid = (float(np.sum(labels_valid == 1)) / float(np.sum(labels_valid == 0)))
        ratio_test = (float(np.sum(labels_test == 1)) / float(np.sum(labels_test == 0)))
        ratio = (float(np.sum(labels == 1))) / float(np.sum(labels == 0))

        assert int(100 * ratio) == int(100 * ratio_train) == int(100 * ratio_valid) == int(100 * ratio_test)

        pairs_train_pos = pairs_train[labels_train == 1]
        pairs_train_neg = pairs_train[labels_train == 0]
        num_pos = np.sum(labels_train == 1)
        num_neg = np.sum(labels_train == 0)
        scale = np.sum(labels_train == 1)
        indices_pos = np.arange(0, num_pos)
        indices_pos = np.random.choice(indices_pos, size=scale, replace=False)
        pairs_train_pos = pairs_train_pos[indices_pos]
        indices_neg = np.arange(0, num_neg)
        indices_neg = np.random.choice(indices_neg, size=scale, replace=False)
        pairs_train_neg = pairs_train_neg[indices_neg]
        pairs_train = np.concatenate((pairs_train_pos, pairs_train_neg), axis=0)
        labels_train = np.concatenate(
            (np.ones(shape=(scale,)), np.zeros(shape=(scale,))), axis=0)
        indices = np.arange(0, 2 * scale)
        np.random.seed(GLOBAL_SEED)
        np.random.shuffle(indices)
        pairs_train = pairs_train[indices]
        labels_train = labels_train[indices]
        assert len(pairs_train) == len(labels_train) == 2 * scale

        # statistics
        # train: num_total, num_pos, num_neg
        # test: num_total, num_pos, num_neg
        num_total_train = labels_train.shape[0]
        num_pos_train = np.sum(labels_train == 1)
        num_neg_train = np.sum(labels_train == 0)
        num_total_valid = labels_valid.shape[0]
        num_pos_valid = np.sum(labels_valid == 1)
        num_neg_valid = np.sum(labels_valid == 0)
        num_total_test = labels_test.shape[0]
        num_pos_test = np.sum(labels_test == 1)
        num_neg_test = np.sum(labels_test == 0)
        print('Statistics: ')
        print('Train# total: {}, pos: {}, neg: {}'.format(num_total_train, num_pos_train, num_neg_train))
        print('Valid# total: {}, pos: {}, neg: {}'.format(num_total_valid, num_pos_valid, num_neg_valid))
        print('Test # total: {}, pos: {}, neg: {}'.format(num_total_test, num_pos_test, num_neg_test))

        split_and_dataset = {
            'train': [(pairs_train, labels_train), train_filepath],
            'valid': [(pairs_valid, labels_valid), valid_filepath],
            'test': [(pairs_test, labels_test), test_filepath],
        }
        # write train dataset and test dataset into disk.
        self.__write_to_disk(split_and_dataset)

    def random_split_based_interaction_different_scales(self, scale, train_filepath, valid_filepath, test_filepath,
                                                        frac_train=0.8, frac_valid=0.1, frac_test=0.1):
        pairs, labels = self.__generate_data_and_labels()

        ss = StratifiedSplitter()
        train_inds, valid_inds, test_inds = ss.train_valid_test_split(
            dataset=pairs, labels=labels,
            frac_train=frac_train, frac_valid=frac_valid, frac_test=frac_test,
            return_index=True)
        assert len(train_inds) + len(valid_inds) + len(test_inds) == NUM_INTERACTIONS

        pairs_train, pairs_valid, pairs_test = pairs[train_inds], pairs[valid_inds], pairs[test_inds]
        labels_train, labels_valid, labels_test = labels[train_inds], labels[valid_inds], labels[test_inds]

        ratio_train = (float(np.sum(labels_train == 1)) / float(np.sum(labels_train == 0)))
        ratio_valid = (float(np.sum(labels_valid == 1)) / float(np.sum(labels_valid == 0)))
        ratio_test = (float(np.sum(labels_test == 1)) / float(np.sum(labels_test == 0)))
        ratio = (float(np.sum(labels == 1))) / float(np.sum(labels == 0))

        assert int(100 * ratio) == int(100 * ratio_train) == int(100 * ratio_valid) == int(100 * ratio_test)

        pairs_train_pos = pairs_train[labels_train == 1]
        pairs_train_neg = pairs_train[labels_train == 0]
        num_pos = np.sum(labels_train == 1)
        num_neg = np.sum(labels_train == 0)
        assert scale <= np.sum(labels_train == 1) and scale <= np.sum(labels_train == 0)
        indices_pos = np.arange(0, num_pos)
        indices_pos = np.random.choice(indices_pos, size=scale, replace=False)
        pairs_train_pos = pairs_train_pos[indices_pos]
        indices_neg = np.arange(0, num_neg)
        indices_neg = np.random.choice(indices_neg, size=scale, replace=False)
        pairs_train_neg = pairs_train_neg[indices_neg]
        pairs_train = np.concatenate((pairs_train_pos, pairs_train_neg), axis=0)
        labels_train = np.concatenate(
            (np.ones(shape=(scale,)), np.zeros(shape=(scale,))), axis=0)
        indices = np.arange(0, 2 * scale)
        np.random.seed(GLOBAL_SEED)
        np.random.shuffle(indices)
        pairs_train = pairs_train[indices]
        labels_train = labels_train[indices]
        assert len(pairs_train) == len(labels_train) == 2 * scale

        # statistics
        # train: num_total, num_pos, num_neg
        # test: num_total, num_pos, num_neg
        num_total_train = labels_train.shape[0]
        num_pos_train = np.sum(labels_train == 1)
        num_neg_train = np.sum(labels_train == 0)
        num_total_valid = labels_valid.shape[0]
        num_pos_valid = np.sum(labels_valid == 1)
        num_neg_valid = np.sum(labels_valid == 0)
        num_total_test = labels_test.shape[0]
        num_pos_test = np.sum(labels_test == 1)
        num_neg_test = np.sum(labels_test == 0)
        print('Statistics: ')
        print('Train# total: {}, pos: {}, neg: {}'.format(num_total_train, num_pos_train, num_neg_train))
        print('Valid# total: {}, pos: {}, neg: {}'.format(num_total_valid, num_pos_valid, num_neg_valid))
        print('Test # total: {}, pos: {}, neg: {}'.format(num_total_test, num_pos_test, num_neg_test))

        split_and_dataset = {
            'train': [(pairs_train, labels_train), train_filepath],
            'valid': [(pairs_valid, labels_valid), valid_filepath],
            'test': [(pairs_test, labels_test), test_filepath],
        }
        # write train dataset and test dataset into disk.
        self.__write_to_disk(split_and_dataset)


class SymmetricPair(object):
    def __init__(self, former, latter):
        self.former = former
        self.latter = latter

    def __eq__(self, other):
        if self.former == other.former and self.latter == other.latter \
                or self.former == other.latter and self.latter == other.former:
            return True
        else:
            return False

    def __getitem__(self, item):
        if item == 0:
            return self.former
        elif item == 1:
            return self.latter
        else:
            raise ValueError('No the third element.')


class KaistSplitter(object):
    """

    Ouput:
        a list of drugs: drug_list.csv
        interaction: ddi_total.csv, ddi_train.csv, ddi_valid.csv, ddi_test.csv

    dict, key=(drugbank_id_1, drugbank_id_2), value=dict(labels=[], masks=[], descs=[]))
    item = {
        (drugbank_id_1, drugbank_id_2): {
            'labels': [labels_1, labels_2, ..., label_n],
            'masks':  [mask_1, mask_2, ..., mask_n],
            'descs':  [desc_1, desc_2, ..., desc_n],
        }
    }

    assert len(labels) == len(masks) == len(descs)

    the key named 'labels', 'masks' and 'descs' should be converted into the string format.
    the delimiter should be '||'

    先分别
    """

    def __init__(self):
        ddi_filename = 'original_ddi_total.csv'
        ddi_filepath = os.path.join(KAIST_PATH, ddi_filename)
        self.ddi_ori_df = pd.read_csv(ddi_filepath)

    def __generate_drug_list(self):

        ddi_df = self.ddi_ori_df
        db_id_1_list = list(ddi_df['Drug1'].values)
        db_id_2_list = list(ddi_df['Drug2'].values)
        db_id_list = list(set(db_id_1_list + db_id_2_list))
        print('Before preprocessing, num of drugs: {}'.format(len(db_id_list)))

        # eliminate the invalid drugs whose SMILES representation are unavailable or which can
        # not be converted into rdkit.Chem.Mol object.
        drug_list_filename = 'drug_list_from_drugbank_latest.csv'
        drug_list_filepath = os.path.join(DRUGBANK_PATH, drug_list_filename)
        drug_list_df = pd.read_csv(drug_list_filepath)
        db_id2smiles = dict(zip(drug_list_df['drugbank_id'], drug_list_df['smiles']))

        invalid_count = 0
        valid_db_id_list = list()
        for db_id in db_id_list:
            smiles = db_id2smiles.get(db_id, None)
            if smiles is None or MolFromSmiles(smiles) is None:
                invalid_count += 1
                print('Invalid drug: {}'.format(db_id))
                continue
            valid_db_id_list.append(db_id)
        print('Invalid count: {}'.format(invalid_count))

        kaist_drug_list = list()
        for row_id, row in drug_list_df.iterrows():
            db_id = row['drugbank_id']
            if db_id in valid_db_id_list:
                item = dict(row)
                kaist_drug_list.append(item)
        kaist_drug_df = pd.DataFrame(kaist_drug_list, columns=drug_list_df.columns.values)
        filename = 'drug_list.csv'
        filepath = os.path.join(KAIST_PATH, filename)
        kaist_drug_df.to_csv(filepath)

        assert len(kaist_drug_df) == NUM_DRUGS_KAIST
        print('After preprocessing, num of drugs: {}'.format(len(kaist_drug_df)))

    def split_train_valid_split(self):
        ddi_ori_df = self.ddi_ori_df
        # generate python dictionary to convert DBID to SMILES representation.
        # the conversion is always successful.
        drug_list_filename = 'drug_list.csv'
        drug_list_filepath = os.path.join(KAIST_PATH, drug_list_filename)
        if not os.path.exists(drug_list_filepath):
            self.__generate_drug_list()
        drug_list_df = pd.read_csv(drug_list_filepath)
        db_id2smiles = dict(zip(drug_list_df['drugbank_id'], drug_list_df['smiles']))

        # dict, key=(drugbank_id_1, drugbank_id_2), value=dict(labels=[], masks=[], descs=[]))
        print('Construction the total DDI dataset for multi-label classification')
        item_list = list()
        for row_id, row in ddi_ori_df.iterrows():
            db_id_1 = row['Drug1']
            db_id_2 = row['Drug2']
            smiles_1 = db_id2smiles.get(db_id_1, None)
            smiles_2 = db_id2smiles.get(db_id_2, None)
            if smiles_1 is not None and smiles_2 is not None:
                if (row_id + 1) % 10000 == 0:
                    print('row_id: {}, db_id_1: {}, db_id_2: {}'.format(
                        row_id + 1, db_id_1, db_id_2,
                    ))
                item = {
                    'drugbank_id_1': db_id_1,
                    'drugbank_id_2': db_id_2,
                    'smiles_1': smiles_1,
                    'smiles_2': smiles_2,
                    'label': row['Label'],
                    'mask': row['mask'],
                    'desc': row['description'],
                }

                item_list.append(item)

        print('Save the total DDI dataset.')
        columns = [
            'drugbank_id_1', 'drugbank_id_2',
            'smiles_1', 'smiles_2',
            'label', 'mask', 'desc']
        ddi_total_df = pd.DataFrame(item_list, columns=columns)

        ddi_train_df = ddi_total_df[ddi_total_df['mask'] == 'training']
        del ddi_train_df['mask']
        ddi_valid_df = ddi_total_df[ddi_total_df['mask'] == 'validation']
        del ddi_valid_df['mask']
        ddi_test_df = ddi_total_df[ddi_total_df['mask'] == 'testing']
        del ddi_test_df['mask']

        for df, flag in zip(
                [ddi_total_df, ddi_train_df, ddi_valid_df, ddi_test_df],
                ['total', 'train', 'valid', 'test']
        ):
            filename = 'ddi_{}.csv'.format(flag)
            filepath = os.path.join(KAIST_PATH, filename)
            df.to_csv(filepath)
            print('{} dataset has been generated'.format(flag))

    @staticmethod
    def to_multi_label_format(filepath, dst_filepath):

        def to_str_format(lst, delimiter='||'):
            if not isinstance(lst[0], str):
                lst = map(str, lst)
            return delimiter.join(lst)

        pair2label = dict()
        pair2desc = dict()

        df = pd.read_csv(filepath)
        for row_id, row in df.iterrows():
            db_id_1 = row['drugbank_id_1']
            db_id_2 = row['drugbank_id_2']
            if row_id % 100 == 0:
                print('row_id:{}, db_id_1:{}, db_id_2:{}'.format(
                    row_id, db_id_1, db_id_2
                ))

            if (db_id_1, db_id_2) not in pair2label.keys() \
                    and (db_id_2, db_id_1) not in pair2label.keys():
                pair = (db_id_1, db_id_2)
                pair2label[pair] = [row['label']]
                pair2desc[pair] = [row['desc']]
            else:
                pair = (db_id_1, db_id_2)
                try:
                    pair2label[pair].append(row['label'])
                    pair2desc[pair].append(row['desc'])
                except KeyError:
                    pair = (db_id_2, db_id_1)
                    pair2label[pair].append(row['label'])
                    pair2desc[pair].append(row['desc'])
                print('Multi-label pair:{}'.format(pair))

        for pair, label in pair2label.iteritems():
            if len(label) > 1:
                print('pair:{}, label:{}'.format(pair, label))

        item_list = list()
        for row_id, row in df.iterrows():
            item = dict(row)
            for key in ['label', 'desc']:
                item.pop(key)

            db_id_1 = row['drugbank_id_1']
            db_id_2 = row['drugbank_id_2']
            pair = (db_id_1, db_id_2)

            labels, descs = None, None
            try:
                labels = pair2label[pair]
                descs = pair2desc[pair]
            except KeyError:
                pair = (db_id_2, db_id_1)
                labels = pair2label[pair]
                descs = pair2desc[pair]

            if labels is not None and descs is not None:
                item.update(
                    {'labels': to_str_format(labels),
                     'descs': to_str_format(descs), }
                )

            item_list.append(item)

        columns = ['drugbank_id_1', 'drugbank_id_2', 'smiles_1', 'smiles_2',
                   'labels', 'descs']
        new_df = pd.DataFrame(item_list, columns=columns)
        new_df.to_csv(dst_filepath)


def generate_known_ddis():
    print('Generate DDI dataset in the edgelist format.')
    filename = 'drug_drug_matrix.csv'
    filepath = os.path.join(GROUND_TRUTH_PATH, filename)
    df = pd.read_csv(filepath)
    if 'Unnamed: 0' in df.columns.values:
        del df['Unnamed: 0']

    known_ddi_list = list()

    ddi_matrix = df.values.astype(np.int32)
    num_edges = int(ddi_matrix.sum() / 2.0)

    num_row, num_col = ddi_matrix.shape
    assert num_row == NUM_DRUGS and num_col == NUM_DRUGS
    cid_list = df.columns.values
    for i in range(num_row):
        cid_i = cid_list[i]
        for j in range(i+1, num_col):
            cid_j = cid_list[j]
            if ddi_matrix[i, j] == 1:
                known_ddi_list.append((cid_i, cid_j))

    assert len(known_ddi_list) == num_edges

    filename = 'ddi_zhang.edgelist'
    filepath = os.path.join(GROUND_TRUTH_PATH, filename)
    with open(filepath, 'w') as writer:
        for cid_i, cid_j in known_ddi_list:
            content = cid_i + ' ' + cid_j
            writer.write(content + '\n')
    print('DDI dataset in the edgelist format has been generated.')


def test_splitter():
    splitter = Splitter()

    print('Drug-based Random Splitter')
    name = 'ddi_dr'
    train_filename = '{}_train.csv'.format(name)
    valid_filename = '{}_valid.csv'.format(name)
    test_filename = '{}_test.csv'.format(name)
    train_filepath = os.path.join(INTERACTION_DB_PATH, train_filename)
    valid_filepath = os.path.join(INTERACTION_DB_PATH, valid_filename)
    test_filepath = os.path.join(INTERACTION_DB_PATH, test_filename)
    splitter.random_split_based_drug(train_filepath, valid_filepath, test_filepath)
    print('\n')

    print('Drug-based Scaffold Splitter')
    name = 'ddi_ds'
    train_filename = '{}_train.csv'.format(name)
    valid_filename = '{}_valid.csv'.format(name)
    test_filename = '{}_test.csv'.format(name)
    train_filepath = os.path.join(INTERACTION_DB_PATH, train_filename)
    valid_filepath = os.path.join(INTERACTION_DB_PATH, valid_filename)
    test_filepath = os.path.join(INTERACTION_DB_PATH, test_filename)
    splitter.scaffold_split_based_drug(train_filepath, valid_filepath, test_filepath)
    print('\n')

    print('Interaction-based Stratified Splitter')
    name = 'ddi_ib'
    train_filename = '{}_train.csv'.format(name)
    valid_filename = '{}_valid.csv'.format(name)
    test_filename = '{}_test.csv'.format(name)
    train_filepath = os.path.join(INTERACTION_IB_PATH, train_filename)
    valid_filepath = os.path.join(INTERACTION_IB_PATH, valid_filename)
    test_filepath = os.path.join(INTERACTION_IB_PATH, test_filename)
    splitter.random_split_based_interaction(train_filepath, valid_filepath, test_filepath)
    print('\n')

    print('Interaction-based Non-stratified Splitter')
    for scale in [5000, 10000, 15000, 20000, 25000, 30000, 35000]:
        name = 'ddi_ib_isc{}'.format(scale)
        train_filename = '{}_train.csv'.format(name)
        valid_filename = '{}_valid.csv'.format(name)
        test_filename = '{}_test.csv'.format(name)
        train_filepath = os.path.join(INTERACTION_ISC_PATH, train_filename)
        valid_filepath = os.path.join(INTERACTION_ISC_PATH, valid_filename)
        test_filepath = os.path.join(INTERACTION_ISC_PATH, test_filename)
        splitter.random_split_based_interaction_different_scales(
            scale, train_filepath, valid_filepath, test_filepath)
        print('\n')


def test_kaist_splitter():
    splitter = KaistSplitter()
    for flag in ['train', 'valid', 'test']:
        print('processing {} dataset.'.format(flag))
        filename = 'ddi_{}.csv'.format(flag)
        filepath = os.path.join(KAIST_PATH, filename)
        dst_filename = 'ddi_{}_multi.csv'.format(flag)
        dst_filepath = os.path.join(KAIST_PATH, dst_filename)
        splitter.to_multi_label_format(filepath, dst_filepath)
        print('{} dataset with the multi-label format has been generated.'.format(flag))


def parse_arguments():
    task_list = ['super-node-generation', 'super-node-embedding']

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='super-node-generation', type=str,
                        choices=task_list)

    return parser.parse_args()


if __name__ == '__main__':
    # args = parse_arguments()
    # if args.task == 'super-node-generation':
    #     add_super_nodes()
    # elif args.task == 'super-node-embedding':
    #     add_super_nodes2()
    #

    # test_kaist_splitter()

    # splitter = Splitter()
    #
    # print('Interaction-based Random Splitter with equal pos/neg ratio.')
    # name = 'ddi_equal'
    # train_filename = '{}_train.csv'.format(name)
    # valid_filename = '{}_valid.csv'.format(name)
    # test_filename = '{}_test.csv'.format(name)
    # train_filepath = os.path.join(INTERACTION_IB_PATH, train_filename)
    # valid_filepath = os.path.join(INTERACTION_IB_PATH, valid_filename)
    # test_filepath = os.path.join(INTERACTION_IB_PATH, test_filename)
    # splitter.random_split_based_interaction_equal(train_filepath, valid_filepath, test_filepath)
    # print('\n')


    generate_known_ddis()