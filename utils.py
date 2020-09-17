#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/8/2018 6:58 PM
# @Author  : chinshin
# @FileName: utils.py
import os
import csv
import sys
import random
import logging
import numpy as np
from os.path import abspath, dirname
from sklearn.model_selection import train_test_split

logging.basicConfig(format='%(asctime)s: %(filename)s: %(funcName)s: %(lineno)d: %(message)s', level=logging.INFO)
ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_PATH)
global_seed = 2018
random.seed(global_seed)

def load_csv(filename, type):
    matrix_data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)
        for row_vector in csvreader:
            if type == 'int':
                matrix_data.append(list(map(int, row_vector[1:])))
            else:
                matrix_data.append(list(map(float, row_vector[1:])))
    return np.matrix(matrix_data)


def index2id():
    drug_list_filename = 'add_zinc_id_smiles.txt'
    drug_list_filepath = os.path.join(ROOT_PATH, 'examples', 'ddi', drug_list_filename)

    cid_list = list()
    drugbank_id_list = list()
    zinc_id_list = list()
    smiles_list = list()
    with open(drug_list_filepath, 'r') as txt_reader:
        for line_no, line in enumerate(txt_reader.readlines()):
            cid, drugbank_id, zinc_id, smiles = line.strip('\n').split(' ')
            cid = 'CID' + '0' * (9 - len(cid)) + cid
            cid_list.append(cid)
            drugbank_id_list.append(drugbank_id)
            zinc_id_list.append(zinc_id)
            smiles_list.append(smiles)

    return cid_list, drugbank_id_list, zinc_id_list, smiles_list


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def str_2_float_list(string):
    return [float(e) for e in string.split(' ')]


def float_list_2_str(lst):
    return ' '.join([str(e) for e in lst])


def check_list(lst):
    if not isinstance(lst, list):
        lst = list(lst)
    return lst




def split_dataset(drug_drug_matrix, format='txt'):
    drug_pairs = list()
    labels = list()

    for row in range(0, len(drug_drug_matrix)):
        for col in range(row + 1, len(drug_drug_matrix)):
            if drug_drug_matrix[row, col] == 1:
                drug_pairs.append([row, col])
                labels.append(1)
            else:
                drug_pairs.append([row, col])
                labels.append(0)

    drug_pairs = np.array(drug_pairs)
    labels = np.array(labels)
    pos_neg_ratio_total = float(np.sum(labels == 1)) / float(np.sum(labels == 0))

    pairs_train, pairs_test, labels_train, labels_test = train_test_split(
        drug_pairs, labels, test_size=0.2, stratify=labels, random_state=global_seed)
    pos_neg_ratio_train = float(np.sum(labels_train == 1)) / float(np.sum(labels_train == 0))
    pos_neg_ratio_test = float(np.sum(labels_test == 1)) / float(np.sum(labels_test == 0))

    assert int(pos_neg_ratio_total * 100) == int(pos_neg_ratio_train * 100)
    assert int(pos_neg_ratio_total * 100) == int(pos_neg_ratio_test * 100)

    cid_list, drugbank_id_list, zinc_id_list, smiles_list = index2id()
    if format == 'txt':
        ddi_train_filename = 'ddi_train.txt'
        ddi_test_filename = 'ddi_test.txt'
        ddi_train_filepath = os.path.join(ROOT_PATH, 'examples', 'ddi', ddi_train_filename)
        ddi_test_filepath = os.path.join(ROOT_PATH, 'examples', 'ddi', ddi_test_filename)

        with open(ddi_train_filepath, 'w') as txt_writer:
            for pair, label in zip(pairs_train, labels_train):
                drug1_id, drug2_id = pair
                drug1_drugbank_id = drugbank_id_list[drug1_id]
                drug1_smiles = smiles_list[drug1_id]
                drug2_drugbank_id = drugbank_id_list[drug2_id]
                drug2_smiles = smiles_list[drug2_id]
                line = ' '.join([drug1_drugbank_id, drug2_drugbank_id, drug1_smiles, drug2_smiles, str(label)]) + '\n'
                txt_writer.write(line)
            logging.info('Train dataset in txt format generated.')

        with open(ddi_test_filepath, 'w') as txt_writer:
            for pair, label in zip(pairs_test, labels_test):
                drug1_id, drug2_id = pair
                drug1_drugbank_id = drugbank_id_list[drug1_id]
                drug1_smiles = smiles_list[drug1_id]
                drug2_drugbank_id = drugbank_id_list[drug2_id]
                drug2_smiles = smiles_list[drug2_id]
                line = ' '.join([drug1_drugbank_id, drug2_drugbank_id, drug1_smiles, drug2_smiles, str(label)]) + '\n'
                txt_writer.write(line)
            logging.info('Test datset in txt format generated.')

        logging.info('Dataset in text format split done.')
    elif format == 'csv':
        ddi_train_filename = 'ddi_train.csv'
        ddi_test_filename = 'ddi_test.csv'
        ddi_train_filepath = os.path.join(ROOT_PATH, 'examples', 'ddi', ddi_train_filename)
        ddi_test_filepath = os.path.join(ROOT_PATH, 'examples', 'ddi', ddi_test_filename)

        with open(ddi_train_filepath, 'w') as csvfile:
            writer = csv.writer(csvfile)

            column_names = ['drugbank_id_1', 'drugbank_id_2', 'smiles_1', 'smiles_2', 'label']
            writer.writerow(column_names)

            for pair, label in zip(pairs_train, labels_train):
                drug1_id, drug2_id = pair
                drug1_drugbank_id = drugbank_id_list[drug1_id]
                drug1_smiles = smiles_list[drug1_id]
                drug2_drugbank_id = drugbank_id_list[drug2_id]
                drug2_smiles = smiles_list[drug2_id]
                line = [drug1_drugbank_id, drug2_drugbank_id, drug1_smiles, drug2_smiles, str(label)]
                writer.writerow(line)
            logging.info('Train dataset in csv format generated.')

        with open(ddi_test_filepath, 'w') as csvfile:
            writer = csv.writer(csvfile)

            column_names = ['drugbank_id_1', 'drugbank_id_2', 'smiles_1', 'smiles_2', 'label']
            writer.writerow(column_names)

            for pair, label in zip(pairs_test, labels_test):
                drug1_id, drug2_id = pair
                drug1_drugbank_id = drugbank_id_list[drug1_id]
                drug1_smiles = smiles_list[drug1_id]
                drug2_drugbank_id = drugbank_id_list[drug2_id]
                drug2_smiles = smiles_list[drug2_id]
                line = [drug1_drugbank_id, drug2_drugbank_id, drug1_smiles, drug2_smiles, str(label)]
                writer.writerow(line)
            logging.info('Train dataset in csv format generated.')
        logging.info('Dataset in csv format split done.')


def split_dataset_imbalance(drug_drug_matrix):
    drug_pairs = list()
    labels = list()

    for row in range(0, len(drug_drug_matrix)):
        for col in range(row + 1, len(drug_drug_matrix)):
            if drug_drug_matrix[row, col] == 1:
                drug_pairs.append([row, col])
                labels.append(1)
            else:
                drug_pairs.append([row, col])
                labels.append(0)

    drug_pairs = np.array(drug_pairs)
    labels = np.array(labels)

    pos_drug_pairs = drug_pairs[labels == 1]
    neg_drug_pairs = drug_pairs[labels == 0]
    num_pos = pos_drug_pairs.shape[0]
    train_pos_ratio = 0.8
    num_pos_train = int(train_pos_ratio * num_pos)
    pos_drug_pair_indices = np.arange(0, num_pos)
    np.random.seed(777)
    np.random.shuffle(pos_drug_pair_indices)
    pos_train_drug_pair_indices = pos_drug_pair_indices[:num_pos_train]
    pos_test_drug_pair_indices = pos_drug_pair_indices[num_pos_train:]
    pos_train_drug_pairs = pos_drug_pairs[pos_train_drug_pair_indices]
    pos_train_labels = np.ones((pos_train_drug_pairs.shape[0], ), dtype=np.int32)
    pos_test_drug_pairs = pos_drug_pairs[pos_test_drug_pair_indices]
    pos_test_labels = np.ones((pos_test_drug_pairs.shape[0], ), dtype=np.int32)
    num_neg = neg_drug_pairs.shape[0]
    num_neg_train = num_pos_train
    neg_drug_pairs_indices = np.arange(0, num_neg)
    neg_train_drug_pair_indices = neg_drug_pairs_indices[:num_neg_train]
    neg_test_drug_pair_indices = neg_drug_pairs_indices[num_neg_train:]
    neg_train_drug_pairs = neg_drug_pairs[neg_train_drug_pair_indices]
    neg_train_labels = np.zeros((neg_train_drug_pairs.shape[0], ), dtype=np.int32)
    neg_test_drug_pairs = neg_drug_pairs[neg_test_drug_pair_indices]
    neg_test_labels = np.zeros((neg_test_drug_pairs.shape[0], ), dtype=np.int32)

    pairs_train = np.concatenate((pos_train_drug_pairs, neg_train_drug_pairs), axis=0)
    pairs_test = np.concatenate((pos_test_drug_pairs, neg_test_drug_pairs), axis=0)
    labels_train = np.concatenate((pos_train_labels, neg_train_labels), axis=0)
    labels_test = np.concatenate((pos_test_labels, neg_test_labels), axis=0)
    train_indices = np.arange(0, pairs_train.shape[0])
    np.random.shuffle(train_indices)
    pairs_train = pairs_train[train_indices]
    labels_train = labels_train[train_indices]
    test_indices = np.arange(0, pairs_test.shape[0])
    random.shuffle(test_indices)
    pairs_test = pairs_test[test_indices]
    labels_test = labels_test[test_indices]

    cid_list, drugbank_id_list, zinc_id_list, smiles_list = index2id()

    ddi_train_filename = 'ddi_train_imbalance.csv'
    ddi_test_filename = 'ddi_test_imbalance.csv'
    ddi_train_filepath = os.path.join(ROOT_PATH, 'examples', 'ddi', ddi_train_filename)
    ddi_test_filepath = os.path.join(ROOT_PATH, 'examples', 'ddi', ddi_test_filename)

    with open(ddi_train_filepath, 'w') as csvfile:
        writer = csv.writer(csvfile)

        column_names = ['drugbank_id_1', 'drugbank_id_2', 'smiles_1', 'smiles_2', 'label']
        writer.writerow(column_names)

        for pair, label in zip(pairs_train, labels_train):
            drug1_id, drug2_id = pair
            drug1_drugbank_id = drugbank_id_list[drug1_id]
            drug1_smiles = smiles_list[drug1_id]
            drug2_drugbank_id = drugbank_id_list[drug2_id]
            drug2_smiles = smiles_list[drug2_id]
            line = [drug1_drugbank_id, drug2_drugbank_id, drug1_smiles, drug2_smiles, str(label)]
            writer.writerow(line)
        logging.info('Train dataset in csv format generated.')

    with open(ddi_test_filepath, 'w') as csvfile:
        writer = csv.writer(csvfile)

        column_names = ['drugbank_id_1', 'drugbank_id_2', 'smiles_1', 'smiles_2', 'label']
        writer.writerow(column_names)

        for pair, label in zip(pairs_test, labels_test):
            drug1_id, drug2_id = pair
            drug1_drugbank_id = drugbank_id_list[drug1_id]
            drug1_smiles = smiles_list[drug1_id]
            drug2_drugbank_id = drugbank_id_list[drug2_id]
            drug2_smiles = smiles_list[drug2_id]
            line = [drug1_drugbank_id, drug2_drugbank_id, drug1_smiles, drug2_smiles, str(label)]
            writer.writerow(line)
        logging.info('Train dataset in csv format generated.')
    logging.info('Dataset in csv format split done.')


def generate_train_test_dataset():
    drug_drug_matrix_filename = 'drug_drug_matrix.csv'
    drug_drug_matrix_filepath = os.path.join(ROOT_PATH, 'examples', 'ddi', drug_drug_matrix_filename)
    drug_drug_matrix = load_csv(drug_drug_matrix_filepath, 'int')
    split_dataset(drug_drug_matrix)


def parse_csv(filepath):
    cid_list = list()
    drugbank_id_list = list()
    drug_list_filename = 'drug_list.txt'
    drug_pair_filepath = os.path.join(ROOT_PATH, 'examples', 'ddi', drug_list_filename)
    with open(drug_pair_filepath, 'r') as txt_reader:
        for line in txt_reader.readlines():
            cid, drugbank_id = line.strip('\n').split(' ')
            cid = 'CID' + '0' * (9-len(cid)) + cid
            cid_list.append(cid)
            drugbank_id_list.append(drugbank_id)

    with open(filepath, 'r') as csv_file:
        reader = csv.reader(csv_file)
        drug_pair_list = []
        label_list = []
        for item in reader:
            if reader.line_num == 1:
                continue
            drugbank_id_1 = item[0]
            drugbank_id_2 = item[1]
            index_1 = drugbank_id_list.index(drugbank_id_1)
            index_2 = drugbank_id_list.index(drugbank_id_2)
            label = int(item[4])
            drug_pair_list.append([index_1, index_2])
            label_list.append(label)

    return drug_pair_list, label_list


def split_ddi_train():
    ddi_filename = 'ddi_train.csv'
    ddi_filepath = os.path.join(ROOT_PATH, 'examples', 'ddi', ddi_filename)
    drug_pair_list, label_list = parse_csv(ddi_filepath)
    train_ratio = 0.8
    pairs_train, pairs_test, labels_train, labels_test = train_test_split(
        drug_pair_list, label_list, train_size=train_ratio, random_state=777
    )
    ddi_train_filename = 'ddi_train{}_train.csv'.format(train_ratio)
    ddi_test_filename = 'ddi_train{}_test.csv'.format(train_ratio)
    ddi_train_filepath = os.path.join(ROOT_PATH, 'examples', 'ddi', ddi_train_filename)
    ddi_test_filepath = os.path.join(ROOT_PATH, 'examples', 'ddi', ddi_test_filename)

    cid_list, drugbank_id_list, zinc_id_list, smiles_list = index2id()

    with open(ddi_train_filepath, 'w') as csvfile:
        writer = csv.writer(csvfile)

        column_names = ['drugbank_id_1', 'drugbank_id_2', 'smiles_1', 'smiles_2', 'label']
        writer.writerow(column_names)

        for pair, label in zip(pairs_train, labels_train):
            drug1_id, drug2_id = pair
            drug1_drugbank_id = drugbank_id_list[drug1_id]
            drug1_smiles = smiles_list[drug1_id]
            drug2_drugbank_id = drugbank_id_list[drug2_id]
            drug2_smiles = smiles_list[drug2_id]
            line = [drug1_drugbank_id, drug2_drugbank_id, drug1_smiles, drug2_smiles, str(label)]
            writer.writerow(line)
        logging.info('Train dataset in csv format generated.')

    with open(ddi_test_filepath, 'w') as csvfile:
        writer = csv.writer(csvfile)

        column_names = ['drugbank_id_1', 'drugbank_id_2', 'smiles_1', 'smiles_2', 'label']
        writer.writerow(column_names)

        for pair, label in zip(pairs_test, labels_test):
            drug1_id, drug2_id = pair
            drug1_drugbank_id = drugbank_id_list[drug1_id]
            drug1_smiles = smiles_list[drug1_id]
            drug2_drugbank_id = drugbank_id_list[drug2_id]
            drug2_smiles = smiles_list[drug2_id]
            line = [drug1_drugbank_id, drug2_drugbank_id, drug1_smiles, drug2_smiles, str(label)]
            writer.writerow(line)
        logging.info('Train dataset in csv format generated.')
    logging.info('Dataset in csv format split done.')

if __name__ == '__main__':
    split_ddi_train()

