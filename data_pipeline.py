# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Preprocessing Pipeline
"""
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# import sys
# from os.path import abspath, dirname
# ROOT_PATH = dirname(abspath(__file__))
# sys.path.insert(0, ROOT_PATH)
from setting import *
from utils.parsers import CSVFileParserForPair
from utils.preprocessors import GGNNPreprocessor
from utils import NumpyTupleDataset


class BinaryDDI(object):
    preprocessor_dict = {
        'ggnn': GGNNPreprocessor
    }

    @staticmethod
    def postprocess_label(label_list):
        label_arr = np.asarray(label_list, dtype=np.int32)
        return label_arr

    @staticmethod
    def augment_dataset(dataset):
        dataset_tuple = dataset.get_datasets()
        atoms1, adjs1, atoms2, adjs2, labels = dataset_tuple
        new_atoms1 = np.concatenate((atoms1, atoms2), axis=0)
        new_atoms2 = np.concatenate((atoms2, atoms1), axis=0)
        new_adjs1 = np.concatenate((adjs1, adjs2), axis=0)
        new_adjs2 = np.concatenate((adjs2, adjs1), axis=0)
        new_labels = np.concatenate((labels, labels), axis=0)
        new_dataset = NumpyTupleDataset(new_atoms1, new_adjs1, new_atoms2, new_adjs2, new_labels)
        return new_dataset

    def __init__(self, filepath, augmentation=True, preprocessor='ggnn'):
        smiles_cols = ['smiles_1', 'smiles_2']
        label_cols = ['label', ]

        filename = os.path.basename(filepath)
        path = os.path.dirname(filepath)
        processed_filename = ''.join([filename.split('.')[0],  '_processed', '.pkl'])
        self.filepath = filepath
        self.processed_filepath = os.path.join(path, processed_filename)

        if not os.path.exists(self.processed_filepath):
            print('Preprocess...')
            preprocessor = BinaryDDI.preprocessor_dict[preprocessor]()
            parser = CSVFileParserForPair(
                preprocessor, postprocess_label=self.postprocess_label,
                labels=label_cols, smiles_cols=smiles_cols)
            self.dataset = parser.parse(filepath)['dataset']

            if augmentation:
                self.dataset = self.augment_dataset(self.dataset)

            # if isinstance(self.dataset, NumpyTupleDataset):
            self.atoms_1, self.adjs_1, self.atoms_2, self.adjs_2, self.labels = self.dataset.get_datasets()

            with open(self.processed_filepath, 'wb') as writer:
                print('Save in disk file {}.'.format(self.processed_filepath))
                pickle.dump(self.dataset, writer, protocol=2)

        else:
            with open(self.processed_filepath, 'rb') as reader:
                print('Load from disk file {}.'.format(self.processed_filepath))
                self.dataset = pickle.load(reader)
            self.atoms_1, self.adjs_1, self.atoms_2, self.adjs_2, self.labels = self.dataset.get_datasets()

        # onehot_enc = OneHotEncoder()
        # self.labels = onehot_enc.fit_transform(self.labels).toarray()
        self.data_list = list(zip(self.atoms_1, self.adjs_1, self.atoms_2, self.adjs_2, self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data_list[idx]

    @property
    def raw_file_path(self):
        return self.filepath

    @property
    def processed_file_path(self):
        return self.processed_filepath

    @property
    def class_num(self):
        labels = self.labels if isinstance(self.labels, np.ndarray) else np.array(self.labels, dtype=np.int)
        return len(np.unique(np.array(labels)))


if __name__ == '__main__':
    ddi_filename = 'sample200.csv'
    ddi_filepath = os.path.join(INTERACTION_SAMPLE_PATH, ddi_filename)

    ddi_dataset = BinaryDDI(filepath=ddi_filepath)
    for ind, ddi in enumerate(ddi_dataset[:10]):
        atom_array_1, adj_1, atom_array_2, adj_2, label = ddi
        print(ind, atom_array_1.shape, adj_1.shape, atom_array_2.shape, adj_2.shape, label.shape)

