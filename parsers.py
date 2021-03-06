#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/8/2018 9:00 PM
# @Author  : chinshin
# @FileName: parsers.py
from logging import getLogger

import numpy
import pandas
from rdkit import Chem
from tqdm import tqdm

from chainer_chemistry.dataset.parsers.base_parser import BaseFileParser
from chainer_chemistry.dataset.preprocessors.common import MolFeatureExtractionError  # NOQA
from chainer_chemistry.dataset.preprocessors.mol_preprocessor import MolPreprocessor  # NOQA
# from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset

import traceback

import os
import six

import numpy

from chainer_chemistry.dataset.indexers.numpy_tuple_dataset_feature_indexer import NumpyTupleDatasetFeatureIndexer  # NOQA


class NumpyTupleDataset(object):

    """Dataset of a tuple of datasets.

    It combines multiple datasets into one dataset. Each example is represented
    by a tuple whose ``i``-th item corresponds to the i-th dataset.
    And each ``i``-th dataset is expected to be an instance of numpy.ndarray.

    Args:
        datasets: Underlying datasets. The ``i``-th one is used for the
            ``i``-th item of each example. All datasets must have the same
            length.

    """

    def __init__(self, *datasets):
        if not datasets:
            raise ValueError('no datasets are given')
        length = len(datasets[0])
        for i, dataset in enumerate(datasets):
            if len(dataset) != length:
                raise ValueError(
                    'dataset of the index {} has a wrong length'.format(i))
        self._datasets = datasets
        self._length = length
        self._features_indexer = NumpyTupleDatasetFeatureIndexer(self)

    def __getitem__(self, index):
        batches = [dataset[index] for dataset in self._datasets]
        if isinstance(index, (slice, list, numpy.ndarray)):
            length = len(batches[0])
            return [tuple([batch[i] for batch in batches])
                    for i in six.moves.range(length)]
        else:
            return tuple(batches)

    def __len__(self):
        return self._length

    def get_datasets(self):
        return self._datasets

    @property
    def features(self):
        """Extract features according to the specified index.

        - axis 0 is used to specify dataset id (`i`-th dataset)
        - axis 1 is used to specify feature index

        .. admonition:: Example

           >>> import numpy
           >>> from chainer_chemistry.datasets import NumpyTupleDataset
           >>> x = numpy.array([0, 1, 2], dtype=numpy.float32)
           >>> t = x * x
           >>> numpy_tuple_dataset = NumpyTupleDataset(x, t)
           >>> targets = numpy_tuple_dataset.features[:, 1]
           >>> print('targets', targets)  # We can extract only target value
           targets [0, 1, 4]

        """
        return self._features_indexer

    @classmethod
    def save(cls, filepath, numpy_tuple_dataset):
        """save the dataset to filepath in npz format

        Args:
            filepath (str): filepath to save dataset. It is recommended to end
                with '.npz' extension.
            numpy_tuple_dataset (NumpyTupleDataset): dataset instance

        """
        if not isinstance(numpy_tuple_dataset, NumpyTupleDataset):
            raise TypeError('numpy_tuple_dataset is not instance of '
                            'NumpyTupleDataset, got {}'
                            .format(type(numpy_tuple_dataset)))
        numpy.savez(filepath, *numpy_tuple_dataset._datasets)

    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            return None
        load_data = numpy.load(filepath)
        result = []
        i = 0
        while True:
            key = 'arr_{}'.format(i)
            if key in load_data.keys():
                result.append(load_data[key])
                i += 1
            else:
                break
        return NumpyTupleDataset(*result)

class CSVFileParserForPair(BaseFileParser):
    """data frame parser

    This FileParser parses pandas dataframe.
    It should contain column which contain SMILES as input, and
    label column which is the target to predict.

    Args:
        preprocessor (BasePreprocessor): preprocessor instance
        labels (str or list or None): labels column
        smiles_cols (list): smiles columns
        postprocess_label (Callable): post processing function if necessary
        postprocess_fn (Callable): post processing function if necessary
        logger:
    """

    def __init__(self, preprocessor,
                 labels=None,
                 smiles_cols=('smiles_1', 'smiles_2'),
                 postprocess_label=None, postprocess_fn=None,
                 logger=None):
        super(CSVFileParserForPair, self).__init__(preprocessor)
        if isinstance(labels, str):
            labels = [labels, ]
        self.labels = labels  # type: list
        if not isinstance(smiles_cols, list):
            self.smiles_cols = list(smiles_cols)
        else:
            self.smiles_cols = smiles_cols
        self.postprocess_label = postprocess_label
        self.postprocess_fn = postprocess_fn
        self.logger = logger or getLogger(__name__)

    def parse(self, filepath, return_smiles_pair=False, return_smiles_pair_original=False, target_index=None,
              return_is_successful=False):
        """parse DataFrame using `preprocessor`

        Label is extracted from `labels` columns and input features are
        extracted from smiles information in `smiles` column.

        Args:
            filepath (str): file path to be parsed.
            return_smiles_pair (bool): If set to `True`, smiles list is returned in
                the key 'smiles', it is a list of SMILES from which input
                features are successfully made.
                If set to `False`, `None` is returned in the key 'smiles'.
            target_index (list or None): target index list to partially extract
                dataset. If None (default), all examples are parsed.
            return_is_successful (bool): If set to `True`, boolean list is
                returned in the key 'is_successful'. It represents
                preprocessing has succeeded or not for each SMILES.
                If set to False, `None` is returned in the key 'is_success'.

        Returns (dict): dictionary that contains Dataset, 1-d numpy array with
            dtype=object(string) which is a vector of smiles for each example
            or None.

        """
        df = pandas.read_csv(filepath)

        logger = self.logger
        pp = self.preprocessor
        smiles_pair_list = []
        smiles_pair_list_original = []
        is_successful_list = []

        # counter = 0
        if isinstance(pp, MolPreprocessor):
            # No influence.
            if target_index is not None:
                df = df.iloc[target_index]

            features = None
            smiles_1_index = df.columns.get_loc(self.smiles_cols[0])
            smiles_2_index = df.columns.get_loc(self.smiles_cols[1])
            if self.labels is None:
                labels_index = []  # dummy list
            else:
                labels_index = [df.columns.get_loc(c) for c in self.labels]

            total_count = df.shape[0]
            fail_count = 0
            success_count = 0
            # iteration on every row within the csv file
            for row in tqdm(df.itertuples(index=False), total=df.shape[0]):
                smiles_1 = row[smiles_1_index]
                smiles_2 = row[smiles_2_index]
                # currently it assumes list
                labels = [int(row[i]) for i in labels_index]
                try:
                    # if '.' in smiles_1.strip().split()[0] or '.' in smiles_2.strip().split()[0]:
                    #     fail_count += 1
                    #     if return_is_successful:
                    #         is_successful_list.append(False)
                    #     continue

                    mol_1 = Chem.MolFromSmiles(smiles_1)
                    mol_2 = Chem.MolFromSmiles(smiles_2)

                    if mol_1 is None or mol_2 is None:
                        fail_count += 1
                        if return_is_successful:
                            is_successful_list.append(False)
                        continue
                    # Note that smiles expression is not unique.
                    # we obtain canonical smiles
                    # canonical_smiles_1, mol_1 = pp.prepare_smiles_and_mol(mol_1)
                    # input_features_1 = pp.get_input_features(mol_1)
                    # canonical_smiles_2, mol_2 = pp.prepare_smiles_and_mol(mol_2)
                    # input_features_2 = pp.get_input_features(mol_2)

                    input_features_1 = pp.get_input_features(mol_1)
                    input_features_2 = pp.get_input_features(mol_2)

                    # Extract label
                    if self.postprocess_label is not None:
                        labels = self.postprocess_label(labels)

                    # if return_smiles_pair:
                    #     smiles_pair_list.append([canonical_smiles_1, canonical_smiles_2])
                    if return_smiles_pair:
                        smiles_pair_list.append([smiles_1, smiles_2])
                    if return_smiles_pair_original:
                        smiles_pair_list_original.append([smiles_1, smiles_2])

                except MolFeatureExtractionError as e:
                    # This is expected error that extracting feature failed,
                    # skip this molecule.
                    fail_count += 1
                    if return_is_successful:
                        is_successful_list.append(False)
                    continue
                except Exception as e:
                    logger.warning('parse(), type: {}, {}'
                                   .format(type(e).__name__, e.args))
                    logger.info(traceback.format_exc())
                    fail_count += 1
                    if return_is_successful:
                        is_successful_list.append(False)
                    continue
                # Initialize features: list of list
                if features is None:
                    if isinstance(input_features_1, tuple):
                        num_features_1 = len(input_features_1)
                    else:
                        num_features_1 = 1
                    if isinstance(input_features_2, tuple):
                        num_features_2 = len(input_features_2)
                    else:
                        num_features_2 = 1
                    num_features = num_features_1 + num_features_2
                    if self.labels is not None:
                        num_features += 1
                    # list of list, a sublist corresponding to a certain feature
                    features = [[] for _ in range(num_features)]
                # for every row in csv file
                if isinstance(input_features_1, tuple):
                    for i in range(len(input_features_1)):
                        # features[i] a list containing the i-th feature
                        features[i].append(input_features_1[i])
                else:
                    features[0].append(input_features_1)
                offset = len(input_features_1)
                if isinstance(input_features_2, tuple):
                    for i in range(len(input_features_2)):
                        features[offset + i].append(input_features_2[i])
                else:
                    features[offset].append(input_features_2)

                # last column corresponding to targeted label
                if self.labels is not None:
                    features[len(features) - 1].append(labels)

                success_count += 1
                if return_is_successful:
                    is_successful_list.append(True)

            ret = []
            for feature in features:
                try:
                    feat_array = numpy.asarray(feature)
                except ValueError:
                    # Temporal work around.
                    # See,
                    # https://stackoverflow.com/questions/26885508/why-do-i-get-error-trying-to-cast-np-arraysome-list-valueerror-could-not-broa
                    feat_array = numpy.empty(len(feature), dtype=numpy.ndarray)
                    feat_array[:] = feature[:]
                ret.append(feat_array)
            result = tuple(ret)
            logger.info('Preprocess finished. FAIL {}, SUCCESS {}, TOTAL {}'
                        .format(fail_count, success_count, total_count))
        else:
            raise NotImplementedError

        smiles_pairs = numpy.array(smiles_pair_list) if return_smiles_pair else None
        smiles_pairs_original = numpy.array(smiles_pair_list_original) if return_smiles_pair_original else None
        if return_is_successful:
            is_successful = numpy.array(is_successful_list)
        else:
            is_successful = None

        if isinstance(result, tuple):
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(*result)
            dataset = NumpyTupleDataset(*result)
        else:
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(result)
            dataset = NumpyTupleDataset(result)
        return {"dataset": dataset,
                "smiles_pair": smiles_pairs,
                "smiles_pair_original": smiles_pairs_original,
                "is_successful": is_successful}

    def extract_total_num(self, df):
        """Extracts total number of data which can be parsed

        We can use this method to determine the value fed to `target_index`
        option of `parse` method. For example, if we want to extract input
        feature from 10% of whole dataset, we need to know how many samples
        are in a file. The returned value of this method may not to be same as
        the final dataset size.

        Args:
            df (pandas.DataFrame): dataframe to be parsed.

        Returns (int): total number of dataset can be parsed.

        """
        return len(df)