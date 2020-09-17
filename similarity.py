#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/23/2019 11:28 PM
# @Author  : chinshin
# @FileName: similarity.py

import os
import csv
import sys
import copy
import argparse
import numpy as np
import pandas as pd
from numpy.fft import fft, ifft
from pandas import DataFrame
from os.path import dirname, abspath

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_PATH)
from setting import *
from utils import *


class SimilarityCalculator(object):

    def __init__(self, out_path):
        drug_list_filename = 'drug_list_with_representation.csv'
        self.drug_list_filepath = os.path.join(out_path, drug_list_filename)
        self.metric_dict = {
            'continous_Jaccard': self.continous_Jaccard_sim,
            'extended_Jaccard': self.extended_Jaccard_sim,
            'cosine': self.cosine_sim,
            # 'circ_conv': self.circ_conv,
            # 'circ_corr': self.circ_corr,
        }

    def calculate(self, metric, out_filepath):

        metric_fun = self.metric_dict[metric]

        cid_and_embedding = pd.read_csv(self.drug_list_filepath)
        cid_list = cid_and_embedding['cid'].values
        embedding_list = cid_and_embedding['entity'].values
        embedding_list = [str_2_float_list(embedding) for embedding in embedding_list]

        cid2embedding = dict(zip(cid_list, embedding_list))

        assert len(cid_list) == NUM_DRUGS
        data = np.zeros(shape=(NUM_DRUGS, NUM_DRUGS), dtype=np.float32)

        frame = DataFrame(data, columns=cid_list, index=cid_list)
        columns = frame.columns
        for row_cid, row in frame.iterrows():
            row_vector = cid2embedding[row_cid]
            for col_cid in columns:
                if row_cid == col_cid:
                    continue
                col_vector = cid2embedding[col_cid]
                try:
                    sim = metric_fun(row_vector, col_vector)
                except ValueError:
                    print(row_cid, col_cid)
                    return
                try:
                    frame.set_value(row_cid, col_cid, sim)
                except KeyError:
                    print(row_cid, col_cid)
                    break
        frame.to_csv(out_filepath)

    @staticmethod
    def continous_Jaccard_sim(x, y):
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        data = np.array([x, y], dtype=np.float32)
        min_value = np.min(data, axis=0)
        sum_min_value = np.sum(min_value)
        max_value = np.max(data, axis=0)
        sum_max_value = np.sum(max_value)
        jaccard_sim = 1. - sum_min_value / sum_max_value
        return jaccard_sim

    @staticmethod
    def extended_Jaccard_sim(x, y):
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        cosine_sim = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        sim = cosine_sim / (norm_x + norm_y - cosine_sim)
        return sim

    @staticmethod
    def cosine_sim(x, y):
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        cosine_sim = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        return cosine_sim / (norm_x * norm_y)

    @staticmethod
    def circ_conv(x, y):
        return ifft(fft(x) * fft(y)).real

    @staticmethod
    def circ_corr(x, y):
        sim1 = ifft(np.conj(fft(x)) * fft(y)).real
        sim2 = ifft(np.conj(fft(y)) * fft(x)).real
        return (sim1 + sim2) / 2


def parse_arguments():
    metric_list = ['continous_Jaccard', 'extended_Jaccard', 'cosine']

    parser = argparse.ArgumentParser(description='Similarity Calculator for binary DDI prediction.')
    parser.add_argument('--out', type=str, default='result',
                        help='path to save the computed models to')
    parser.add_argument('--metric', type=str, default='cosine', choices=metric_list,
                        help='metric used to calculate the similarity between distinct drugs.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    out = './output/{}'.format(args.out)
    cal = SimilarityCalculator(out_path=out)
    similarity_filename = 'similarity_{metric}.csv'.format(metric=args.metric)
    similarity_filepath = os.path.join(out, similarity_filename)
    cal.calculate(metric=args.metric, out_filepath=similarity_filepath)
    print('Similarity measured by {metric} has been saved in the file {filepath}'.format(
        metric=args.metric, filepath=similarity_filepath
    ))





