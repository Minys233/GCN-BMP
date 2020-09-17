#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/26/2019 7:23 PM
# @Author  : chinshin
# @FileName: mol2vec_predict.py

import os
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from setting import *

test_filename = "ddi_ib_isc35000_test.csv"
test_filepath = os.path.join(INTERACTION_ISC_PATH, test_filename)

smiles2vec_filename = "smiles2vec.pkl"
smiles2vec_filepath = os.path.join(DRUG_LIST_PATH, smiles2vec_filename)
with open(smiles2vec_filepath, "rb") as reader:
    smiles2vec = pickle.load(reader)

def cosine_sim(x, y):
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    cosine_sim = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return cosine_sim / (norm_x * norm_y)


test_df = pd.read_csv(test_filepath, index_col=0)
smiles1_list = test_df['smiles_1']
vec1_arr = np.array([smiles2vec[smiles] for smiles in smiles1_list], dtype=np.float32)
smiles2_list = test_df['smiles_2']
vec2_arr = np.array([smiles2vec[smiles] for smiles in smiles2_list], dtype=np.float32)
label_arr = np.array(test_df['label'], dtype=np.float32)

predict_list = list()
score_list = list()
for vec1, vec2, label in zip(vec1_arr, vec2_arr, label_arr):
    sim_value = cosine_sim(vec1, vec2)
    pred = 1 if sim_value > 0.5 else 1
    score_list.append(sim_value)
    predict_list.append(pred)
score_arr = np.array(score_list, dtype=np.int32)
predict_arr = np.array(predict_list, dtype=np.int32)

auroc = metrics.roc_auc_score(label_arr, score_arr)
auprc = metrics.average_precision_score(label_arr, score_arr)
f1_score = metrics.f1_score(label_arr, predict_arr)

print("auroc:{:.4f}, auprc:{:.4f}, f1: {:.4f}".format(
    auroc, auprc, f1_score
))
