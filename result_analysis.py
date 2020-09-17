#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/14/2019 9:21 PM
# @Author  : chinshin
# @FileName: result_analysis.py
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import logging
from scipy import stats
import pandas as pd
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from setting import *

logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.INFO)
"""
result analysis for our proposed model: ggnn-hole
"""

reference_filepath = os.path.join(INTERACTION_ISC_PATH, "ddi_ib_isc35000_train.csv")

def pair_similarity(drugbank_id_1, drugbank_id_2):
    inter_drugs_1 = set()
    inter_drugs_2 = set()
    result_df = pd.read_csv(reference_filepath)
    for row_id, row in result_df.iterrows():
        label = row['label']
        if label == 1:
            if drugbank_id_1 == row['drugbank_id_1']:
                inter_drugs_1.add(row['drugbank_id_2'])
            if drugbank_id_1 == row['drugbank_id_2']:
                inter_drugs_1.add(row['drugbank_id_1'])
            if drugbank_id_2 == row['drugbank_id_1']:
                inter_drugs_2.add(row['drugbank_id_2'])
            if drugbank_id_2 == row['drugbank_id_2']:
                inter_drugs_2.add(row['drugbank_id_1'])

    sim = len(inter_drugs_1 & inter_drugs_2) / len(inter_drugs_1 | inter_drugs_2)
    return sim


def size_of_intersection(drugbank_id_1, drugbank_id_2):
    inter_drugs_1 = set()
    inter_drugs_2 = set()
    result_df = pd.read_csv(reference_filepath)
    for row_id, row in result_df.iterrows():
        label = row['label']
        if label == 1:
            if drugbank_id_1 == row['drugbank_id_1']:
                inter_drugs_1.add(row['drugbank_id_2'])
            if drugbank_id_1 == row['drugbank_id_2']:
                inter_drugs_1.add(row['drugbank_id_1'])
            if drugbank_id_2 == row['drugbank_id_1']:
                inter_drugs_2.add(row['drugbank_id_2'])
            if drugbank_id_2 == row['drugbank_id_2']:
                inter_drugs_2.add(row['drugbank_id_1'])

    # logging.info(inter_drugs_1 & inter_drugs_2)
    size = len(inter_drugs_1 & inter_drugs_2)
    return size


def test_inspiration():
    cases = [
        # ("DB00526", "DB01413"),
        # ("DB01413", "DB00582"),
        # ("DB00181", "DB00268"),
        # ("DB01212", "DB01004"),
        # ("DB00454", "DB01413"),
        # ("DB00709", "DB00268"),
        # ("DB01204", "DB01248"),
        ("DB01413", "DB00695"), # 0.999958 1
        ("DB01413", "DB00448"), # 0.999905 1
        ("DB01413", "DB00441"), # 0.776317 1
        ("DB01413", "DB00328"), # 0.743091 0
        ("DB01413", "DB00631"), # 0.138230
    ]

    # # Cefepime
    # drugbank_id_1 = "DB01413"
    # # Voriconazole
    # drugbank_id_2 = "DB00582"

    for drugbank_id_1, drugbank_id_2 in cases:
        sim = pair_similarity(drugbank_id_1, drugbank_id_2)
        logging.info("Jaccard coefficient for {} and {}: {:.3f}".format(
            drugbank_id_1, drugbank_id_2, sim))


def test_correlation(filename, fig_path=None, drug_bank_id=None):
    result_path = "/home/chenx/drug_mining/representation_learning/chainer-chemistry/examples/ddi/output/result_isc35000_ggnn_epoch200_lr0.001_exp0.5_wt0_fph32_conv8_nh_hole_x41"
    # result_filepath = os.path.join(result_path, "ddi_ib_isc35000_test_e_y_DB01413.csv")
    result_filepath = os.path.join(result_path, filename)

    result_df = pd.read_csv(result_filepath, index_col=0)

    predicted_scores_true = list()
    inter_sizes_true = list()
    predicted_scores_false = list()
    inter_sizes_false = list()
    if drug_bank_id is not None:
        for row_id, row in result_df.iterrows():
            if row['drugbank_id_1'] == drug_bank_id or row['drugbank_id_2'] == drug_bank_id:
                if int(row['y'] > 0.5) == row['label']:
                    predicted_scores_true.append(row['y'])
                    inter_sizes_true.append(row['inter_size'])
                else:
                    predicted_scores_false.append(row['y'])
                    inter_sizes_false.append(row['inter_size'])
    else:
        for row_id, row in result_df.iterrows():
            if int(row['y'] > 0.5) == row['label']:
                predicted_scores_true.append(row['y'])
                inter_sizes_true.append(row['inter_size'])
            else:
                predicted_scores_false.append(row['y'])
                inter_sizes_false.append(row['inter_size'])

    predicted_scores_true = np.array(predicted_scores_true, dtype=np.float32)
    inter_sizes_true = np.array(inter_sizes_true, dtype=np.int32)
    predicted_scores_false = np.array(predicted_scores_false, dtype=np.float32)
    inter_sizes_false = np.array(inter_sizes_false, dtype=np.int32)
    predicted_scores = np.concatenate((predicted_scores_true, predicted_scores_false))
    inter_sizes = np.concatenate((inter_sizes_true, inter_sizes_false))
    coef = stats.pearsonr(predicted_scores, inter_sizes)

    # inter_sizes = (inter_sizes - np.min(inter_sizes)) / (np.max(inter_sizes) - np.min(inter_sizes))
    logging.info('For {}, pearson coefficient is: {:.4f}, p-value is: {:E}'.format(drug_bank_id, *coef))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(predicted_scores_true, inter_sizes_true, c='g', marker='o', label='Correctly-predicted')
    ax.scatter(predicted_scores_false, inter_sizes_false, c='r', marker='s', label='Wrongly-predicted')
    plt.xlim([0-0.05, 1+0.05])
    plt.ylim([-10, 260])
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks([0, 50, 100, 150, 200, 250])
    plt.legend(loc='upper left')
    plt.xlabel('Predicted Score for Interaction')
    plt.ylabel('Size of Intersection')
    plt.title("Relationship between Predicted Score and Size of Interaction for {}".format(drug_bank_id), fontsize=10)
    plt.text(0.05, 150, "Coef      :  {:.4f} \nP-Value :  {:.4f}".format(coef[0], coef[1]), fontsize=10,
             size=10,
             bbox=dict(boxstyle='round,pad=1', ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
    # plt.text(0.05, 160, "P-value: {:.4f}".format(coef[1]), fontsize=10)
    fig_path = result_path if fig_path is None else fig_path
    if drug_bank_id is not None:
        fig_filepath = os.path.join(fig_path, 'correlation_{}_{:.4f}.png'.format(drug_bank_id, coef[0]))
    else:
        fig_filepath = os.path.join(fig_path, 'correlation_all_{:.4f}.png'.format(coef[0]))
    plt.savefig(fig_filepath)

    coef, p_val = coef
    return coef, p_val


def calculate_inter_size(filepath=None, dst_filepath=None):
    # probe into the relationship between predicted score and number of intersection between two interaction drug sets.
    if not os.path.exists(filepath):
        logging.info("{} doesn't not exist!".format(filepath))
        return

    result_df = pd.read_csv(filepath)

    inter_sizes = list()
    for row_id, row in result_df.iterrows():
        label = row['label']
        predicted_score = row['y']
        num_intersection = size_of_intersection(row['drugbank_id_1'], row['drugbank_id_2'])
        inter_sizes.append(num_intersection)
        logging.info("{}: Number of intersection between {} and {}: {:d}, predicted score: {:.4f}, label: {:d}".format(
            row_id + 1, row['drugbank_id_1'], row['drugbank_id_2'], num_intersection, predicted_score, label
        ))

    result_df['inter_size'] = inter_sizes
    result_df.to_csv(dst_filepath)


def generate_sorted_drug_list():
    drug_drug_similarity_path = GROUND_TRUTH_PATH
    drug_list_path = DRUG_LIST_PATH
    drug_list_filepath = os.path.join(drug_list_path, "drug_list.csv")
    drug_drug_similarity_filepath = os.path.join(drug_drug_similarity_path, "drug_drug_matrix.csv")

    drug_list_df = pd.read_csv(drug_list_filepath, index_col=0)
    drug_drug_similarity_df = pd.read_csv(drug_drug_similarity_filepath, index_col=0)

    num_interaction_list = list()
    for row_id, row in drug_list_df.iterrows():
        cid = row['cid']
        num_interaction = drug_drug_similarity_df[cid].sum()
        num_interaction_list.append(num_interaction)

    drug_list_df['num_interactions'] = num_interaction_list
    sorted_drug_list_df = drug_list_df.sort_values("num_interactions", inplace=False, ascending=False)

    drug_list_filename = "drug_list_sorted.csv"
    drug_list_filepath = os.path.join(drug_list_path, drug_list_filename)
    sorted_drug_list_df.to_csv(drug_list_filepath)


if __name__ == '__main__':
    filename = "ddi_ib_isc35000_test_e_y_probe.csv"
    result_path = "/home/chenx/drug_mining/representation_learning/chainer-chemistry/examples/ddi/output/result_isc35000_ggnn_epoch200_lr0.001_exp0.5_wt0_fph32_conv8_nh_hole_x41"

    # The top 20 drugs with the most number of interacting drugs.
    fig_path = os.path.join(result_path, "Top20Drugs")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    corr_filename = "corr_top20.csv"
    corr_filepath = os.path.join(result_path, corr_filename)
    corr_list = list()
    for drugbank_id, num_interactions in [
        ("DB00338", 509),
        ("DB00316", 495),
        ("DB00451", 493),
        ("DB00448", 468),
        ("DB00863", 466),
        ("DB00186", 463),
        ("DB00641", 459),
        ("DB00215", 458),
        ("DB00695", 457),
        ("DB00425", 457),
        ("DB00722", 456),
        ("DB00213", 455),
        ("DB00264", 449),
        ("DB01001", 448),
        ("DB00295", 439),
        ("DB00635", 439),
        ("DB00813", 438),
        ("DB01050", 435),
        ("DB00758", 435),
        ("DB00945", 435),
    ]:
        logging.info("For {}, number of interactions: {:d}".format(drugbank_id, num_interactions))
        coef, p_val = test_correlation(filename, fig_path, drug_bank_id=drugbank_id)
        item = {
            "drugbank_id": drugbank_id,
            "num_interactions": num_interactions,
            "coef": coef,
            "p_val": p_val,
        }
        corr_list.append(item)

        logging.info('\n')
    corr_df = pd.DataFrame(corr_list)
    corr_df.to_csv(corr_filepath)


    corr_filename = "corr_bottom20.csv"
    corr_filepath = os.path.join(result_path, corr_filename)
    # The bottom 20 drugs with the most number of interacting drugs.
    fig_path = os.path.join(result_path, "Bottom20Drugs")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    corr_list = list()
    for drugbank_id, num_interactions in [
        ("DB00307", 11),
        ("DB00701", 9 ),
        ("DB00591", 9 ),
        ("DB00593", 8 ),
        ("DB00558", 8 ),
        ("DB00543", 7 ),
        ("DB00294", 6 ),
        ("DB01053", 6 ),
        ("DB00353", 6 ),
        ("DB00261", 5 ),
        ("DB00278", 5 ),
        ("DB00419", 5 ),
        ("DB00632", 4 ),
        ("DB00714", 4 ),
        ("DB01019", 3 ),
        ("DB00671", 3 ),
        ("DB00548", 3 ),
        ("DB00917", 2 ),
        ("DB01265", 2 ),
        ("DB00934", 1 ),
    ]:
        logging.info("For {}, number of interactions: {:d}".format(drugbank_id, num_interactions))
        coef, p_val = test_correlation(filename, fig_path, drug_bank_id=drugbank_id)
        item = {
            "drugbank_id": drugbank_id,
            "num_interactions": num_interactions,
            "coef": coef,
            "p_val": p_val,
        }
        corr_list.append(item)

        logging.info('\n')
    corr_df = pd.DataFrame(corr_list)
    corr_df.to_csv(corr_filepath)


    corr_filename = "corr_middle20.csv"
    corr_filepath = os.path.join(result_path, corr_filename)
    # The middle 20 drugs with the most number of interacting drugs.
    fig_path = os.path.join(result_path, "Middle20Drugs")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    corr_list = list()
    for drugbank_id, num_interactions in [
        ("DB00582", 162),
        ("DB00951", 160),
        ("DB00804", 160),
        ("DB00862", 160),
        ("DB00853", 159),
        ("DB01005", 158),
        ("DB00495", 158),
        ("DB00440", 158),
        ("DB00612", 157),
        ("DB00489", 156),
        ("DB00877", 156),
        ("DB00542", 153),
        ("DB00889", 152),
        ("DB01072", 151),
        ("DB01045", 150),
        ("DB01026", 149),
        ("DB00757", 149),
        ("DB01085", 148),
        ("DB01204", 147),
        ("DB00277", 147),
    ]:
        logging.info("For {}, number of interactions: {:d}".format(drugbank_id, num_interactions))
        coef, p_val = test_correlation(filename, fig_path, drug_bank_id=drugbank_id)
        item = {
            "drugbank_id": drugbank_id,
            "num_interactions": num_interactions,
            "coef": coef,
            "p_val": p_val,
        }
        corr_list.append(item)

        logging.info('\n')
    corr_df = pd.DataFrame(corr_list)
    corr_df.to_csv(corr_filepath)





