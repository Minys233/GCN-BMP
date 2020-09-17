#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/27/2019 10:53 PM
# @Author  : chinshin
# @FileName: setting.py
"""
Some important static parameters.
"""
import os

ROOT_PATH = os.getcwd()

DATASET_PATH = os.path.join(ROOT_PATH, 'dataset')

DRUG_LIST_PATH = os.path.join(DATASET_PATH, 'drug_list')
GROUND_TRUTH_PATH = os.path.join(DATASET_PATH, 'ground_truth')
INTERACTION_PATH = os.path.join(DATASET_PATH, 'interaction')
SUPER_NODE_PATH = os.path.join(DATASET_PATH, 'super_node')
YEAST_PATH = os.path.join(DATASET_PATH, 'yeast')
KAIST_PATH = os.path.join(DATASET_PATH, 'kaist')
DRUGBANK_PATH = os.path.join(DATASET_PATH, 'drugbank')

INTERACTION_DB_PATH = os.path.join(INTERACTION_PATH, 'drug_based')
INTERACTION_IB_PATH = os.path.join(INTERACTION_PATH, 'inter_based')
INTERACTION_ISC_PATH = os.path.join(INTERACTION_PATH, 'isc')
INTERACTION_SAMPLE_PATH = os.path.join(INTERACTION_PATH, 'sample')

GLOBAL_SEED = 2018

NUM_DRUGS = 544
NUM_INTERACTIONS = int(NUM_DRUGS * (NUM_DRUGS - 1) / 2.)

NUM_DRUGS_KAIST = 1704