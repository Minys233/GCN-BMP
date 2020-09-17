#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/26/2019 4:07 PM
# @Author  : chinshin
# @FileName: machine_learning_methods.py
"""

input: ecfp4, ssp, mol2vec
model: rf, lr, svm

"""
import os
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn import metrics

from setting import  *


def parse_arguments():
    feature_list = ["mol2vec", "ecfp4", "ssp", "gnn"]
    model_list = ["rf", "lr", "svm"]
    dataset_list = ["isc35000"]

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="isc35000",
                        choices=dataset_list)
    parser.add_argument("--feature", type=str, default="mol2vec",
                        choices=feature_list)
    parser.add_argument("--model", type=str, default="svm",
                        choices=model_list)
    parser.add_argument("--augment", type=str, default="True")

    return parser.parse_args()


def load_data_for_training(args):
    if args.dataset == "isc35000":
        filename = "ddi_ib_isc35000_{part}.csv"
        train_filepath = os.path.join(INTERACTION_ISC_PATH, filename.format(part="train"))
        train_data = pd.read_csv(train_filepath, index_col=0)
        x_train = zip(train_data["smiles_1"], train_data["smiles_2"])
        y_train = train_data["label"]
        valid_filepath = os.path.join(INTERACTION_ISC_PATH, filename.format(part="valid"))
        valid_data = pd.read_csv(valid_filepath, index_col=0)
        x_valid = zip(valid_data["smiles_1"], valid_data["smiles_2"])
        y_valid = valid_data["label"]
        return x_train, y_train, x_valid, y_valid
    else:
        raise ValueError("Dataset is not valid.")


def extract_feature(x_train, x_valid, args):
    feature_path = DRUG_LIST_PATH
    if args.feature == "mol2vec":
        feature_filepath = os.path.join(feature_path, "smiles2vec.pkl")
    elif args.feature == "ecfp4":
        feature_filepath = os.path.join(feature_path, "smiles2ecfp4.pkl")
    elif args.feature == "ssp":
        feature_filepath = os.path.join(feature_path, "smiles2ssp.pkl")
    elif args.feature == "gnn":
        feature_filepath = os.path.join(feature_path, "smiles2repr.pkl")
    else:
        raise ValueError("Feature is not valid.")

    with open(feature_filepath, "rb") as reader:
        smiles2feature = pickle.load(reader)

    feat_train, feat_valid = list(), list()
    for train_sample in x_train:
        train_feat = np.concatenate((smiles2feature[train_sample[0]], smiles2feature[train_sample[1]]), axis=0)
        feat_train.append(train_feat)

    for valid_sample in x_valid:
        valid_feat = np.concatenate((smiles2feature[valid_sample[0]], smiles2feature[valid_sample[1]]), axis=0)
        feat_valid.append(valid_feat)

    feat_train = np.array(feat_train, dtype=np.float32)
    feat_valid = np.array(feat_valid, dtype=np.float32)
    return feat_train, feat_valid


def train_model(feat_train, y_train, feat_valid, y_valid, args):
    model = None
    if args.model == "svm":
        model = SVC()
    elif args.model == "rf":
        model = RandomForestClassifier(n_estimators=100)
    elif args.model == "lr":
        model = LogisticRegression(C=0.00001, penalty='L2')

    print("Feature: {feature}, Model: {model}".format(
        feature=args.feature, model=args.model
    ))

    # Train model
    model.fit(feat_train, y_train)

    # predict model
    y_score = np.max(model.predict_proba(feat_valid), axis=1)
    y_pred = np.round(y_score)
    # auroc:
    auroc = metrics.roc_auc_score(y_valid, y_score)
    # auprc:
    auprc = metrics.average_precision_score(y_valid, y_score)
    # f1:
    f1_score = metrics.f1_score(y_valid, y_pred)
    print("=== Classification Report ===")
    print("AUROC: {:.4f}, AUPRC: {:.4f}, F1: {:.4f}".format(
        auroc, auprc, f1_score
    ))

    # save model
    joblib.dump(model, "{feature}_{model}.m".format(
        feature=args.feature, model=args.model
    ))


def main():
    args = parse_arguments()
    augment = True if args.augment == "True" else False

    smiles_train, y_train, smiles_valid, y_valid = load_data_for_training(args)
    feat_train, feat_valid = extract_feature(smiles_train, smiles_valid, args)
    if augment:
        data_train = list(zip(feat_train, y_train))
        ano_data_train = data_train
        data_train = data_train + ano_data_train
        np.random.shuffle(data_train)
        feat_train, y_train = zip(*data_train)
        feat_train, y_train = np.array(feat_train, dtype=np.float32), np.array(y_train, dtype=np.int32)

    train_model(feat_train, y_train, feat_valid, y_valid, args)


if __name__ == "__main__":
    main()



