#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/28/2019 5:44 PM
# @Author  : chinshin
# @FileName: data_validator.py
import os
import pandas as pd
from rdkit import Chem
import warnings
from pandas import DataFrame
from os.path import dirname, abspath

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
from setting import *

warnings.filterwarnings('ignore')


filename = 'drug_list.csv'
filepath = os.path.join(DRUG_LIST_PATH, filename)

df = pd.read_csv(filepath)

for index, smiles in enumerate(df['smiles']):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('index: {}, None'.format(index))
    else:
        print('index: {}, not None'.format(index))