#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/15/2019 1:36 PM
# @Author  : chinshin
# @FileName: __init__.py.py
import sys
from os.path import abspath, dirname
ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_PATH)

from models.update.ggnn_update import GGNNUpdate
from models.update.relgcn_update import RelGCNUpdate
from models.update.mpnn_update import MPNNUpdate