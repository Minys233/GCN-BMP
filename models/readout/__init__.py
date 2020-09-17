#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/15/2019 1:36 PM
# @Author  : chinshin
# @FileName: __init__.py.py

import sys
from os.path import abspath, dirname
ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_PATH)

from models.readout.ggnn_readout import GGNNReadout
from models.readout.set2set import Set2Set
from models.readout.mpnn_readout import MPNNReadout

