#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/14/2019 4:02 PM
# @Author  : chinshin
# @FileName: __init__.py.py

import sys
from os.path import abspath, dirname
ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_PATH)

from models.link_prediction.bilinear import Bilinear
from models.link_prediction.hole import HOLE
from models.link_prediction.dist_mult import DistMult