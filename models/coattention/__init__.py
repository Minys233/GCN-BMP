#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/5/2019 5:41 PM
# @Author  : chinshin
# @FileName: __init__.py.

import sys
from os.path import abspath, dirname
ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, ROOT_PATH)

from models.coattention.alternating_coattention import AlternatingCoattention
from models.coattention.parallel_coattention import ParallelCoattention
from models.coattention.global_coattention import GlobalCoattention