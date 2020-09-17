#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/12/2018 11:15 AM
# @Author  : chinshin
# @FileName: file_for_test.py
import numpy as np
import chainer.functions as F

y = np.array([
    [0.1, 0.7],    # 1
    [8.0, 1.0],    # 0
    [0.0, -1.0],   # 0
    [-8.0, -1.0], # 1
], dtype=np.float32)
t = np.array([1, 0, 0, 0], dtype=np.int32)
label_num = np.unique(t).shape[0]
acc = F.accuracy(y, t)
print('Accuracy: ', acc.data)

precision, _ = F.precision(y, t)
print('Precision for positive class: ', precision.data[0])
relcall, _ = F.recall(y, t)
print('Recall for negative class: ', relcall.data[0])
f1, _ = F.f1_score(y, t)
print('F1 score for positive class: ', f1.data[0])