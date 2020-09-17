#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/12/2018 4:46 PM
# @Author  : chinshin
# @FileName: precision_evaluator.py
import numpy

from chainer.dataset import convert
from sklearn import metrics

# from chainer_chemistry.training.extensions.batch_evaluator import BatchEvaluator  # NOQA
from batch_evaluator import BatchEvaluator


def _to_list(a):
    """convert value `a` to list

    Args:
        a: value to be convert to `list`

    Returns (list):

    """
    if isinstance(a, (int, float)):
        return [a, ]
    else:
        # expected to be list or some iterable class
        return a


class RecallEvaluator(BatchEvaluator):

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, name=None,
                 pos_labels=1, ignore_labels=None, raise_value_error=True,
                 logger=None):
        metrics_fun = {'recall': self.precision_score}
        super(RecallEvaluator, self).__init__(
            iterator, target, converter=converter, device=device,
            eval_hook=eval_hook, eval_func=eval_func, metrics_fun=metrics_fun,
            name=name, logger=logger)

        self.pos_labels = _to_list(pos_labels)
        self.ignore_labels = _to_list(ignore_labels)
        self.raise_value_error = raise_value_error

    def precision_score(self, y_total, t_total,
                        labels=None, pos_label=1,
                        average='binary', sample_weight=None):
        # --- ignore labels if specified ---
        if self.ignore_labels:
            valid_ind = numpy.in1d(t_total, self.ignore_labels, invert=True)
            y_total = y_total[valid_ind]
            t_total = t_total[valid_ind]

        # # --- set positive labels to 1, negative labels to 0 ---
        # pos_indices = numpy.in1d(t_total, self.pos_labels)
        # t_total = numpy.where(pos_indices, 1, 0)
        #
        # if len(numpy.unique(t_total)) != 2:
        #     if self.raise_value_error:
        #         raise ValueError("Only one class present in y_true. PRC AUC "
        #                          "score is not defined in that case.")
        #     else:
        #         return numpy.nan

        # y_total = numpy.where(y_total > 0, 1, 0)
        # y_total = numpy.asarray(y_total, dtype=numpy.int32)
        recall = metrics.recall_score(
            y_true=t_total, y_pred=y_total,
            labels=labels, pos_label=pos_label,
            average=average, sample_weight=sample_weight
        )
        return recall
