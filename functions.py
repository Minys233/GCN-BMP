#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/27/2019 2:12 PM
# @Author  : chinshin
# @FileName: functions.py

import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.functions.activation import sigmoid
from chainer import utils
from chainer.utils import type_check


class SigmoidFocalLoss(function_node.FunctionNode):

    """Sigmoid activation followed by a sigmoid cross entropy loss."""

    ignore_label = -1

    def __init__(self, gamma=2.0, normalize=True, reduce='mean'):
        self.gamma = gamma
        self.normalize = normalize
        if reduce not in ('mean', 'no'):
            raise ValueError(
                "only 'mean' and 'no' are valid for 'reduce', but '%s' is "
                'given' % reduce)
        self.reduce = reduce
        self.count = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype.kind == 'i',
            x_type.shape == t_type.shape
        )

    def forward(self, inputs):
        self.retain_inputs((0, 1))

        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        self.ignore_mask = (t != self.ignore_label)

        # stable computation of the cross entropy.
        # loss = -(
        #     self.ignore_mask *
        #     (x * (t - (x >= 0)) - xp.log1p(xp.exp(-xp.abs(x)))))
        prob = chainer.functions.sigmoid(x)
        if t == 1:
            prob_t = prob
        else:
            prob_t = 1 - prob
        loss = -(
            self.ignore_mask * xp.log(prob_t))

        if not self.reduce == 'mean':
            return utils.force_array(loss.astype(x.dtype)),

        if self.normalize:
            count = xp.maximum(1, self.ignore_mask.sum())
        else:
            count = max(1, len(x))
        self.count = count

        return utils.force_array(
            xp.divide(xp.sum(loss), self.count, dtype=x.dtype)),

    def backward(self, inputs, grad_outputs):
        x, t = self.get_retained_inputs()
        gy, = grad_outputs
        gx, = SigmoidFocalLossGrad(
            self.reduce, self.count, self.ignore_mask, t.data).apply((x, gy))
        return gx, None


class SigmoidFocalLossGrad(function_node.FunctionNode):

    def __init__(self, gamma, reduce, count, ignore_mask, t):
        self.gamma = gamma
        self.reduce = reduce
        self.count = count
        self.ignore_mask = ignore_mask
        self.t = t

    def forward(self, inputs):
        self.retain_inputs((0, 1))

        xp = cuda.get_array_module(*inputs)
        x, gy = inputs

        y, = sigmoid.Sigmoid().forward((x,))
        prob = y
        if self.reduce == 'mean':
            gx = xp.divide(
                gy * self.ignore_mask * (prob_t - self.t), self.count,
                dtype=y.dtype)
        else:
            gx = (gy * self.ignore_mask * (prob_t - self.t)).astype(y.dtype)

        return gx,

    def backward(self, indexes, grad_outputs):
        ggx, = grad_outputs
        x, gy = self.get_retained_inputs()
        y = chainer.functions.sigmoid(x)
        yp = y * (1 - y)
        gx = yp * chainer.functions.broadcast_to(gy, yp.shape)
        ggy = y - self.t.astype(y.dtype)
        gx *= self.ignore_mask * ggx
        ggy *= self.ignore_mask * ggx

        if self.reduce == 'mean':
            gx /= self.count
            ggy = chainer.functions.sum(ggy) / self.count

        return gx, ggy


def sigmoid_focal_loss(x, t, normalize=True, reduce='mean'):
    return SigmoidFocalLoss(normalize, reduce).apply((x, t))[0]
