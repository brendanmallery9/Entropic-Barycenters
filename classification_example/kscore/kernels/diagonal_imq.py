#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

from .diagonal import Diagonal
from kscore.utils import median_heuristic

class DiagonalIMQ(Diagonal):

    def __init__(self, kernel_hyperparams=None, heuristic_hyperparams=median_heuristic):
        super().__init__(kernel_hyperparams, heuristic_hyperparams)

    def _gram_impl(self, x, y, kernel_width):
        d = tf.shape(x)[-1]
        x_m = tf.expand_dims(x, -2)  # [M, 1, d]
        y_m = tf.expand_dims(y, -3)  # [1, N, d]
        diff = x_m - y_m
        dist2 = tf.reduce_sum(diff * diff, -1) # [M, N]
        imq = tf.math.rsqrt(1 + dist2 / kernel_width ** 2)
        divergence = tf.expand_dims(imq ** 3, -1) * (diff / kernel_width ** 2)

        return imq, divergence

class DiagonalIMQp(Diagonal):

    def __init__(self, p=0.5, kernel_hyperparams=None,
            heuristic_hyperparams=median_heuristic):
        super().__init__(kernel_hyperparams, heuristic_hyperparams)
        self._p = p

    def _gram_impl(self, x, y, kernel_width):
        d = tf.shape(x)[-1]
        x_m = tf.expand_dims(x, -2)  # [M, 1, d]
        y_m = tf.expand_dims(y, -3)  # [1, N, d]
        diff = x_m - y_m
        dist2 = tf.reduce_sum(diff * diff, -1) # [M, N]
        imq = 1.0 / (1.0 + dist2 / kernel_width ** 2)
        imq_p = tf.pow(imq, self._p)
        divergence = 2.0 * self._p * tf.expand_dims(imq * imq_p, -1) \
                * (diff / kernel_width ** 2)

        return imq_p, divergence
