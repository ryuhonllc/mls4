#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from tfimp import TfImp as imp
import tensorflow as tf

import unittest


class TestTfImp(unittest.TestCase):

    def setUp(self):
        o = imp()
        self.o = o

    def test_give_42(self):
        with tf.Session() as sess:
            c = self.o.give_42()
            assert sess.run(c) == 42

    def test_give_float_42(self):
        c = self.o.give_float_42()
        assert c.dtype == tf.float32

    def test_convert_np_array(self):
        for i in range(1, 6):
            dim = [5] * i
            a = np.random.normal(size=5**i).reshape(dim)
            out = self.o.convert_np_array(a)
            assert out.get_shape() == dim
            assert out.name == ("Const_%d:0" % i)

    def test_sum_constant_vector(self):
        with tf.Session() as sess:
            c, s = self.o.sum_constant_vector()
            assert c.get_shape() == (5,)
            val = sess.run(s)
            assert val == 42

    def test_sum_constant_matrix(self):
        with tf.Session() as sess:
            c, s = self.o.sum_constant_matrix()
            assert c.get_shape() == (5, 5)
            val = sess.run(s)
            assert round(val - 42) == 0

    def test_sum_constant_tensor(self):
        with tf.Session() as sess:
            c, s = self.o.sum_constant_tensor()
            assert c.get_shape() == (5, 5, 5)
            val = sess.run(s)
            assert round(val - 42) == 0

    def test_variable_vector(self):
        with tf.Session() as sess:
            v = self.o.variable_vector()
            sess.run(tf.global_variables_initializer())
            c = tf.constant([-42, 42, 46, -2, -2])
            modded = tf.assign(v, c)
            summed = tf.reduce_sum(modded)
            result = sess.run(summed)
            assert result == 42

    def test_add_variable(self):
        with tf.Session() as sess:
            x = tf.Variable(41)
            y = tf.constant(1)
            v = self.o.add_variable(x, y)
            sess.run(tf.global_variables_initializer())
            result = sess.run(v)
            assert result == 42

    def test_logistic(self):
        with tf.Session() as sess:
            expected = tf.constant(
                [0.12, 0.18, 0.27, 0.38, 0.5, 0.62, 0.73, 0.82])
            x = tf.range(-2, 2, 0.5)
            out = self.o.logistic(x)
            zero = tf.sub(out, expected)
            result = sess.run(zero)
            for val in result:
                assert round(val, 2) == 0

    def test_variable_sum_plus_logistic_squash(self):
        with tf.Session() as sess:
            expected = 0.82
            x = tf.constant([-1., -2, -3, 7.516])
            out = self.o.variable_sum_plus_logistic_squash(x)
            result = sess.run(out)
            assert round(result - expected, 2) == 0
