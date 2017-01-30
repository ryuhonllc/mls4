#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np  # noqa


class TfImp:

    """
    Tensorflow implementation of functions for unit test.
    Implement these to get tests to pass.
    """

    def give_42(self):
        """
        TensorFlow uses sessions to run code.
        All you have to do to make this pass
        is return a constant
        """
        return tf.constant(42)

    def give_float_42(self):
        """
        TensorFlow can be picked about types.
        Return a floating point constant
        """
        # return tf.constant(42.0)
        return tf.constant(42, dtype=tf.float32)

    def convert_np_array(self, npa):
        """
        given a numpy array
        convert it to a TF constant tensor
        """
        return tf.constant(npa)

    def sum_constant_vector(self):
        """
        create a pipeline with a constant array of 5 element
        that sums to 42 when run
        """
        c = tf.constant([10, 10, 10, 10, 2])
        s = tf.reduce_sum(c)
        return c, s

    def sum_constant_matrix(self):
        """
        make this one 5x5
        still sums to 42
        """
        a = np.array([2, 2, 2, 2, .4] * 5).reshape((5, 5))
        c = tf.constant(a)
        s = tf.reduce_sum(c)
        return c, s

    def sum_constant_tensor(self):
        """
        make this one 5x5x5.
        still sums to 42
        Confused about what tensors are? that's all there is to it
        """
        a = np.array([2, 2, 2, 2, .4] * 25).reshape((5, 5, 5))
        a /= 5
        c = tf.constant(a)
        s = tf.reduce_sum(c)
        return c, s

    def variable_vector(self):
        """
        as constant vector but make placeholder variables
        the unit test will determine which values to try
        """
        return tf.Variable(tf.zeros(5, dtype=tf.int32))

    def add_variable(self, x, y):
        """
        add one variable to another
        """
        added = tf.add(x, y)
        new_val = tf.assign(x, added)
        return new_val

    def logistic(self, x):
        """
        unlike the numpy examples
        with TF, you will return a pipeline
        the unit test will then run computations
        using that pipeline
        """
        # easy mode
        # return tf.sigmoid(x)
        return tf.divide(1., tf.add(1., tf.exp(-x)))

    def variable_sum_plus_logistic_squash(self, x):
        """
        This is very close to a single perceptron.
        Create a pipeline that takes a variable array
        sums it, and then squashes it to a single
        value using a logistic function
        """
        s = tf.reduce_sum(x)
        return self.logistic(s)


class NpImp:

    """
    Numpy implementation of similar stuff
    If you did the previous session these will make sense
    """

    def give_42(self):
        return np.array([42])

    def sum_constant_vector():
        x = np.array([21, 21, 0, 1, -1])
        return x.sum()

    def sum_variable_vector(x):
        return x.sum()

    def logistic(x):
        """
        given an array, process the input via a logistic function and return
        """


if __name__ == "__main__":
    """
    speed comparison between numpy and TF.
    numpy will probably be faster except for large sets with a GPU
    """
    pass
