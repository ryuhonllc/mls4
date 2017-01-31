#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf  # noqa
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
        pass

    def give_float_42(self):
        """
        TensorFlow can be picked about types.
        Return a floating point constant
        """
        pass

    def convert_np_array(self, npa):
        """
        given a numpy array
        convert it to a TF constant tensor
        """
        pass

    def sum_constant_vector(self):
        """
        create a pipeline with a constant array of 5 element
        that sums to 42 when run
        """
        pass

    def sum_constant_matrix(self):
        """
        make this one 5x5
        still sums to 42
        """
        pass

    def sum_constant_tensor(self):
        """
        make this one 5x5x5.
        still sums to 42
        Confused about what tensors are? that's all there is to it
        """
        pass

    def variable_vector(self):
        """
        as constant vector but make placeholder variables
        the unit test will determine which values to try
        """
        pass

    def add_variable(self, x, y):
        """
        add one variable to another
        """
        pass

    def logistic(self, x):
        """
        unlike the numpy examples
        with TF, you will return a pipeline
        the unit test will then run computations
        using that pipeline
        """
        pass

    def variable_sum_plus_logistic_squash(self, x):
        """
        This is very close to a single perceptron.
        Create a pipeline that takes a variable array
        sums it, and then squashes it to a single
        value using a logistic function
        """
        pass
