#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `newton_raphson` package."""


import unittest
import numpy as np
from newton_raphson.newton_raphson import *
from newton_raphson import logistic_regression


class TestNewton_raphson(unittest.TestCase):
    """Tests for `newton_raphson` package."""

    def setUp(self):
        self.rng = np.random.RandomState(235)
        self.n   = 100
        self.k   = 4
        self.X   = self.rng.normal(0., 1, size=(self.n, self.k))
        self.y   = self.rng.binomial(1, p=0.5, size=self.n)

    def test_intercept_padding(self):
        X = add_intercept_column_to_feature_matrix(self.X)
        self.assertEqual((self.X.shape[0], self.X.shape[1]+1), X.shape)

    def test_init_parameters(self):
        self.assertTrue(np.all(np.equal(initialize_parameters(5), np.zeros(5))))

    def test_f_logistic(self):
        X = add_intercept_column_to_feature_matrix(self.X)
        w = initialize_parameters(n=X.shape[1])
        yhat = f_logistic(X, w)
        self.assertTrue(yhat.size, self.n)
        self.assertTrue(np.all(np.less_equal(yhat, 1)))
        self.assertTrue(np.all(np.greater_equal(yhat, 0)))

    def test_negative_log_likelihood(self):
        X = add_intercept_column_to_feature_matrix(self.X)
        w = initialize_parameters(n=X.shape[1])
        yhat = f_logistic(X, w)
        J = negative_log_likelihood(self.y, yhat)
        self.assertTrue(J.size, yhat.size)
        self.assertTrue(np.all(np.greater_equal(J, 0)))

    def test_compute_hessian(self):
        # TODO: Add unit test for scaling factor
        X = add_intercept_column_to_feature_matrix(self.X)
        w = initialize_parameters(n=X.shape[1])
        yhat = f_logistic(X, w)
        H, s = compute_hessian(X, X.T, self.y, yhat)
        self.assertEqual(H.shape, (w.size, w.size))
        self.assertEqual(s.size, w.size)

    def test_convergence_test(self):
        done, msg = convergence_test(3000,1e-15,1,1e-15,1000,1e-12,1e-12,2)
        self.assertTrue(done)

    def test_logistic_regression(self):
        # TODO: Add unit test for condition number of hessian
        # TODO: Add unit test for t-statistic
        res = logistic_regression(self.X, self.y)
        self.assertTrue(np.greater_equal(res.deviance, 0))
        self.assertTrue(np.greater_equal(res.aic, 0))
        self.assertTrue(np.greater_equal(res.bic, 0))
        self.assertTrue(np.greater_equal(res.p_model, 0))
        self.assertTrue(np.less_equal(res.p_model, 0))
        self.assertTrue(np.all(np.greater_equal(res.p_coefs, 0)))
        self.assertTrue(np.all(np.less_equal(res.p_coefs, 1)))
