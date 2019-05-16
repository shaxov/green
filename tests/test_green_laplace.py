import unittest
import numpy as np
from numpy import sinh as sh
import green


class GreenForLaplaceOperatorTest(unittest.TestCase):
    def test_line_segment_laplace(self):
        def line_segment(a, b):
            def grin_function(x, s):
                x_, s_ = x, s
                result = np.zeros_like(s_)
                idx = np.logical_and(x_ >= a, x_ <= s_)
                result[idx] = ((x_[idx] - a) * (b - s_[idx])) / (b - a)
                idx = np.logical_and(x_ >= s_, x_ <= b)
                result[idx] = ((s_[idx] - a) * (b - x_[idx])) / (b - a)
                return result
            return grin_function

        a, b = -7.56, 6.35
        gr = line_segment(a, b)

        x = np.random.rand(100) * 10 - 5
        s = np.random.rand(100) * 10 - 5

        is_equal = np.allclose(gr(x, s), green.line_segment(x, s, ab=(a, b), operator="laplace"))
        self.assertTrue(is_equal)

    def test_line_segment_gelmgols(self):
        def line_segment(a, b, kappa):
            def grin_function(x, s):
                x_, s_ = x, s
                result = np.zeros_like(s_)
                idx = np.logical_and(x_ >= a, x_ <= s_)
                result[idx] = (sh(kappa * (x_[idx] - a)) * sh(kappa * (b - s_[idx]))) / (kappa * sh(kappa * (b - a)))
                idx = np.logical_and(x_ >= s_, x_ <= b)
                result[idx] = (sh(kappa * (s_[idx] - a)) * sh(kappa * (b - x_[idx]))) / (kappa * sh(kappa * (b - a)))
                return result
            return grin_function

        a, b, kappa = -7.56, 6.35, 1.2
        gr = line_segment(a, b, kappa)

        x = np.random.rand(100) * 10 - 5
        s = np.random.rand(100) * 10 - 5

        is_equal = np.allclose(gr(x, s), green.line_segment(x, s, ab=(a, b), kappa=kappa, operator="gelmgols"))
        self.assertTrue(is_equal)

    def test_square_laplace(self):
        def square(a, b, n):
            def grin_function(x, s):
                P = (np.pi * np.arange(1, n + 1)) / a
                Q = (np.pi * np.arange(1, n + 1)) / b
                Pm, Qm = np.meshgrid(P, Q)
                Pm = np.tile(Pm[:, :, np.newaxis, np.newaxis], (1, 1, s.shape[1], s.shape[2]))
                Qm = np.tile(Qm[:, :, np.newaxis, np.newaxis], (1, 1, s.shape[1], s.shape[2]))
                return (4 / (a * b)) * np.sum(np.sum(
                    (np.sin(Pm * x[0]) * np.sin(Qm * x[1]) * np.sin(Pm * s[0]) * np.sin(Qm * s[1])) / (
                                np.power(Pm, 2) + np.power(Qm, 2)), axis=0), axis=0)
            return grin_function

        a, b, n = 1, 1, 5
        gr = square(a, b, n)

        x1 = np.random.rand(16, 4)
        x2 = np.random.rand(16, 4)
        s1 = np.random.rand(16, 4)
        s2 = np.random.rand(16, 4)
        x = np.array([x1, x2])
        s = np.array([s1, s2])

        is_equal = np.allclose(gr(x, s), green.square(x1, x2, s1, s2, ab=(a, b), n=n, operator="laplace"))
        self.assertTrue(is_equal)

    def test_square_gelmgols(self):
        def square(a, b, n, kappa):
            def grin_function(x, s):
                P = (np.pi * np.arange(1, n + 1)) / a
                Q = (np.pi * np.arange(1, n + 1)) / b
                Pm, Qm = np.meshgrid(P, Q)
                Pm = np.tile(Pm[:, :, np.newaxis, np.newaxis], (1, 1, s.shape[1], s.shape[2]))
                Qm = np.tile(Qm[:, :, np.newaxis, np.newaxis], (1, 1, s.shape[1], s.shape[2]))
                Kappa = np.ones_like(Qm)
                Kappa *= kappa
                return (4 / (a * b)) * np.sum(np.sum(
                    (np.sin(Pm * x[0]) * np.sin(Qm * x[1]) * np.sin(Pm * s[0]) * np.sin(Qm * s[1])) / (
                                np.power(Pm, 2) + np.power(Qm, 2) + np.power(Kappa, 2)), axis=0), axis=0)
            return grin_function

        a, b, n, kappa = 1, 1, 5, 2.5
        gr = square(a, b, n, kappa)

        x1 = np.random.rand(16, 4)
        x2 = np.random.rand(16, 4)
        s1 = np.random.rand(16, 4)
        s2 = np.random.rand(16, 4)
        x = np.array([x1, x2])
        s = np.array([s1, s2])

        is_equal = np.allclose(gr(x, s), green.square(x1, x2, s1, s2, ab=(a, b), n=n, kappa=kappa, operator="gelmgols"))
        self.assertTrue(is_equal)
