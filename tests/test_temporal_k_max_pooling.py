import unittest
import sys
import numpy as np

sys.path.append("../ccnn/")
from chainer import testing
from chainer import gradient_check
from ccnn.temporal_k_max_pooling import TemporalKMaxPooling, temporal_k_max_pooling
class TestTemporalKMaxPooling(unittest.TestCase):
    def test_forward_cpu(self):
        x = np.random.randint(0, 9, size=(3, 100))
        x[0, [0, 5, 6, 35, 65]] = 11
        x[1] = [0 if i < 96 else 1 for i in xrange(100)]
        x[2, [1, 44, 55, 98]] = 10
        x[2, [4, 22, 99]] = 11
        tmp = TemporalKMaxPooling(k=5)
        x = np.array(x, dtype=np.float64)
        y, = tmp.forward_cpu((x,))
        self.assertTrue(np.array_equal(y[0], [11.0, 11.0, 11.0, 11.0, 11.0]))
        self.assertTrue(np.array_equal(y[1], [0.0, 1.0, 1.0, 1.0, 1.0]))
        self.assertTrue(np.array_equal(y[2],  [10.0, 11.0, 11.0, 10.0, 11.0]))

    def test_backward_cpu(self):

        x = np.random.randn(3, 100).astype(np.float32)

        def f(x):
            return temporal_k_max_pooling(x, 5)

        y_grad = np.random.randn(3, 5).astype(np.float32)

        gradient_check.check_backward(f, x, y_grad)

    #(y[0], [11.0, 11.0, 11.0, 11.0, 11.0])
testing.run_module(__name__, __file__)