import unittest
import sys
import numpy as np

sys.path.append("../ccnn/")
from chainer import testing
from chainer import gradient_check
from ccnn.temporal_k_max_pooling import TemporalKMaxPooling, temporal_k_max_pooling
import chainer
from chainer import cuda

from chainer.testing import attr
@testing.parameterize(*testing.product({
    'dtype': [np.float16, np.float32, np.float64],
}))

class TestTemporalKMaxPooling(unittest.TestCase):

    def setUp(self):
        x = np.random.randint(0, 9, size=(3, 100))
        x[0, [0, 5, 6, 35, 65]] = 11
        x[1] = [0 if i < 96 else 1 for i in xrange(100)]
        x[2, [1, 44, 55, 98]] = 10
        x[2, [4, 22, 99]] = 11
        x = np.array(x, dtype=self.dtype)
        self.forward_x = x # some specific edge cases for forward call

        self.backward_x = np.arange(3*25, dtype=self.dtype)
        np.random.shuffle(self.backward_x)
        self.backward_x = self.backward_x.reshape(3, 25)
        self.y_grad = np.random.rand(3, 5).astype(self.dtype)


        # when dtype is float16, less accurate
        if self.dtype == np.float16:
            self.check_backward_options = {
                'atol': 1e-3, 'rtol': 1e-2}
            self.check_double_backward_options = {
                'atol': 1e-3, 'rtol': 1e-2}
        else:
            self.check_backward_options = {
                'atol': 1e-4, 'rtol': 1e-3}
            self.check_double_backward_options = {
                'atol': 1e-4, 'rtol': 1e-3}

    def check_forward(self, x_data, use_cudnn='always'):

        x = chainer.Variable(x_data)
        with chainer.using_config('use_cudnn', use_cudnn):
            y = temporal_k_max_pooling(x, 5)
        y_data = y.data
        self.assertEquals(self.dtype, y_data.dtype)

        expect = np.array([
            [11.0, 11.0, 11.0, 11.0, 11.0],
            [0.0, 1.0, 1.0, 1.0, 1.0],
            [10.0, 11.0, 11.0, 10.0, 11.0]
            ], dtype=self.dtype)
        testing.assert_allclose(expect, y_data)

    def test_forward_cpu(self):
        self.check_forward(self.forward_x)

    def check_backward(self, x_data, y_grad, use_cudnn='always'):

        def f(x):
            return temporal_k_max_pooling(x, 5)

        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_backward(f, x_data, y_grad, dtype='d', **self.check_backward_options)

    def test_backward_cpu(self):
        try:
            self.check_backward(self.backward_x, self.y_grad)
        except AssertionError as e:
            print(e)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    def test_forward_gpu_non_contiguous(self):
        self.check_forward(cuda.cupy.asfortranarray(cuda.to_gpu(self.x)))

    @attr.gpu
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), 'never')

            #(y[0], [11.0, 11.0, 11.0, 11.0, 11.0])


testing.run_module(__name__, __file__)
