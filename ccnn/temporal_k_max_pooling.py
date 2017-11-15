from chainer import Function
import numpy as np


class TemporalKMaxPooling(Function):
    """ The temporal K max pooling selects the k highest values of an input vector
    while keeping the ordering of the elements in tact.
    """
    def __init__(self, k):
        self.k = k
        self.indexes = None

    def forward_cpu(self, x):
        x, = x
        assert len(x.shape) == 2, ("Temperal K max pooling only support 2d input, got %d" % len(x.shape))
        N, D = x.shape
        assert x.shape[1] >= self.k, "Input of temporal K-max pooling layer should be equal or higher then k (k=%d, %d given)" % (self.k , x.shape[1])
        k = np.partition(x, D-self.k,axis=1)[:, D-self.k]
        NE = np.ones(N,dtype=np.int32) * self.k
        self.indexes = np.empty((N, self.k), dtype=np.int32)
        vals = np.empty((N, self.k), dtype=x.dtype)
        for n in xrange(N):
            for i in xrange(D):
                if x[n,i] > k[n]:
                    NE[n] -= 1
            ki = 0

            for i in xrange(D):
                if x[n,i] > k[n]:
                    self.indexes[n, ki] = i
                    vals[n, ki] = x[n,i]
                    ki += 1
                if x[n,i] == k[n] and NE[n] > 0:
                    self.indexes[n, ki] = i
                    vals[n, ki] = x[n, i]
                    ki += 1
                    NE[n] -= 1

                if ki >= self.k:
                    break
        return vals,

    def backward_cpu(self, x, grad_outputs):
        x, = x
        gx = np.zeros(x.shape, dtype=x.dtype)
        gy, = grad_outputs
        for n in range(x.shape[0]):
            gx[n,self.indexes[n]] = gy[n]
        return gx,


def temporal_k_max_pooling(x, k):
    y = TemporalKMaxPooling(k)(x)
    return y

if __name__ == '__main__':
    from chainer import Variable
    x = np.random.randint(0, 9, size=(3, 100))
    x[0, [0, 5, 6, 35, 65]] = 11
    x[1] = [0 if i < 96 else 1 for i in xrange(100)]
    x[2, [1, 44, 55, 98]] = 10
    x[2, [4, 22, 99]] = 11
    x = Variable(np.array(x, dtype=np.float32))
    y = temporal_k_max_pooling(x, 5)


    y.grad = np.ones((3,5), dtype=np.float32)
    y.backward()
    print x.grad



