from chainer import Function
import numpy as np

from chainer import cuda

class TemporalKMaxPooling(Function):
    """ The temporal K max pooling selects the k highest values of an input vector
    while keeping the ordering of the elements in tact.
    """
    def __init__(self, k):
        self.k = k
        self.indexes = None

    def forward_cpu(self, x):
        x, = x
        # Only works for two dimensional input for now
        assert len(x.shape) == 2, ("Temperal K max pooling only support 2d input, got %d" % len(x.shape))
        N, D = x.shape
        assert x.shape[1] >= self.k, "Input of temporal K-max pooling layer should be equal or higher then k (k=%d, %d given)" % (self.k , x.shape[1])

        k = np.partition(x, D-self.k, axis=1)[:, D-self.k:]
        # k[n], will contain the k highest element but not necessarily in the correct order
        # k[n][0] will be kth highest element

        # keeps track of max number of elements that can be equal to the k-th element
        # this will prevent the top-3 of [1,1,1,2] to be [1,1,1]
        # begin by assuming all k elements are equal and we thus want k elements to be equal
        ne = np.ones(N, dtype=np.int32) * self.k

        # indexes and values of array
        self.indexes = np.empty((N, self.k), dtype=np.int32) # indexes should be kept for back propagation
        vals = np.empty((N, self.k), dtype=x.dtype)

        for n in xrange(N):
            # for every element after the kth element check if it's higher, a higher element means
            # that we should select one item  equal to the  kth highest element less
            for i in xrange(1, self.k):
                if k[n, i] > k[n][0]:
                    ne[n] -= 1

            # select higher or equal to the k-th highest element in the order they exist in the data
            ki = 0 # number of elements we already found
            for i in xrange(D):
                # Higher then k-th element, select it anyway
                if x[n, i] > k[n][0]:
                    self.indexes[n, ki] = i
                    vals[n, ki] = x[n,i]
                    ki += 1

                # If it's equal, check that we are not exceeding or maximum
                if x[n, i] == k[n][0] and ne[n] > 0:
                    self.indexes[n, ki] = i
                    vals[n, ki] = x[n, i]
                    ki += 1

                    ne[n] -= 1  # One less to select

                if ki >= self.k:
                    break
        return vals,

    def backward_cpu(self, x, grad_outputs):
        x, = x
        gx = np.zeros(x.shape, dtype=x.dtype)

        gy, = grad_outputs
        for n in range(x.shape[0]):
            gx[n, self.indexes[n]] = gy[n]

        return gx,

    def forward_gpu(self, x):
        x, = x
        # Only works for two dimensional input for now
        assert len(x.shape) == 2, ("Temperal K max pooling only support 2d input, got %d" % len(x.shape))
        N, D = x.shape
        assert x.shape[
                   1] >= self.k, "Input of temporal K-max pooling layer should be equal or higher then k (k=%d, %d given)" % (
        self.k, x.shape[1])
        # cupy.partition actually does full sort, but might improve in future
        M = cuda.cupy.partition(x, D - self.k, axis=1)[:, D - self.k:]
        # k[n], will contain the k highest element but not necessarily in the correct order
        # k[n][0] will be kth highest element

        # indexes and values of array
        self.indexes = cuda.cupy.empty((N, self.k), dtype=np.int32)  # indexes should be kept for back propagation
        y = cuda.cupy.empty((N, self.k), dtype=x.dtype)
        cuda.elementwise(
            'T x, T m, int32 k, int32 n',
            'T out, int32 indexes',
            '''
                int ne = k;
                for (int ki = 1; ki < k; k++) {
                    if ( m[ki] > m[0] ) {
                        ne--;
                    }
                }
                
                int ki = 0;
                for (int d = 0; d < n; d++){
                    if (x[d] > m[0]) {
                        indexes[ki] = d;
                        out[ki] = x[d];
                        ki++;
                    }
                    
                    if (x[d] == m[0] && ne > 0) {
                        indexes[ki] = d;
                        out[ki] = x[d];
                        ki++;
                        ne--;
                    }
                }
            ''',
            'temp_k_max_fwd'

        )(M, x, self.k, D, y, self.indexes)

        return y,

def temporal_k_max_pooling(x, k):
    y = TemporalKMaxPooling(k)(x)
    return y


if __name__ == '__main__':
    x = np.random.randint(0, 9, size=(3, 100))
    x[0, [0, 5, 6, 35, 65]] = 11
    x[1] = [0 if i < 96 else 1 for i in xrange(100)]
    x[2, [1, 44, 55, 98]] = 10
    x[2, [4, 22, 99]] = 11
    x = cuda.cupy.array(x, dtype=np.float64)
    tmp = TemporalKMaxPooling(3)
    tmp.forward_gpu((x,))


