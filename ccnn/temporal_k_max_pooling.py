from chainer import Function
import numpy as np
from operator import mul
from chainer import cuda
from six.moves import xrange
class TemporalKMaxPooling(Function):
    """ The temporal K max pooling selects the k highest values of an input vector
    while keeping the ordering of the elements in tact. It always assumes that the last dimension is the highest dimension
    """
    def __init__(self, k):
        self.k = k
        self.indexes = None
        self.org_shape = None # original input shape
        self.d = None # Temporal dimension (last) shape
        self.n = None # Number to loop over product of

    def forward(self, inputs):
        # Common re-shaping and for forward pass
        x, = inputs

        # Will reshape back to this
        self.org_shape = x.shape
        self.d = self.org_shape[-1]

        # Checks
        assert x.ndim > 1, ("Temporal K max pooling only support 2d or higher input, got {}".format(x.ndim))
        assert x.shape[-1] >= self.k, "Input of temporal K-max pooling layer should be equal or higher then k (k={}, {} given)".format(self.k , x.shape[-1])

        xp = cuda.get_array_module(*inputs)
        k = xp.partition(x, self.d-self.k, axis=-1)[..., -self.k:]

        inputs = (x.flatten(), k.flatten())
        self.n = inputs[0].shape[0] // self.d

        output = super(TemporalKMaxPooling, self).forward(inputs)
        # reshape back
        (x,) = output
        x = x.reshape(self.org_shape[:-1] + tuple([self.k]))
        return x,

    def forward_cpu(self, inputs):
        x, k = inputs
        # k[n*self.k:(n+1)*self.k], will contain the k highest elements of item n but not necessarily in the correct order
        # k[n*self.k] will be kth highest element

        # keeps track of max number of elements that can be equal to the k-th element
        # this will prevent the top-3 of [1,1,1,2] to be [1,1,1]
        # begin by assuming all k elements are equal and we thus want k elements to be equal

        # indexes and values of array
        self.indexes = np.empty((self.n * self.k), dtype=np.int32) # indexes should be kept for back propagation
        vals = np.empty((self.n * self.k), dtype=x.dtype)

        for n in xrange(self.n):

            start_k = n * self.k
            start_d = n * self.d
            ne = self.k
            # for every element after the kth element check if it's higher, a higher element means
            # that we should select one item  equal to the  kth highest element less
            for ki in xrange(1, self.k):
                if k[start_k + ki] > k[start_k]:
                    ne -= 1

            # select higher or equal to the k-th highest element in the order they exist in the data
            ki = 0 # number of elements we already found
            for di in xrange(self.d):
                # Higher then k-th element, select it anyway
                if x[start_d + di] > k[start_k]:
                    self.indexes[start_k + ki] = start_d + di
                    vals[start_k + ki] = x[start_d + di]
                    ki += 1

                # If it's equal, check that we are not exceeding or maximum
                if x[start_d + di] == k[start_k] and ne > 0:
                    self.indexes[start_k + ki] = start_d + di
                    vals[start_k + ki] = x[start_d + di]
                    ki += 1

                    ne -= 1  # One less to select

                if ki >= self.k:
                    break
        return vals,

    def forward_gpu(self, inputs):
        x, k = inputs

        self.indexes = cuda.cupy.empty(self.n * self.k, dtype=np.int32)  # indexes should be kept for back propagation
        y = cuda.cupy.empty(self.n * self.k, dtype=x.dtype)
        cuda.elementwise(
            'raw T x, raw T m, int32 k, int32 d',
            'raw T out, raw int32 indexes',
            '''
                int start_d = d * i;
                int start_k = k * i;
                int ne = k;
                for (int ki = 1; ki < k; ki++) {
                    if ( m[start_k + ki] > m[start_k] ) {
                        ne = ne-1;
                    }
                }
                indexes[start_k] = m[start_k];
                indexes[start_k+1] = ne;
                int ki = 0;
                for (int di = 0; di < d; di++){
                    if (x[start_d+di] > m[start_k]) {
                        indexes[start_k+ki] = start_d+di;
                        out[start_k+ki] = x[start_d+di];
                        ki++;
                    }
                    
                    if (x[start_d+di] == m[start_k] && ne > 0) {
                        indexes[start_k+ki] = start_d+di;
                        out[start_k+ki] = x[start_d + di];
                        ki++;
                        ne--;
                    }
                }
            ''',
            'temp_k_max_fwd'

        )(x, k, self.k, self.d, y, self.indexes, size=self.n)

        return y,

    def backward(self, x, grad_outputs):
        x, = x
        xp = cuda.get_array_module(x)
        gx = xp.zeros(self.n * self.d, dtype=x.dtype)
        gy, = grad_outputs
        gx[self.indexes] = gy.flatten()
        gx = gx.reshape(x.shape)
        return gx,


def temporal_k_max_pooling(x, k):
    y = TemporalKMaxPooling(k)(x)
    return y


if __name__ == '__main__':
    x = np.random.randint(0, 9, size=(3,1, 100))
    x[0, 0, [0, 5, 6, 35, 65]] = 11
    x[1, 0] = [0 if i < 96 else 1 for i in xrange(100)]
    x[2, 0, [1, 44, 55, 98]] = 10
    x[2, 0, [4, 22, 99]] = 11
    x = np.array(x, dtype=np.float32).reshape((3, 1, 100))
    x = cuda.to_gpu(x)
    tmp = TemporalKMaxPooling(5)
    print(tmp.forward((x,)))
    print(tmp.indexes)


