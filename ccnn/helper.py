from chainer import Link, Chain, ChainList, Parameter
from operator import mul

def count_model_parameters(model):
    total = 0
    for x,y in model.namedparams():
        total += reduce(mul, y.shape)
    return total