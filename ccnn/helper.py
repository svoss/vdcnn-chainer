from chainer import Link, Chain, ChainList, Parameter

def count_model_parameters(model):
    total = 0
    for x,y in model.namedparams():
        t = 1
        for s in y.shape:
            t *= s
        total += t
    return total