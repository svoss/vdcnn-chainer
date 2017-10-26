import argparse
import matplotlib
from model import ConvBlock, CombinedBlock
matplotlib.use('Agg')


def define_args():
    parser = argparse.ArgumentParser(description='Trains the very deep convolutional  ')
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID (negative value indicates CPU)")
    return parser.parse_args()


if __name__ == "__main__":
    args = define_args()
    model = CombinedBlock(128)
    import numpy as np
    x = np.random.rand(10,64,1024).astype(np.float32)
    y = model(x)