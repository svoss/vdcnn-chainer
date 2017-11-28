import argparse
import matplotlib
from config import get_config
from model import VDCNN
from dataset import get_character_encoding_dataset, AlphabetEncoder
matplotlib.use('Agg')
import numpy as np
import chainer.optimizers as O
import chainer.training.extensions as E
import chainer.iterators as I
import chainer.training as T
import chainer.links as L
import os
import sys
from helper import count_model_parameters
import six
def define_args():
    config = get_config()
    # defaults are loaded from config.ini file
    parser = argparse.ArgumentParser(description='Trains the very deep convolutional, all parameters can be entered ')
    parser.add_argument('--gpu', type=int, default=config.get('training', 'gpu'), help="GPU ID (negative value indicates CPU)")
    parser.add_argument('--depth', type=int, default=config.get('model', 'depth'), help="Depth of the network in terms of convolutional layers either 9, 17, 29 or 49")
    #parser.add_argument('--shortcut', type=int, default=config.get('model', 'shortcut'), help="Use resnet like shortcuts, int below 1 indicates no shortcut at least 1 indicates using shortcut ")
    parser.add_argument('--alphabet', type=type(six.u('')), default=config.get('model', 'alphabet'), help="Alphabet of characters to use")
    parser.add_argument('--test', type=int, default=config.get('training','test'), help="Set to 1 for test modus, training and test set to 1000 examples for testing purposes")
    parser.add_argument('--fixed_size', type=int, default=config.get('model', 'fixed_size'), help="Fixed size to pad sentences to, input larger then this text will be truncated")
    parser.add_argument('--dataset', type=str, default=config.get('training', 'dataset'), help="The dataset to train on. Choose from ag-news, yelp-full, yelp-polarity ")
    parser.add_argument('--lr', type=float, default=config.get('training', 'lr'), help="Learning rate")
    parser.add_argument('--batch-size', type=int, default=config.get('training', 'batch_size'), help="Batch size")
    parser.add_argument('--momentum', type=float, default=config.get('training', 'momentum'), help="Momentum")
    parser.add_argument('--epochs', type=int, default=config.get('training', 'epochs'), help="Number of epochs to perform training for")
    parser.add_argument('--prefix', type=str, default=config.get('output', 'prefix'), help="Prefix to prefix output folder with")
    parser.add_argument('--out', type=str, default=config.get('output', 'out'), help="Output folder, concatenated with the prefix as output folder")
    parser.add_argument('--yelp-location', type=str, default=config.get('dataset', 'yelp_location'), help="Local filesystem location of yelp dataset review json, obtained from https://www.yelp.com/dataset/challenge. Required if you are training on a yelp dataset")
    return parser.parse_args()


def train_model(args, model, train, test):
    train_iter = I.SerialIterator(train, args.batch_size)
    val_iter = I.SerialIterator(test, args.batch_size, shuffle=False, repeat=False)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    optimizer = O.MomentumSGD(lr=args.lr, momentum=args.momentum)
    optimizer.setup(model)
    eval_model = model.copy()

    updater = T.StandardUpdater(train_iter, optimizer, device=args.gpu)
    out = os.path.join(args.prefix, args.out)

    trainer = T.Trainer(updater, (args.epochs, 'epoch'), out)
    log_interval = (100 if args.test else 100), 'iteration'
    # Validate on test set after every epoch
    trainer.extend(E.Evaluator(val_iter, eval_model, device=args.gpu))

    #dump graph for loss and acc
    loss_r = E.PlotReport(['validation/main/loss', 'main/loss'], 'epoch', file_name="loss.png")
    acc_r = E.PlotReport(['validation/main/accuracy', 'main/accuracy'], 'epoch', file_name="accuracy.png")
    trainer.extend(loss_r)
    trainer.extend(acc_r)
    trainer.extend(E.PrintReport(['main/loss','main/accuracy', 'validation/main/loss', 'validation/main/accuracy']), trigger=log_interval)
    trainer.extend(E.LogReport(trigger=log_interval))
    trainer.extend(T.extensions.ProgressBar(update_interval=10))
    trainer.run()

def main(args):
    encoder = AlphabetEncoder(args.alphabet, args.fixed_size)
    train, test, n_classes = get_character_encoding_dataset(args.dataset, encoder, yelp_loc=args.yelp_location, test_mode=(args.test > 0))
    print("Training set has a size of {}".format(len(train)))
    print("Test set has a size of {}".format(len(test)))
    print("Number of classes: {}".format(n_classes))
    model = VDCNN(depth=args.depth, n_classes=n_classes)
    model = L.Classifier(model)
    print("Total number of model parameters: {}M".format(count_model_parameters(model) / float(10 ** 6)))
    train_model(args, model, train, test)


if __name__ == "__main__":
    args = define_args()
    main(args)