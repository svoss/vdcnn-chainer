import numpy as np
from chainer.dataset import download
from chainer.datasets import TupleDataset
import os
import csv
import codecs


class AlphabetEncoder(object):
    """
    Encodes alphabet to int index, padding with zeros to fixed size
    """
    def __init__(self, alphabet, fixed_size):
        """

        :param alphabet: string of all characters in alphabet, space will be put in fron
        :param fixed_size: fixed size to pad to
        """
        self.fixed_size = fixed_size
        alphabet = " " + alphabet
        self.alphabet = dict([(c,i) for i,c in enumerate(alphabet)])

    def encode_sentence(self, x):
        x = x.lower()
        return np.array([self.alphabet[x[i]] if len(x) > i and x[i] in self.alphabet else 0 for i in xrange(self.fixed_size)], dtype=np.int32)

    def encode_sentences(self, X):
        return np.array([self._encode_sentence(x) for x in X], dtype=np.int32)


class CharacterEncodingDataset(TupleDataset):
    def __init__(self, encoder, datasets):
        super(CharacterEncodingDataset, self).__init__(*datasets)
        self.encoder = encoder

    def __getitem__(self, index):
        items = super(CharacterEncodingDataset, self).__getitem__(index)
        if isinstance(index, slice):
            return [(self.encoder.encode_sentence(x), y) for x, y in items]
        else:
            X,Y = items
            return self.encoder.encode_sentence(X), Y


def download_ag_news_dataset():
    """
    Makes sure dataset of ag news is downloaded
    :return:
    """
    TRAIN_URL = "https://github.com/mhjabreel/CharCNN/raw/master/data/ag_news_csv/train.csv"
    TEST_URL = "https://github.com/mhjabreel/CharCNN/raw/master/data/ag_news_csv/test.csv"


    def creator(url, path):
        cached = download.cached_download(url)
        X = []
        Y = []
        with codecs.open(cached,'rb', encoding='utf8') as io:
            reader = csv.reader(io)

            for l in reader:
                x = l[1] + " " + l[2]
                x = x.replace("\\", " ")
                X.append(x)
                Y.append(int(l[0])-1)

        X = np.array(X, dtype=np.unicode)
        Y = np.array(Y, dtype=np.int32)
        np.savez(path, X=X,Y=Y)
        return X,Y

    def loader(path):
        data = np.load(path)
        return data['X'], data['Y']

    root = download.get_dataset_directory('VDCCN/ag-news')
    train_npz, test_npz = os.path.join(root, 'train.npz'), os.path.join(root, 'test.npz')
    train = download.cache_or_load_file(train_npz, lambda path: creator(TRAIN_URL, path), loader)
    test = download.cache_or_load_file(test_npz, lambda path: creator(TEST_URL, path), loader)
    return train, test


def get_artificial_dataset(n=10000):
    """
    Very simple artificial dataset that can be used to see if no bugs in your code
    :return:
    """
    Y = np.random.randint(0, 2, n).astype(np.int32)
    X = np.ones((1014, n), dtype=np.int32)
    X = X * Y
    X = np.rollaxis(X, 1, 0).astype(np.int32)
    m = int(n * .9)
    return (X[:m, :], Y[:m]), (X[m:, :], Y[m:])


def get_dataset(name):
    if name == 'ag-news':
        train, test = download_ag_news_dataset()
        return train,test,4
        #return val, test
    if name == 'artificial':
        train, test = get_artificial_dataset()
        return train, test, 2
    raise ValueError("Dataset %s not known, available options are: ag-news," % name)

def get_character_encoding_dataset(name, encoder, test_mode=False):
    train, test, n_classes = get_dataset(name)
    if test_mode:
        train = train[0][:1000], train[1][:1000]
        test = test[0][:1000], test[1][:1000]
    from collections import Counter
    print Counter(train[1])
    if name == 'artificial':
        return TupleDataset(*train), TupleDataset(*test), n_classes
    return CharacterEncodingDataset(datasets=train, encoder=encoder), CharacterEncodingDataset(datasets=test, encoder=encoder), n_classes