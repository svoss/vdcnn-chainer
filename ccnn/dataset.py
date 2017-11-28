import numpy as np
from chainer.dataset import download
from chainer.datasets import TupleDataset
import os
import csv
import codecs
import json
from hashlib import md5


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


def get_yelp(type='full', loc=None):
    if loc is None:
        raise ValueError("Yelp dataset should be downloaded manually and location should be provided as yelp_location argument")

    h = md5(loc).hexdigest()

    root = download.get_dataset_directory('%s_%s' % (type, h))
    data_npz = os.path.join(root, 'data.npz')
    def _get_class(stars):
        if type == 'full':
            return stars - 1
        #polarity
        elif stars < 3:
            return 0
        elif stars > 3:
            return 1
        return -1

    def creator(path):
        x_test = []
        y_test = []
        x_train = []
        y_train = []

        n_train = dict([(n, 130000) for n in range(1,6)])
        n_test = dict([(n, 10000) for n in range(1,6)])
        with codecs.open(loc, encoding='utf8') as io:
            for l in io:
                data = json.loads(l)
                c = _get_class(data['stars'])
                if n_train[data['stars']] > 0:
                    if c >= 0:
                        x_train.append(data['text'])
                        y_train.append(c)
                    n_train[data['stars']] -= 1
                elif n_test[data['stars']] > 0:
                    n_test[data['stars']] -= 1
                    if c >= 0:
                        x_test.append(data['text'])
                        y_test.append(c)

        x_train = np.array(x_train, dtype=np.unicode)
        y_train = np.array(y_train, dtype=np.int32)
        x_test = np.array(x_test, dtype=np.unicode)
        y_test = np.array(y_test, dtype=np.int32)
        np.savez(path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

        return (x_train, y_train), (x_test, y_test)

    def loader(path):
        data = np.load(path)
        return (data['x_train'], data['y_train']), (data['x_test'], data['y_test'])

    train, test = download.cache_or_load_file(data_npz, creator, loader)

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


def get_dataset(name, yelp_loc=None):
    if name == 'ag-news':
        train, test = download_ag_news_dataset()
        return train, test, 4
    if name == 'yelp-polarity':
        train,test = get_yelp('polarity',yelp_loc)
        return train, test, 2
    if name == 'yelp-full':
        train, test = get_yelp('full', yelp_loc)
        return train, test, 5
    if name == 'artificial':
        train, test = get_artificial_dataset()
        return train, test, 2
    raise ValueError("Dataset %s not known, available options are: ag-news," % name)

def get_character_encoding_dataset(name, encoder, test_mode=False, yelp_loc=None):
    train, test, n_classes = get_dataset(name, yelp_loc=yelp_loc)
    if test_mode:
        train = train[0][:1000], train[1][:1000]
        test = test[0][:1000], test[1][:1000]
    from collections import Counter
    print Counter(train[1])
    if name == 'artificial':
        return TupleDataset(*train), TupleDataset(*test), n_classes
    return CharacterEncodingDataset(datasets=train, encoder=encoder), CharacterEncodingDataset(datasets=test, encoder=encoder), n_classes