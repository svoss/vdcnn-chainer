import chainer
import chainer.links as L
import chainer.functions as F


class VDCNN(chainer.Chain):
    def __init__(self, train, depth=9, n_classes=2, shortcut=False, alphabet_size=70):
        self.n_classes = n_classes
        self.depth = depth
        d64,d256,d512 = self._get_depths()

        super(VDCNN, self).__init__(
            embed=L.EmbedId(alphabet_size, 16),
            first=L.ConvolutionND(1,16, 64, 3, stride=1, pad=1),
            c64=CombinedBlock(features=64, n=d64),
            c128=CombinedBlock(features=128, n=d64),# 128 block always has same number of blocks as 64 block
            c256=CombinedBlock(features=256, n=d256),
            c512=CombinedBlock(features=512, n=d512),
            fc0=L.Linear(4096, 2048),
            fc1=L.Linear(2048, 2048),
            fc2=L.Linear(2048, self.n_classes)
        )

    def _get_depths(self):
        """
         Gets number of convolutional blocks per feature level
        as tuple (d64,d256,d512):
         where d64 represents the number of conv blocks for the 64 and 128 features layers
         d256 represents the number of conv blocks of 256 layer
         d512 represents the number of conv blocks of 512 layer
        :return:
        """
        if self.depth == 9:
            return 2,2,2
        elif self.depth == 17:
            return 4,4,4
        elif self.depth == 29:
            return 10,4,4
        elif self.depth == 49:
            return 16,10,6

class CombinedBlock(chainer.ChainList):
    def __init__(self, features, n=2, shortcut=False):
        self.features = features
        self.n = n
        self.shortcut = shortcut
        super(CombinedBlock, self).__init__()
        in_features = 64 if features == 64 else features/2 # input features of first layer
        for _ in xrange(self.n):
            self.add_link(ConvBlock(in_features, features))
            in_features = features

    def __call__(self, x):
        h = x
        for l in iter(self):
            h = l(h)
            print h.shape
            # Todo: add layer short cut support
        return h

class ConvBlock(chainer.Chain):
    # Basic convolutional building block of network architecture
    def __init__(self, input_features, features):
        """

        :param input_features:
        :param features:
        """
        self.features = features
        self.input_features = input_features
        super(ConvBlock, self).__init__(
            l0=L.ConvolutionND(1,self.input_features, self.features, 3, pad=1),
            b0=L.BatchNormalization(self.features),
            l1=L.ConvolutionND(1, self.features, self.features, 3, pad=1),
            b1=L.BatchNormalization(self.features)
        )

    def __call__(self, x):
        h = F.relu(self.b0(self.l0(x)))
        h = F.relu(self.b1(self.l1(h)))

        return h