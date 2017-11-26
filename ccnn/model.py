import chainer
import chainer.links as L
import chainer.functions as F
from temporal_k_max_pooling import temporal_k_max_pooling

class VDCNN(chainer.Chain):
    """
    The VDCNN architecture is displayed in figure 1 of the paper
    It's start with an embedding with vector size 16, followed by one convolutional layer of 1x3 with 64 features
    The network basically

    The authors conclude: Max-pooling performs better than other pooling types. We only use max pooling to reduce the
    temporary dimensions in between
    """
    def __init__(self, depth=9, n_classes=2, shortcut=False, alphabet_size=69):
        """
        :param depth: total number of convolutional layers in network, should be in [9,17,29,49]
        :param n_classes: number of output classes
        :param shortcut: use shortcut functionality, where optionally the network can skip a convolutional operation, like resnet
        :param alphabet_size: number of characters as input, default for VDCNN is 69
        """
        self.n_classes = n_classes
        self.depth = depth
        self.shortcut = shortcut
        if self.shortcut:
            raise NotImplementedError("Shortcuts are not implemented yet")
        d64, d256, d512 = self._get_depths() # number of convolutional blocks per features

        super(VDCNN, self).__init__(
            # Lookup table, 16
            embed=L.EmbedID(alphabet_size, 16),
            # 3, TempConv, 64
            first=L.ConvolutionND(1,16, 64, 3, stride=1, pad=1),
            # [Convolutional Block, 3, 64]^d64
            c64=FeatureBlock(features=64, n=d64, shortcut=self.shortcut),
            # [Convolutional Block, 3, 128]^d128, 128 block always has same number of blocks as 64 block (see table 2)
            c128=FeatureBlock(features=128, n=d64, shortcut=self.shortcut),
            # [Convolutional Block, 3, 256]^d256
            c256=FeatureBlock(features=256, n=d256, shortcut=self.shortcut),
            # [Convolutional Block, 3, 512]^d512
            c512=FeatureBlock(features=512, n=d512, shortcut=self.shortcut),
            # fc(4096, 2048)
            fc0=L.Linear(4096, 2048),
            # fc(2048, 2048)
            fc1=L.Linear(2048, 2048),
            # fc(2048, self.n_classes)
            fc2=L.Linear(2048, self.n_classes)
        )

    def _get_depths(self):
        """
         Gets number of convolutional blocks per feature level as tuple (d64,d256,d512):
         where d64 represents the number of conv blocks for the 64 and 128 features layers,
         d256 represents the number of conv blocks of 256 layer,
         d512 represents the number of conv blocks of 512 layer.
         This is based on table 2 of the paper, please note that every convolutional block contains
         two convolutional layers
        :return:
        """
        if self.depth == 9:
            return 1, 1, 1
        elif self.depth == 17:
            return 2, 2, 2
        elif self.depth == 29:
            return 5, 2, 2
        elif self.depth == 49:
            return 8, 5, 3
        raise ValueError("Depth of VDCNN network should be 9, 17, 29 or 49, %d given" % self.depth)

    def __call__(self, x):

        h = self.embed(x)
        # chainer for some reason let embed layer put channel on last dimension while convND expects them in first dim after batch
        h = F.rollaxis(h, 2, 1)
        h = self.first(h)


        h, prev = self.c64(h)
        h = F.max_pooling_nd(h, 3, 2)
        #h = self._perform_shortcut(h, prev, 'sc128')

        h, prev = self.c128(h)
        h = F.max_pooling_nd(h, 3, 2)
        #h = self._perform_shortcut(h, prev, 'sc256')

        h, prev = self.c256(h)
        h = F.max_pooling_nd(h, 3, 2)
        #h = self._perform_shortcut(h, prev, 'sc512')

        h,_ = self.c512(h)

        # should be 8-max pooling

        h = temporal_k_max_pooling(h,8)
        #h = F.max_pooling_nd(h, 119,stride=1)

        h = F.relu(self.fc0(h))
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return h

    def _perform_shortcut(self, h, prev, down_scale_layer):
        """
        To make sure we can shortcut 'over' a pooling layer we apply a linear projection. This is not explicitly discussed in the
        'VDCNN' paper, but is what happens in the resnet paper. This function returns h if shortcut is set to false
        :param h:
        :param prev:
        :param down_scale_layer: linear projecti on ayer
        :return:
        """
        if self.shortcut:
            raise NotImplementedError("Shortcut is not implemented yet")

        return h


class FeatureBlock(chainer.ChainList):
    """
    This block combines multiple n ConvBlocks depending on the overall depth of the network
    Within this block the number of features is equal for every block and no pooling operation takes place
    When shortcut is used a shortcut is added between each conv block instance
    """
    def __init__(self, features, n=1, shortcut=False):

        super(FeatureBlock, self).__init__()
        self.features = features
        self.n = n
        self.shortcut = shortcut
        in_features = 64 if features == 64 else features/2 # input features of first layer
        for _ in xrange(self.n):
            self.add_link(ConvBlock(in_features, features))
            in_features = features

    def __call__(self, x):
        h = x
        # Shortcut should keep track of previous activation
        prev = None
        for l in iter(self):
            if self.shortcut:
                raise NotImplementedError("Shortcut is not implemented yet")
                #if prev is not None:
                #    h = F.relu(h + prev) # previous activation is added and RELU for shortcut
                #prev = h # keep current activation for next pass
            h = l(h)
        # dont forget to return previous activation for call
        return h, prev


class ConvBlock(chainer.Chain):
    """
    Basic convolutional building block of network architecture, as described in figure 2 of the paper
    Consists of two 1x3 convolutional layers with Relu and batch normalization in between
    """
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