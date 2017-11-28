# Chainer implementation of the VDCNN architecture 
This repository contains an implementation of the [Very Deep Convolutional Networks
for Text Classification](https://arxiv.org/pdf/1606.01781.pdf) paper in the [chainer](https://chainer.org/) library.

## Training the network
Both python 2.7 and python 3 are supported. The following code will install the dependencies and setup the config.ini file.
__Installation:__
```
pip install -r requirements.txt
cp ccnn/config.ini.dist ccnn/config.ini
```

__Training:__
```
python cnn/train.py
```

All arguments available can be found by running `train.py --help`. The default value for each argument will be loaded from config.ini (you only have to pass the arguments that you want to change).


## Very Deep Convolutional Networks for Text Classification
This paper implements the VDCNN architecture with the following comments:

- In between the different convolutional feature blocks only max-pooling layers with stride 2 and kernel size 3 are implemented. The authors also experiment with k max pooling in between and convolutional layers instead of max pooling but found that max pooling performs best overall. At the end of the network we still make use a [k-max-pooling](#user-content-dynamic-k-max-pooling-in-chainer) layer.
- At the moment using shortcuts inspired by the resnet architecture from computer vision is not implemented.

## Datasets
In the paper the datasets introduced in the author's [earlier paper](https://arxiv.org/pdf/1509.01626.pdf) are re-used, from which I implemented the following at the moment.

| Dataset | Classes | Training examples | Test examples |
| ---  | --- | --- | --- |
| AG news         | 4       |120,0000 |30,000     |
| Yelp-full*      | 5       |650,000 |50,000   |
| Yelp-polarity*  | 2       |520,000  |40,000    |

#### Ag-news dataset
The ag-news dataset will be downloaded automatically from [mhjabreel](https://github.com/mhjabreel/CharCNN) when you try to use it for the first time.

#### Yelp dataset
Yelp dataset has to be manually downloaded from the [yelp dataset challenge webpage](https://www.yelp.com/dataset/challenge) and the review.json file location has to provided by the user using the `yelp_location` variable.
 I couldn't find the exact dataset the papers dataset was based on as yelp seems to have extended their dataset since the publication. Also they seem to have selected a subset of the original dataset but don't describe how.
I picked the first 130,000 examples for each star rating as training set and the following 10,000 as test set. In case of polarity the 1,2 and 4,5 classes are merged.

## Dynamic k-max pooling in chainer
The VDCNN makes use of a dynamic k-max pooling layer to select the 8 most important features over the complete sentence,
while maintaining the order in which they were found. This layer is used just before the fully connected layers. The layer is not implemented in the chainer package so
I made my own implementation that can can be found [here](ccnn/temporal_k_max_pooling.py).

To give an indication of how well the layer performs computationally. On the ag-news dataset, a network with depth level 17 can be trained on a GTX-1070 gpu for 15 epochs in 1h 30m with the k max pooling layer. If we replace this layer with a max pooling layer of size 119 and stride 1 (this results in exactly the same output size), so all other calculation remain the same. The same training takes 1h 23m.

## Short cut connections
Short cut connections are not implemented yet.


