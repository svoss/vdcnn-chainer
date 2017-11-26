# Chainer implementation of the VDCNN architecture 
This repository contains an implementation of the [Very Deep Convolutional Networks
for Text Classification](https://arxiv.org/pdf/1606.01781.pdf) paper in the [chainer](https://chainer.org/) library.

**This is work in progress**

## Very Deep Convolutional Networks for Text Classification
This paper implements the VDCNN architecture it provides an

## Training the network

## Datasets
In the paper the datasets introduced in [this paper](https://arxiv.org/pdf/1509.01626.pdf) are re-used, from which the following are supported at the moment:
| Dataset         | Classes |Training |Test    | Description |
| :-------------- | :-----: |:-------:|:-----: | :---------- |
| AG news         | 2       |2        |2       | I made use of [mhjarbreel](https://github.com/mhjabreel/CharCNN/) for providing the pre-processed csv|
| Yelp-full*      | 5       |1,150,00 |100,000 |             |
| Yelp-polarity*  | 2       |920,000  |80,000  |             |

#### Yelp dataset
Yelp dataset has to be manually downloaded from the yelp dataset challenge webpage and the review.json file location has to provided by the user using the `yelp_location` variable.

* I couldn't find the exact dataset they based their dataset on as yelp seems to have extended their dataset since the publication. Also they seem to have selected a subset of the original dataset but don't describe how.
I picked the first 230,000 examples for each star rating as training set and the 20,000 as test set. In case of polarity the 1,2 and 4,5 classes are merged.

## Dynamic k-max pooling in chainer
The VDCNN makes use of a dynamic k-max pooling layer to select the 8 most important features over the complete sentence,
while maintaining the order in which they were found. This layer is used just before the fully connected layers. The layer is not implemented in the chainer package so
I made my own implementation that can can be found in [ccnn/temporal_k_max_pooling.py].

To  depth 17, 1.23 vs 1.30.

## Short cut connections


