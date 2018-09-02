# Neural Processes implemented with Chainer
Implemenation of [Neural Processes](https://arxiv.org/pdf/1807.01622) with [Chainer](https://chainer.org/).

This is the blog post (only in Japanese) about this repository. (now writing...)

The experiments and implementation are inspired by [blog post by Kaspar Märtens](https://kasparmartens.rbind.io/post/np/)

MIT license. Contributions welcome.

## Requirements
python 2.x, chainer 4.3.1, numpy, matplotlib, and [binarized mnist dataset](https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz).

## examples

### Train an NP on a single small data set.

![](fig/1d.gif)


### Train an NP on repeated draws from the GP.

![](fig/gp.gif)
