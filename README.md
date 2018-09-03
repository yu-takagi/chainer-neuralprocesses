# Neural Processes implemented with Chainer
Implemenation of [Neural Processes](https://arxiv.org/pdf/1807.01622) (NPs) introduced by Garnelo et al. (DeepMind) with [Chainer](https://chainer.org/). Neural Processes is a class of neural latent variable models, which can be considered as a combination of Gaussian Process (GP) and neural network (NN). Like GPs, NPs define distributions over functions, are capable of rapid adaptation to new observations, and can estimate the uncertainty in their predictions. Like NNs, NPs are computationally efficient during training and evaluation but also learn to adapt their priors to data.

This is the blog post (only in Japanese) about this repository. (now writing...)

The experiments and implementation are inspired by [blog post by Kaspar Märtens](https://kasparmartens.rbind.io/post/np/)

MIT license. Contributions welcome.

## Requirements
python 2.x, chainer 4.3.1, numpy, matplotlib, and [binarized mnist dataset](https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz).

## examples

### Train an NP on a single small data set.

![1d](./fig/1d.png)


### Train an NP on repeated draws from the GP.

![gp](./fig/gp.png)