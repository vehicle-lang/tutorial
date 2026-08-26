# MNIST robustness

MNIST is a collection of 70,000 greyscale images of hand-written digits, each 28
by 28 pixels, and is one of the most widely used benchmarks in machine learning.
It was assembled by Yann LeCun, Léon Bottou, Yoshua Bengio and Patrick Haffner
from the NIST special databases, and introduced in 1998 in
[Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf).
It is split into 60,000 training and 10,000 test images. A classifier takes one
such image and scores each of the ten digits; the digit with the highest score is
the one it reports.

This chapter verifies the widely studied *adversarial robustness* problem: any
small perturbation to the input, for example adjusting a few pixels, should not
change the digit the network reports. The problem was brought to prominence by
[Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199), and
the variant used here is the *classification robustness* catalogued by
[Casadio et al.](https://doi.org/10.1007/978-3-031-13185-1_11). Although this
example is specialised to image classification, it should be relatively easy to
adapt to other domains.

## Fashion MNIST

The exercises additionally use Fashion MNIST, introduced in 2017 by Han Xiao,
Kashif Rasul and Roland Vollgraf at Zalando Research in
[Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms](https://arxiv.org/abs/1708.07747).
It comprises 70,000 greyscale images of fashion products, 60,000 for training
and 10,000 for testing, drawn from ten categories with 7,000 images each:
T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag and
ankle boot.

It was designed as a direct drop-in replacement for MNIST, sharing the same
image size, data format and train/test split, so a specification written for one
transfers to the other with no change beyond the network and the data files.
Classifying clothing is the harder task of the two, which makes it a more
demanding subject for robustness verification: a model that is accurate on
Fashion MNIST is typically easier to falsify inside a given epsilon-ball than
one trained on digits.

## Where things are

- `chapter-code` - the specification, the network, the data, and the commands to
  run them. See its README.
- `exercises` - the exercises for this chapter, and what you need to solve them.
- `solutions` - sample solutions.
