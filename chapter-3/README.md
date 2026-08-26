# MNIST robustness

MNIST is a collection of 70,000 greyscale images of hand-written digits, each 28
by 28 pixels, and is one of the most widely used benchmarks in machine learning.
A classifier takes one such image and scores each of the ten digits; the digit
with the highest score is the one it reports.

This chapter verifies the widely studied *adversarial robustness* problem: any
small perturbation to the input, for example adjusting a few pixels, should not
change the digit the network reports. The problem was brought to prominence by
[Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199), and
the variant used here is the *classification robustness* catalogued by
[Casadio et al.](https://doi.org/10.1007/978-3-031-13185-1_11). Although this
example is specialised to image classification, it should be relatively easy to
adapt to other domains.

The exercises additionally use Fashion MNIST, which has the same shape — 28 by
28 greyscale images in ten classes — but shows items of clothing rather than
digits.

## Where things are

- `chapter-code` - the specification, the network, the data, and the commands to
  run them. See its README.
- `exercises` - the exercises for this chapter, and what you need to solve them.
- `solutions` - sample solutions.
