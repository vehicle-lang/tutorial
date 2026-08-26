# MNIST robustness example

MNIST is a collection of 70,000 greyscale images of hand-written digits, each 28
by 28 pixels, and is one of the most widely used benchmarks in machine learning.
This example is a specification for the widely studied adversarial robustness
problem: any small perturbation to the input, for example adjusting a few
pixels, should not change the digit the network reports.
The property was brought to prominence by the paper
[Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199), and
the variant used here is the *classification robustness* catalogued by
[Casadio et al.](https://doi.org/10.1007/978-3-031-13185-1_11).
Although this example is specialised to image classification, it should be
relatively easy to adapt to other domains.

## Input files

These files are in the folder `chapter-code`:

- `mnist-robustness.vcl` - the specification describing the desired behaviour.

- `mnist-classifier.onnx` - the network being verified. It is the small
  convolutional model from the
  [ONNX model zoo](https://github.com/onnx/models/blob/main/validated/vision/classification/mnist/model/mnist-12.onnx).

- `t2-images.idx` - a data set of input images. Doubles between 0.0 and 1.0
  inclusive.

- `t2-labels.idx` - a data set of output labels. Integers between 0 and 9
  inclusive.

Because verifying this property is expensive, nine queries per image, the
example data sets contain only 2 of the original 10,000 test images. The
specification works for larger data sets without any change, but expect
verification to take considerably longer. `exercises/create_dataset.py`
generates larger data sets.

## Verifying using Marabou

The following command verifies the `robust` property for the network
`mnist-classifier.onnx` at an epsilon of 0.005:

```bash
vehicle verify \
  --specification mnist-robustness.vcl \
  --network classifier:mnist-classifier.onnx \
  --parameter epsilon:0.005 \
  --dataset trainingImages:t2-images.idx \
  --dataset trainingLabels:t2-labels.idx \
  --solver Marabou
```

Note that the epsilon value can be changed, but the memory requirements of
Marabou may increase drastically as epsilon increases.

Note also that the classifier is a convolutional network. Each solver supports
a different range of layers, so this model can be verified with Marabou but not
with vibecheck, whose intermediate representation does not yet cover
convolutions. The Fashion MNIST model used in the exercises is a plain
multi-layer perceptron, and works with either.
