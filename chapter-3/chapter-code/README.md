# Running the MNIST robustness example

The data set and the verification problem are described in the
[chapter README](../README.md). This file covers the files in this folder and
how to run them.

## Input files

- `mnist-classifier.onnx` - the neural network being verified.

- `mnist-robustness.vcl` - the specification describing the desired behaviour.

- `t2-images.idx` - a data set of input images. Doubles between 0.0 and 1.0
  inclusive.

- `t2-labels.idx` - a data set of output labels. Integers between 0 and 9
  inclusive.

## Verifying using Marabou

A network can be verified against the specification by running the following
command:

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

## Notes

1. The classifier is obtained from
   [here](https://github.com/onnx/models/blob/main/validated/vision/classification/mnist/model/mnist-12.onnx).
   It is a convolutional network, so it can be verified with Marabou but not
   with vibecheck, whose intermediate representation does not yet cover
   convolutions.

2. The `.idx` files are obtained from the MNIST test set (see the
   [data set description](https://www.tensorflow.org/datasets/catalog/mnist));
   the original `yann.lecun.com` downloads are no longer available, so
   `../exercises/create_dataset.py` fetches them from the
   [ossci-datasets mirror](https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz).

3. In the data set available from the link above, pixels are stored as integers
   between 0 and 255. In the `idx` files in this folder, their values have been
   normalised to doubles between 0.0 and 1.0.

4. This specification is particularly expensive to verify (9 queries per image),
   and therefore the example data sets only contain 2 of the original 10,000
   test images. The specification works for the full data set without any
   further changes, although expect verification to take a long time.
   `../exercises/create_dataset.py` generates larger data sets.
