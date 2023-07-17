# MNIST robustness example

This file is a (semi-complete) example of a specification for the widely studied adversarial robustness problem. The specification states that any small small pertubation to the input, e.g. adjusting a few pixels, should not significantly change the output
of the network. Although the example we consider here is specialised to image classification with MNIST data set, it should be relatively easy to tweak it to work with other domains.

If you are familiar with proofs of robustness already, your job will be to complete the missing lines in the provided (incomplete) specification (file `mnist-robustness.vcl`). If you are not familiar with the subject, 
please study the chapter _"Proving Neural Network Robustness"_ here:  https://vehicle-lang.github.io/tutorial/

If you want to check your solution, you can find a complete Vehicle spec for this exercise in the "examples" section on the [tutorial page](https://github.com/vehicle-lang/tutorial), in the code accompanying Chapter 3. Robustness. However, we recommend that, as a challenge, you first try to solve this exercise using only the code available for the ACAS Xu spec.


This folder contains the following files:

- `mnist-classifier.onnx` - the neural network used to implement the controller.

- `mnist-robustness.vcl` - the specification describing the desired behaviour (with wholes left for you to complete).

- `t2-images.idx` - a dataset of input images. Doubles between 0.0 and 1.0 inclusive.

- `t2-labels.idx` - a dataset of output labels. Integers between 0 and 9 inclusive.

## Notes

1. The classifier is obtained from [here](https://github.com/onnx/models/blob/main/vision/classification/mnist/model/mnist-12.onnx).

2. The `.idx` files are obtained from [here](http://yann.lecun.com/exdb/mnist/).

3. Note that in the dataset available from the link above, pixels are stored as integers between 0 and 255. In the `idx` files in this folder, their values have been normalised to doubles between 0.0 and 1.0.

4. This specification is particularly expensive to verify (9 queries per image), and therefore the example datasets only contain 2 of the original 10000 test images.
The specification should work for the full dataset without any further changes, although expect verification to take a long time.

## Verifying using Marabou

The outputs of the above Vehicle commands can be found in the test suite:

A network can be verified against the specification by running the following command:

```bash
vehicle verify \
  --specification examples/mnist-robustness/mnist-robustness.vcl \
  --network classifier:examples/mnist-robustness/mnist-classifier.onnx \
  --parameter epsilon:0.005 \
  --dataset trainingImages:examples/mnist-robustness/t2-images.idx \
  --dataset trainingLabels:examples/mnist-robustness/t2-labels.idx \
  --verifier Marabou
```

Note that the epsilon value can be changed, but the memory requirements of
Marabou may increase drastically as epsilon increases.
