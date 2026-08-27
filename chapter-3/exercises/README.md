To complete the exercises, you will need:

- Exercise #1 (⭑): Run the Chapter code.
- 
All code is available from the chapter-code folder.

- Exercise #2 (⭑) : Experimenting with ε-balls of different size

Same

- Exercise #3 (⭑⭑) : Getting a statistical evaluation of robustness with respect to the given data set, for various epsilons

The only thing you need here is the file `create_dataset.py`, in order to generate larger `idx` files

- Exercise #4 (⭑) : Strong Classification Robustness in Vehicle
- 
  All files except for the spec with `vcl` extension remain the same. 

- Exercise #5 (⭑⭑): Explore Other Definitions of Robustness
- 
  All files except for the spec with `vcl` extension remain the same. 

- Exercise #6 (⭑⭑): Other Distances in Vehicle
- 
  All files except for the spec with `vcl` extension remain the same. 
  
- Exercise #7 (⭑⭑⭑): Conduct a complete “training - verification” experiment from start to finish

  For this one, we recommend you to use the data set Fashion MNIST (or FMNIST). You can use your own Python script to prepare `ONNX` and `IDX` files. However, we also provide these pre-cooked in this repository

## Additional comments

### Generating larger data sets (Exercise #3)

The two images shipped in `chapter-code` are not enough for a statistical
evaluation. `create_dataset.py` downloads the MNIST test set and writes IDX
files at whatever sizes you ask for:

```bash
python create_dataset.py 2 10 100
```

This writes `t2-images.idx` / `t2-labels.idx`, `t10-...`, `t100-...` and so on
into the current directory. It needs `numpy` and `idx2numpy`, both listed in
`requirements.txt`, and writes images as float64 and labels as uint8 to match
the files used in the chapter.

Be aware that Marabou can run out of memory on individual images as the data
set or epsilon grows. When that happens Vehicle reports those images as
`errored` rather than verified or falsified, and prints a reproducer path. This
is expected, not a mistake on your part, so record the counts as they come.

### Which solver can handle which definition (Exercises #4 - #6)

Marabou works with linear real arithmetic and admits only one occurrence of a
neural network per specification. So two kinds of definition fall outside it:
those comparing two network outputs, such as standard robustness
`|f(x) - f(y)| <= delta`, and those using a Euclidean rather than an
L-infinity ball, which makes the constraint quadratic.

Note also that the MNIST classifier is a convolutional network, so it can be
verified with Marabou but not with vibecheck. The Fashion MNIST model below is
a plain multi-layer perceptron and works with either.

### The Fashion MNIST material (Exercise #7)

The `FMNIST` folder contains:

- `fashion1l32n.onnx` - a small multi-layer perceptron, 784 -> 32 -> 10.

- `fashionRobustness-solution.vcl` - a model solution, if you would rather study
  one than write your own.

- `idxdata/` - images and labels in ten disjoint fifty-image slices, `0-49`,
  `50-99`, `100-149` and so on up to `450-499`, covering 500 images in total.
  There is also a `0-99` slice that overlaps the first two, a single-image pair
  `1Image`/`1Label`, and `individuals/` holding 1,000 single-image files. All of
  these are `float64` images of shape `[n, 28, 28]` with `uint8` labels, matching
  the format the specification expects.

  Note that `redDimImages.idx` is **not** usable with this specification: it
  holds three 2-by-2 images, not 28-by-28 ones, and its labels are `int32`
  rather than `uint8`.

The command below verifies one slice — the first fifty images. There is no
single 500-image file, so evaluating the whole set means running the command
once per slice and adding up the results, in the manner of Exercise #3. Verifying
fifty images takes several minutes, so budget accordingly before reaching for all
ten.

For example, to verify the first fifty images:

```bash
vehicle verify \
  --specification FMNIST/fashionRobustness-solution.vcl \
  --network classifier:FMNIST/fashion1l32n.onnx \
  --parameter epsilon:0.005 \
  --dataset trainingImages:FMNIST/idxdata/0-49Images.idx \
  --dataset trainingLabels:FMNIST/idxdata/0-49Labels.idx \
  --solver Marabou
```

Of these fifty images, forty are proved robust at this epsilon and ten are
falsified. Vehicle reports each image separately. Below is one of each: a proved
image and a falsified one, with its perturbation abbreviated.

```plain
Verifying properties:
  robust!0 [=======================================================] 9/9 queries
    result: 🗸 - Marabou proved no counterexample exists
  ...
  robust!11 [======================================================] 9/9 queries
    result: ✗ - Marabou found a counterexample
      perturbation: [ [ 0.0, 0.0, 0.0, ..., -5.0e-3, -5.0e-3, ... ], ... ]
  ...
robust:
    verified:  40/50
    falsified: 10/50
    timed-out: 0/50
    errored:   0/50
```

The perturbation is a full 28 by 28 array of the pixel changes Marabou found,
so a single counterexample runs to several thousand characters. The complete
transcript, with all fifty results and every perturbation in full, is in
[FMNIST/expected-output-0-49.txt](FMNIST/expected-output-0-49.txt).

Sample solutions for the other exercises are in `chapter-3/solutions`, but we
recommend that you try solving them first.
