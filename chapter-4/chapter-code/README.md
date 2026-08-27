# Training the Vanilla classifier and Checking its robustness

Let us first train a Vanilla classifier, similar to the one that was used to produce the onnx file in Chapter 3.


# Running the property-driven training example

The data sets are described in the [chapter README](../README.md) — they are the same
ones used in Chapters 2 and 3. This file covers the files in this folder, the libraries
needed to run them, and how to run them.

## Input files

- `fmnist-robustness.vcl` - the specification. It declares one network
  (`classifier : Image -> Tensor Real [10]`), one parameter to supply
  (`epsilon : Real`), one inferred parameter (`n : Nat`, the data set size), two data
  sets (`trainingImages`, `trainingLabels`) and one property (`robust`). Compiled to a
  loss function rather than to queries, the property becomes the constraint loss.

- `pt_classifier.py` - the PyTorch version: loads Fashion MNIST, trains a small
  multi-layer perceptron against task loss and constraint loss combined, and exports
  the result to ONNX.

- `tf_classifier.py` - the TensorFlow version of the same thing.

Neither script needs a network or data files as input: it downloads Fashion MNIST
itself and produces the network as output.

## Required libraries

Both scripts need Vehicle's Python bindings, which provide the differentiable-logic
compiler used by `vehicle_lang.loss`:

- `vehicle-lang` - tested with 0.27.1.

For `pt_classifier.py`:

- `torch` and `torchvision` - `torchvision` supplies the Fashion MNIST loader.
- `onnxscript` - **required for the ONNX export**, not just for training. Recent
  versions of PyTorch route `torch.onnx.export` through a new exporter that imports
  `onnxscript`; without it the script trains successfully and then fails on the very
  last step with `ModuleNotFoundError: No module named 'onnxscript'`.
- Note that `torch` and `numpy` must be ABI-compatible. A `torch` built against NumPy
  1.x cannot be used with NumPy 2.x: the symptom is
  `RuntimeError: Numpy is not available` raised from inside a `.numpy()` call. Tested
  with torch 2.13.0 and numpy 2.2.6.

For `tf_classifier.py`:

- `tensorflow` - supplies both the model API and the Fashion MNIST loader.
- `tf2onnx` - needed to convert the exported SavedModel to ONNX, as shown below.

For the verification step at the end:

- `maraboupy` - provides the `Marabou` executable. Tested with 2.0.0.

## Before you run: create the output directory

Both scripts write their trained network under `models/`, and neither creates it. Git
does not track empty directories, so it will not be present after a fresh clone and the
export step will fail with `FileNotFoundError`. Create it first:

```bash
mkdir -p models
```

## Training

```bash
python pt_classifier.py
```

The script prints the task loss, the constraint loss and their weighted total at every
step, which is the interesting part: the constraint loss is the specification being
optimised. With the default settings there are 16 steps per epoch and 5 epochs, ending
in the export:

```plain
Step: 0,   Loss (task | constraint | total): 2.3082 | 0.1831 | 1.2456
Step: 1,   Loss (task | constraint | total): 2.2493 | 0.1037 | 1.1765
Step: 2,   Loss (task | constraint | total): 2.2106 | 0.0886 | 1.1496
...
Epoch: 1, Total loss: 1.1496
...
Saved to models/simple_classifier.onnx
```

The numbers will differ on every run, because the model is initialised randomly and the
data loader shuffles. What to look for is the *middle* column falling: that is the
constraint loss, and a downward trend means the network is being pushed towards
satisfying the specification. The task loss in the first column should fall too, more
slowly. If the constraint loss stays flat at zero from the first step, the property is
already trivially satisfied and the run tells you nothing.

PyTorch also prints its own progress lines from the ONNX exporter
(`[torch.onnx] ... ✅`); those are informational.

The `SUBSET_SIZE` constant at the top limits training to the first 1024 images. Raise
it for better accuracy at the cost of a considerably longer run, keeping it an exact
multiple of `BATCH_SIZE`.

The TensorFlow script instead writes a SavedModel directory, which needs a further step
to reach ONNX:

```bash
python tf_classifier.py
python -m tf2onnx.convert \
    --saved-model models/tf_simple_classifier \
    --output models/tf_simple_classifier.onnx
```

## Verifying using Marabou

The exported network can be checked against the same specification it was trained on,
exactly as in Chapter 3. Using one image from the Chapter 3 exercise data:

```bash
vehicle verify \
  --specification fmnist-robustness.vcl \
  --network classifier:models/simple_classifier.onnx \
  --parameter epsilon:0.005 \
  --dataset trainingImages:../../chapter-3/exercises/FMNIST/idxdata/1Image.idx \
  --dataset trainingLabels:../../chapter-3/exercises/FMNIST/idxdata/1Label.idx \
  --solver Marabou
```

Do not expect this to succeed on a network trained with the default settings. A few
epochs over 1024 images is not enough for robustness at this epsilon, so the expected
result is a counterexample:

```plain
Verifying properties:
  robust!0 [=======================================================] 9/9 queries
    result: ✗ - Marabou found a counterexample
      perturbation: [ [ ... ] ]
```

Vehicle also warns here that the property uses a strict inequality (`<`), which the
Marabou query format does not support and which Vehicle therefore converts; see
[vehicle issue 74](https://github.com/vehicle-lang/vehicle/issues/74).

A `✗` is the honest outcome of this example, not a failure of the setup. The point of
the chapter is that constraint loss moves the network in the right direction, not that
a few minutes of training makes it provably robust. Comparing this outcome against a
network trained *without* the constraint loss is Exercise #2, and that comparison is
where the effect shows up.
