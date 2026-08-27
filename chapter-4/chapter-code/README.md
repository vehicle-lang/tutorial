# Training the Vanilla classifier and Checking its robustness

Let us first train a Vanilla classifier, similar to the one that was used to produce the onnx file in Chapter 3. For verification we will use exactly the same set up (same Vehicle command) as was used in the verification experiment of Exercise 7, Chapter 3.

## Worked experiment: the vanilla baseline, trained long and verified

We will start with `vanilla_classifier.py` trained in the FMNIST data set using
cross-entropy loss.

What follows is a complete record of one such run, so that you can reproduce it
step by step.

### Step 1: check the prerequisites

Training needs `torch` and `torchvision`; the ONNX export additionally needs
`onnxscript`; verification needs `maraboupy`. All are listed in the repository's
`requirements.txt`. The Fashion MNIST data is already committed under `data/`,
so nothing is downloaded.

```bash
pip install -r ../../requirements.txt
```

### Step 2: run the script as it ships

```bash
python vanilla_classifier.py
```

With its default `num_epochs = 5` this prints a loss for each of the 16 steps per
epoch and writes `onnx_models/vanilla_classifier.onnx`. That is enough to see the
mechanics, but five epochs is far too few to say anything about robustness.

### Step 3: train for longer, saving a checkpoint along the way

To watch robustness change as training proceeds, raise the epoch count and export
the network at several points rather than only at the end. Two edits are needed.
Replace

```python
num_epochs = 5
```

with

```python
num_epochs = 300
CHECKPOINTS = [100, 200, 300]
```

and, at the end of the epoch loop — that is, inside `for epoch in ...` but
outside the inner `for step, ...` loop — add

```python
    if (epoch + 1) in CHECKPOINTS:
        model.eval()
        torch.onnx.export(
            model,
            torch.randn(1, 1, 28, 28),
            f"onnx_models/vanilla_e{epoch + 1}.onnx",
            input_names=["input"],
            output_names=["output"],
            external_data=False,
        )
        model.train()
```

Then run it again:

```bash
python vanilla_classifier.py
```

The script reports once per epoch, giving the mean loss over the epoch and the
training accuracy, so 300 epochs is 300 lines rather than thousands.

#### What the training looked like

Timings are from a CPU-only machine; an epoch took between 0.8 and 1.8 seconds,
so all 300 epochs finished in under three minutes.

```plain
Epoch: 1, mean loss: 1.8607, train accuracy: 37.4%
...
Epoch: 25, mean loss: 0.1290, train accuracy: 96.7%
...
Epoch: 50, mean loss: 0.0217, train accuracy: 99.9%
...
Epoch: 75, mean loss: 0.0051, train accuracy: 100.0%
...
Epoch: 100, mean loss: 0.0020, train accuracy: 100.0%
```

The subset is only 1,024 images, so the network memorises it completely by about
epoch 75. Past that point accuracy cannot improve; the loss keeps falling because
the network grows more *confident* about answers it already gets right. Whether
that helps or harms provable robustness is exactly what the verification below
measures.

### Step 4: verify each checkpoint

Each exported network is checked with the command from Chapter 3's Exercise #7 —
the same specification, epsilon and data, with only the network swapped:

```bash
cd ../../chapter-3/exercises

vehicle verify \
  --specification FMNIST/fashionRobustness-solution.vcl \
  --network classifier:../../chapter-4/chapter-code/onnx_models/vanilla_e100.onnx \
  --parameter epsilon:0.005 \
  --dataset trainingImages:FMNIST/idxdata/0-49Images.idx \
  --dataset trainingLabels:FMNIST/idxdata/0-49Labels.idx \
  --solver Marabou
```

Repeat with `vanilla_e200.onnx` and `vanilla_e300.onnx`. Each run checks 50
images at nine queries apiece and takes several minutes.

#### Verifying using Marabou

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


