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
epoch and writes `vanilla-experiment/onnx_models/vanilla_classifier.onnx`. That is enough to see the
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
num_epochs = 150
CHECKPOINTS = [75, 100, 150]
```

and, at the end of the epoch loop — that is, inside `for epoch in ...` but
outside the inner `for step, ...` loop — add

```python
    if (epoch + 1) in CHECKPOINTS:
        model.eval()
        torch.onnx.export(
            model,
            torch.randn(1, 1, 28, 28),
            f"vanilla-experiment/onnx_models/vanilla_e{epoch + 1}.onnx",
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
training accuracy, so 150 epochs is 150 lines rather than thousands.

#### What the training looked like

Timings are from a CPU-only machine; an epoch took between 0.3 and 2.5 seconds,
so all 150 epochs finished in about two and a half minutes.

```plain
Epoch: 1, mean loss: 2.1178, train accuracy: 32.9%
...
Epoch: 25, mean loss: 0.3325, train accuracy: 89.5%
...
Epoch: 50, mean loss: 0.1744, train accuracy: 95.1%
...
Epoch: 75, mean loss: 0.0785, train accuracy: 98.1%
...
Epoch: 100, mean loss: 0.0413, train accuracy: 99.5%
...
Epoch: 150, mean loss: 0.0103, train accuracy: 99.9%
```

The subset is only 1,024 images, so the network very nearly memorises it: it passes
98% by epoch 75 and touches 100% intermittently from about epoch 135, though it does
not stay there — the shuffled batches move it between 99% and 100% from one epoch to
the next. Your own numbers will differ in the same way, since the script fixes no
random seed.

Past that point there is almost no accuracy left to gain, yet the loss keeps falling
— by a factor of seven and a half between epochs 75 and 150 — because the network
grows more *confident* about answers it already gets right. Whether that helps or
harms provable robustness is exactly what the verification below measures.

### Step 4: verify each checkpoint

Each exported network is checked with the command from Chapter 3's Exercise #7 —
the same specification, epsilon and data, with only the network swapped:

```bash
cd ../../chapter-3/exercises

vehicle verify \
  --specification FMNIST/fashionRobustness-solution.vcl \
  --network classifier:../../chapter-4/chapter-code/vanilla-experiment/onnx_models/vanilla_e100.onnx \
  --parameter epsilon:0.005 \
  --dataset trainingImages:FMNIST/idxdata/0-49Images.idx \
  --dataset trainingLabels:FMNIST/idxdata/0-49Labels.idx \
  --solver Marabou
```

Repeat with `vanilla_e200.onnx` and `vanilla_e300.onnx`. Each run checks 50
images at nine queries apiece and takes several minutes — around sixteen on the
machine used here.

Vehicle reports each image separately, so a single image gives a single verdict —
either a proof or a counterexample:

```plain
Verifying properties:
  robust!0 [=======================================================] 9/9 queries
    result: ✗ - Marabou found a counterexample
      perturbation: [ [ 0.0, 0.0, 0.0, ..., -5.0e-3, -5.0e-3, ... ], ... ]
```

Vehicle also warns here that the property uses a strict inequality (`<`), which the
Marabou query format does not support and which Vehicle therefore converts; see
[vehicle issue 74](https://github.com/vehicle-lang/vehicle/issues/74).

A `✗` is not a failure of the setup: this network was trained only to classify, with
nothing asking it to be robust, so counterexamples are expected. The fifty-image runs
below quantify how often they occur.

#### Results

Each checkpoint was trained exactly as described above, on raw `[0,1]` pixels. The
complete Vehicle output for every run is kept in `vanilla-experiment/marabou-outputs/`,
since a single
counterexample prints its perturbation as a full 28 by 28 array and fifty images
produce far too much text to read inline.

### Training statistics

| epoch | mean loss | train accuracy |
| ----: | --------: | -------------: |
| 75 | 0.0785 | 98.1% |
| 100 | 0.0413 | 99.5% |
| 150 | 0.0103 | 99.9% |

Both figures are measured on the 1024 images the network trains on. Between epoch 75 and
epoch 150 the mean loss falls by a factor of seven and a half while training accuracy
climbs from 98.1% to 99.9%.

Note that the fifty images the specification is checked against are the first fifty
FashionMNIST **test** images, held out of those 1024. Everything in the verification
table below is therefore held-out behaviour, and not comparable with the training
accuracy above.

**These figures come from a re-run of the experiment.** An earlier version of these
scripts normalised its inputs while the verifier was given raw pixels, so the networks
were measured outside the input space they had been trained on. The scripts no longer
normalise. See [normalisation-options.md](normalisation-options.md) for the full account
and [folded-vanilla-models/](vanilla-experiment/folded-vanilla-models/) for the repaired versions of the
older checkpoints.

### Verification

Each checkpoint is verified with Chapter 3 Exercise #7's command, unchanged. Full output
is in `vanilla-experiment/marabou-outputs/`.

| epoch | correctly classified | robust, `epsilon 0.005` | robust, `epsilon 0.02` |
| ----: | -------------------: | ----------------------: | ---------------------: |
| 75 | 38/50 | **37/50** | **25/50** |
| 100 | 38/50 | **37/50** | _pending_ |
| 150 | 37/50 | **36/50** | **21/50** |

Solver times were 1033 s, 1029 s and 980 s at `epsilon 0.005`, and 924 s for the
150-epoch checkpoint at `epsilon 0.02`. No image timed out or errored in any run, so
every count is decisive rather than an artefact of Marabou giving up.

The `correctly classified` column is what makes the other two readable. An image the
network already misclassifies cannot be robust — `advises perturbedImage label` fails at
zero perturbation — so it is counted as a failure for reasons that have nothing to do with
robustness. Read each robustness figure against that column rather than against 50.

At `epsilon 0.005` the network sits exactly one image below it at all three checkpoints:
37 of 38, 37 of 38, 36 of 37. Epochs 75 and 100 agree on both columns, and their solver
times are within four seconds of each other, even though the mean loss almost halves
between them. Epoch 150's drop from 37 to 36 is not a loss of robustness either — its
`correctly classified` figure fell from 38 to 37, so one image left that column and took
its robustness result with it. Across all three checkpoints exactly one correctly
classified image fails to be proved robust.

At `epsilon 0.02` the 150-epoch checkpoint manages 21 of the 37 it classifies correctly,
so sixteen images that survive the smaller radius fail at the larger one.

The `epsilon 0.005` column is the chapter's claim in its sharpest form. Further
optimisation of cross-entropy made the network more confident about answers it already
had, and changed nothing the verifier could see.

#### A second experiment: how much does epsilon decide?

Everything above is measured on the fifty images of Chapter 3's set, thirteen of which
this network misclassifies. To ask about robustness alone, the experiment below drops
those thirteen and keeps only the 37 images the epoch-150 checkpoint classifies correctly
— the set in
[vanilla-experiment/accurate-test-E150/](vanilla-experiment/accurate-test-E150/). Nothing
in it can fail at zero perturbation, so every falsification is a robustness failure.

The same network and the same specification, changing nothing but the radius:

| `epsilon` | images | provably robust | genuinely non-robust | robust share | solver |
| --------: | -----: | --------------: | -------------------: | -----------: | -----: |
| 0.005 | 37 | 36/37 | 1 | 97.3% | 980 s |
| 0.02 | 37 | 21/37 | 16 | 56.8% | 699 s |

```bash
vehicle verify \
  --specification ../../chapter-3/exercises/FMNIST/fashionRobustness-solution.vcl \
  --network classifier:vanilla-experiment/onnx_models/vanilla_e150.onnx \
  --parameter epsilon:0.02 \
  --dataset trainingImages:vanilla-experiment/accurate-test-E150/accurateImages.idx \
  --dataset trainingLabels:vanilla-experiment/accurate-test-E150/accurateLabels.idx \
  --solver Marabou
```

**The choice of epsilon, not the network, decided the earlier result.** A network trained
on cross-entropy alone, with nothing whatsoever asking it to be robust, is provably robust
around 97.3% of the images it classifies correctly at `epsilon 0.005`. Quadruple the
radius and that falls to 56.8%. The same weights, the same images, the same specification
— only the size of the neighbourhood differs.

Two practical consequences.

**`epsilon 0.005` cannot support a comparison between training methods.** It leaves one
image of headroom out of 37. No method, however good, could demonstrate anything against
that baseline, because there is almost nothing left to win. `epsilon 0.02` leaves sixteen,
which is a deficit large enough to measure an improvement against.

**Falsification is cheaper than proof.** The wider ball took 699 s against 980 s, despite
having 13 fewer images to consider. Marabou stops as soon as it finds a counterexample,
whereas proving robustness means exhausting the search space. A verification run that
finishes surprisingly quickly is often reporting bad news.

Two further cautions on the numbers above:

- **Fifty images is a coarse instrument.** Across epochs 75 to 150 the held-out count
  wanders between 37 and 40 (standard deviation 0.93) on a strictly falling loss. A
  difference of one to three images between checkpoints is within noise and should not be
  read as a trend either way.
- **Chapter 3's `fashion1l32n.onnx` is not a like-for-like comparison.** It scores 40/50
  on the same command, but classifies 46 of the 50 correctly against 38 here, having been
  trained on far more data. Per eligible image it is the *less* robust of the two, 40/46
  (87.0%) against 37/38 (97.4%). Comparing raw verified counts measures the
  generalisation gap, not robustness.

The point the rest of the chapter builds on survives all of this, in a slightly narrower
form: **once cross-entropy is satisfied, further optimisation of it is simply
uninformative about anything the objective does not measure.** The loss falls sevenfold
between epochs 75 and 150 while held-out accuracy does not improve at all. Robustness is
not a quantity the task objective measures, so driving that objective lower cannot be
expected to improve it. If we want robustness — at radii where it is genuinely at risk —
it has to enter the objective itself, which is what the rest of this chapter does.



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


