# Chapter 4 code: from ordinary training to property-driven training

This README is in three parts, and they are meant to be read in order.

1. **Training the vanilla classifier and checking its robustness.** A network trained
   on cross-entropy alone, verified with Chapter 3's command. This establishes the
   baseline (22 of 50 images provably robust at `epsilon 0.02`) and shows that training
   for longer erodes robustness rather than improving it.
2. **Running the property-driven training example.** The chapter's `pt_classifier.py`
   and `tf_classifier.py`, which train from scratch on the blended objective, and the
   libraries they need.
3. **Does it work? Continuing the vanilla network with property-driven training.** The
   experiment the chapter rests on: the baseline network from part 1, trained for ten
   more epochs with the specification in the objective, and every snapshot verified.
   Provable robustness rises from 22 to 27 of 50 with no loss of accuracy.

Everything needs **Vehicle 0.28.0 or later**; see the note on versions at the end of
part 3.

# Part 1: Training the Vanilla classifier and Checking its robustness

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
| 100 | 38/50 | **37/50** | **22/50** |
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

At `epsilon 0.02` the picture changes, and the checkpoints stop agreeing:

| epoch | correctly classified | provably robust | share of correctly classified |
| ----: | -------------------: | --------------: | ----------------------------: |
| 75 | 38 | 25 | 65.8% |
| 100 | 38 | 22 | 57.9% |
| 150 | 37 | 21 | 56.8% |

**Training for longer made the network less robust.** That is the expected direction, and
it is worth spelling out why. Once the 1024 training images are fitted, further
optimisation of cross-entropy cannot change which side of the decision boundary they fall
on — so it draws the boundary closer to them instead, buying greater confidence on
examples already answered correctly. The margin around each point narrows. A sufficiently
small neighbourhood never notices, which is why the `epsilon 0.005` column is flat; a
wider one does, and points that sat comfortably inside the correct region start to straddle
its edge.

So robustness is not simply a quantity cross-entropy fails to optimise. It is one that
optimising cross-entropy erodes, and the erosion stays invisible until the property is
checked at a radius wide enough to reveal it. This is the case for putting robustness into
the objective rather than expecting it to follow from fitting the data well.

The decline is orderly rather than a single jump — 25, then 22, then 21 — and it is not
the ceiling moving: correctly-classified goes 38, 38, 37 over the same checkpoints, down by
one, while provable robustness falls by four.

One caveat remains on precision. Fifty images resolve differences of only a few images, and
the correctly-classified count itself wanders by up to three across these epochs, so the
individual figures should not be read to the last image. What survives that caveat is the
ordering: three measurements in sequence, each lower than the last, on a strictly falling
training loss. A larger image set would sharpen the numbers but is unlikely to reverse
their direction.

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



# Part 2: Running the property-driven training example

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

- `vehicle-lang` - **0.28.0 or later is required**; tested with 0.28.0. Versions up to
  0.27.1 compile the specification to a loss whose adversarial search steps in the wrong
  direction, so the loss reports a property as *better* satisfied inside a wider
  neighbourhood, and training against it makes the network less robust. The scripts run
  without complaint on those versions, so check `vehicle --version` rather than waiting
  for an error. Part 3 records what the bug looked like and how it was confirmed fixed.

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

## Before you run: the output directory

`pt_classifier.py` now writes its network to `pdt-experiment/onnx_models/pdt_classifier.onnx`
and creates that directory itself, so nothing needs to be made in advance. Earlier
versions of both scripts wrote under `models/` without creating it, and failed at the
export step with `FileNotFoundError` after a fresh clone; if you are using a script that
still does, create the directory first:

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
    --saved-model pdt-experiment/onnx_models/tf_pdt_classifier \
    --output pdt-experiment/onnx_models/tf_pdt_classifier.onnx
```

(The `Saved to models/simple_classifier.onnx` line in the sample output above is from an
earlier version of the script; the current one prints
`Saved to pdt-experiment/onnx_models/pdt_classifier.onnx`.)

### What the run looked like under Vehicle 0.28.0

For the record, `pt_classifier.py` as it ships (seed 0, `alpha 0.5`, `epsilon 0.005`,
five epochs, Vehicle's default logic) printed the following on 2026-09-11:

```plain
Step 0:
	task loss:        2.2898
	constraint loss:  0.2350
	total loss:       1.2624
Step 1:
	task loss:        2.2174
	constraint loss:  0.0928
	total loss:       1.1551
...
Epoch: 1, mean total loss: 1.0139, train accuracy: 51.1%
Epoch: 2, mean total loss: 0.6981, train accuracy: 68.4%
Epoch: 3, mean total loss: 0.5285, train accuracy: 74.9%
Epoch: 4, mean total loss: 0.4549, train accuracy: 78.6%
Epoch: 5, mean total loss: 0.3991, train accuracy: 79.5%
Folded normalisation into the first layer for export (max deviation 1.3e-06)
Saved to pdt-experiment/onnx_models/pdt_classifier.onnx
```

(The `Folded normalisation` line is printed by the current script; see the next
subsection for what it does and why.) The network classifies 34 of the 50 held-out test
images correctly, against 38 for the 100-epoch vanilla network of part 1: five epochs
from scratch is not a trained classifier yet.

The task loss falls steadily. The constraint loss does not: it wanders between about
0.03 and 0.23 from step to step for the whole run. That is expected at `epsilon 0.005`
on a network that is still learning to classify: the constraint is measured on the
current batch, whose images the network mostly still gets wrong, and at this small radius
the robustness term is dominated by the misclassifications rather than by the margin.
Part 1 found that at `epsilon 0.005` even a well-trained network fails on only one
eligible image, so there is little for the constraint term to do here. The from-scratch
run therefore shows the mechanics, not the effect; part 3 is where the effect is measured.

Each step took about 40 s, so the five epochs took 55 minutes. Left to its own devices
PyTorch used every core and ran three times *slower* (about two minutes a step), which is
worth knowing before a long run; capping the threads through the environment fixes it
without touching the script:

```bash
OMP_NUM_THREADS=4 python pt_classifier.py
```

### The exported network, and why the script folds its normalisation

`pt_classifier.py` keeps its inputs in `[0, 1]` and normalises *inside* the network,
with a small `Normalize` module as the first layer, so that the network sees the same
input space the specification describes. That is the right design for training. It has
one consequence for verification, found when the network above was first handed to
Marabou: the module exports as ONNX `Sub` and `Div` operations, and Marabou 2.0.0
supports neither (nor `Mul`), so every one of the fifty images came back `errored`
with

```plain
Caught a MarabouError error. Code: 105, Errno: 0, Message: Onnx operation Div not currently supported by Marabou.
```

The script therefore folds the normalisation into the first linear layer at export
time. Normalisation is affine and so is the layer, so they compose exactly:

    W ((x - MEAN)/STD) + b  =  (W/STD) x + (b - (MEAN/STD) * rowsum(W))

The exported network contains only `Gemm` and `Relu` operations, computes the same
function as the trained one to within float32 rounding (the script checks this and
refuses to export otherwise; the deviation was `1.3e-6`), and takes the same raw
pixels. Training is untouched: only the file written at the end differs. The same
trick, applied after the fact to checkpoints trained with an external `Normalize`
transform, is in
[vanilla-experiment/folded-vanilla-models/fold_normalisation.py](vanilla-experiment/folded-vanilla-models/fold_normalisation.py),
and the trade-offs are discussed in [normalisation-options.md](normalisation-options.md).

If you write your own training script and Marabou reports every image as `errored`,
this is the first thing to check: `python -c "import onnx; print([n.op_type for n in onnx.load('model.onnx').graph.node])"`
lists the operations in the exported graph.

### Verifying the network it produces

With the fold in place, the network from the run above verifies with Chapter 3
Exercise #7's command. Run from this folder:

```bash
vehicle verify \
  --specification ../../chapter-3/exercises/FMNIST/fashionRobustness-solution.vcl \
  --network classifier:pdt-experiment/onnx_models/pdt_classifier.onnx \
  --parameter epsilon:0.005 \
  --dataset trainingImages:../../chapter-3/exercises/FMNIST/idxdata/0-49Images.idx \
  --dataset trainingLabels:../../chapter-3/exercises/FMNIST/idxdata/0-49Labels.idx \
  --solver Marabou
```

Because `pt_classifier.py` fixes its seed, a re-run reproduces the network exactly (the
two runs made here exported identical weights), so these counts are reproducible rather
than a sample of one. Full transcripts are in `pdt-experiment/marabou-outputs/`.

| `epsilon` | correctly classified | provably robust | genuinely non-robust | robust share of eligible | solver |
| --------: | -------------------: | --------------: | -------------------: | -----------------------: | -----: |
| 0.005 | 34/50 | **32/50** | 2 | 94.1% | 2021 s |
| 0.02 | 34/50 | **29/50** | 5 | 85.3% | 2082 s |

Set beside part 1's networks at `epsilon 0.02`, where the vanilla checkpoints proved
25, 22 and 21 of 38, 38 and 37 correctly classified images (65.8%, 57.9%, 56.8%), this
five-epoch network proves 29 of its 34, or 85.3%. Two cautions before reading too much
into that. The network is far less trained (five epochs against 75 to 150, 34 correct
against 38), and part 1 showed that *less* cross-entropy training means wider margins, so
some of the difference is that and not the constraint term. And the constraint was
trained at `epsilon 0.005`, not 0.02. The like-for-like comparison is with the vanilla
script run for the same five epochs, below; the controlled experiment, where the only
difference is the constraint term, is part 3.

### Like for like: the two chapter scripts, five epochs each

`vanilla_classifier.py` as it ships is the same network, data, optimiser and epoch count
as `pt_classifier.py` with the constraint term left out (and no fixed seed). Its
five-epoch output, `vanilla-experiment/onnx_models/vanilla_classifier.onnx`, verified
with the same command:

| network | `epsilon` | correctly classified | provably robust | genuinely non-robust | robust share of eligible | solver |
| --- | --------: | -------------------: | --------------: | -------------------: | -----------------------: | -----: |
| `vanilla_classifier.py`, 5 epochs | 0.005 | 33/50 | **30/50** | 3 | 90.9% | 750 s |
| `pt_classifier.py`, 5 epochs | 0.005 | 34/50 | **32/50** | 2 | 94.1% | 2021 s |
| `vanilla_classifier.py`, 5 epochs | 0.02 | 33/50 | **25/50** | 8 | 75.8% | 680 s |
| `pt_classifier.py`, 5 epochs | 0.02 | 34/50 | **29/50** | 5 | 85.3% | 2082 s |

The two networks classify almost the same number of test images correctly, 33 and 34,
so their ceilings match and the robustness columns can be read directly against each
other. At both radii the property-driven network proves more images and leaves fewer
correctly classified images breakable: at `epsilon 0.02`, 29 against 25 verified and 5
against 8 genuinely non-robust. The difference is modest, as it should be after five
epochs at `epsilon 0.005`, and the vanilla run is unseeded and single, so this pair is
consistent with part 3 rather than independent proof of it. Transcripts are in
`vanilla-experiment/marabou-outputs/verify_vanilla_classifier_5ep_eps*.txt`.

Note the solver times once more: the property-driven network took three times longer to
verify at the same radius. Proofs are more work than counterexamples, and a network that
verifies quickly is usually one that is being falsified.

Both scripts train **from scratch** for five epochs. That is enough to see the mechanics
and to watch the constraint loss fall, but five epochs from random weights is not enough
to say anything about robustness: the network is still learning to classify. The
experiment that shows property-driven training working is in part 3, and it takes the
opposite approach: it starts from a fully trained classifier and adds the constraint.

# Part 3: Does it work? Continuing the vanilla network with property-driven training

Part 1 ended with a network that classifies well and whose provable robustness *falls*
as cross-entropy training continues. This part takes that network, the 100-epoch
checkpoint `vanilla_e100.onnx`, and trains it for ten more epochs on the blended
objective

    total = ALPHA * cross_entropy + (1 - ALPHA) * constraint_loss

where `constraint_loss` is the robustness specification compiled to a loss by Vehicle.
Every epoch's network is verified with Chapter 3 Exercise #7's command, unchanged, so the
results sit directly beside part 1's table.

Everything for this part lives in [capucci-pdt/](capucci-pdt/), and its own
[README](capucci-pdt/README.md) is the complete laboratory record, including two earlier
runs made under Vehicle 0.27.1 that failed because of the bug described at the end of
this part. What follows is the replication path.

## Step 1: check the prerequisites

```bash
vehicle --version      # must be 0.28.0 or later
Marabou --version      # 2.0.0
python -c "import torch, torchvision, onnx, onnxscript; print('ok')"
```

The Fashion MNIST training images are downloaded on first use into `capucci-pdt/data/`.

## Step 2: what is in the folder

| file | role |
| --- | --- |
| `vanilla_e100.onnx` | the starting network: part 1's 100-epoch checkpoint, 99.5% training accuracy, 38/50 test images correct, 22/50 provably robust at `epsilon 0.02` |
| `fashionRobustness-solution.vcl` | Chapter 3 Exercise #7's specification, used for **verification**; a verbatim copy |
| `fashionRobustness-capucci.vcl` | the same property in the form the loss compiler accepts, plus a differentiable logic; used for **training** |
| `0-49Images.idx`, `0-49Labels.idx` | the fifty held-out test images Exercise #7 checks; verbatim copies |
| `pdt-Capucci-v028.py` | the training script |
| `verify_v028.py` | verifies each snapshot as the trainer produces it |
| `record_results_v028.py` | regenerates the results table in `capucci-pdt/README.md` from the CSV traces |
| `capucci-models-v028/` | the ten snapshots the run described here produced |
| `marabou-outputs-v028/` | the ten complete solver transcripts |

The folder is self-contained so that the experiment can be re-run without reaching across
the repository.

## Step 3: the training specification

`fashionRobustness-capucci.vcl` differs from the verification specification in two ways.

**`advises` is stated non-strictly.** Exercise #7 writes

    advises image label = forall j . j != label => classifier image ! label > classifier image ! j

but the loss compiler does not yet support comparing two `Index` values, so `j != label`
cannot be compiled. Dropping the guard while keeping `>` would demand a score strictly
greater than itself when `j = label`. Weakening to `>=` makes that case trivially true:

    advises image label = forall j . classifier image ! label >= classifier image ! j

The two forms differ only when the advised label *ties* with another. On this problem
they return identical verdicts for every image and every snapshot (checked in the
laboratory record), so training on the non-strict form and verifying on the strict one is
a like-for-like comparison.

**A differentiable logic is declared.** Vehicle needs to know how to interpret `and`,
`forall`, `>=` and the rest as real-valued operations. The built-in choices are
`VehicleDifferentiableLogic` and `DL2DifferentiableLogic`; this specification instead
declares its own, the additive quantitative linear logic of Capucci et al. that the
chapter text introduces:

```vehicle
p : Real
p = 2.0

qllAdditive : DifferentiableTensorLogic
qllAdditive =
  { trueElement                = -infinity
  , falseElement               = infinity
  , pointwiseNegation          = \x -> -x
  , pointwiseConjunction       = \{dims} x y -> (const (1/p) dims) * log(exp(const p dims * x) + exp(const p dims * y))
  , pointwiseDisjunction       = \{dims} x y -> -(const (1/p) dims) * log(exp(const (-p) dims * x) + exp(const (-p) dims * y))
  , pointwiseLessThan          = \x y -> x - y
  , pointwiseLessEqualThan     = \x y -> x - y
  , pointwiseGreaterThan       = \x y -> y - x
  , pointwiseGreaterEqualThan  = \x y -> y - x
  , pointwiseEqual             = \x y -> max (x - y) (y - x)
  , pointwiseNotEqual          = \x y -> - max (x - y) (y - x)
  , reduceConjunction          = \{dims} xs -> (1/p) * log(reduceAdd (exp (const p dims * xs)))
  , reduceDisjunction          = \{dims} xs -> (1/p) * log(reduceAdd (exp (const (-p) dims * xs)))
  }
```

Smaller is truer: a comparison `a >= b` becomes `b - a`, which is at most zero when it
holds, and conjunction is a smooth maximum (log-sum-exp) whose sharpness is set by the
hardness `p`. Because `advises` quantifies over *all* labels including the advised one,
the `j = label` term contributes exactly zero, and the compiled loss for an image is
therefore zero when the image is robust throughout its neighbourhood and positive
otherwise. That is the signal the optimiser follows.

## Step 4: the training script

`pdt-Capucci-v028.py` is short enough to read in full; these are its moving parts.

**It continues the vanilla network rather than training from scratch.** The weights are
read straight out of the ONNX file's initialisers into the equivalent PyTorch module:

```python
weights = {init.name: numpy_helper.to_array(init)
           for init in onnx.load("vanilla_e100.onnx").graph.initializer}
state = model.state_dict()
for key in ("1.weight", "1.bias", "3.weight", "3.bias", "5.weight", "5.bias"):
    state[key] = torch.tensor(weights[key])
model.load_state_dict(state)
```

**The specification is compiled to a loss, naming the logic declared in it:**

```python
spec = loss_pt.load_specification("fashionRobustness-capucci.vcl",
                                  logic=vcl.CustomDifferentiableLogic("qllAdditive"))
constraint_loss_fn = spec["robust"]
```

`spec["robust"]` is a Python function with one argument per resource the specification
declares: the network, `epsilon`, the two datasets and the inferred size `n`. It returns
one loss value per image.

**Each step blends the two losses and takes one optimiser step:**

```python
constraint_loss = constraint_loss_fn(n=BATCH_SIZE, classifier=network,
                                     epsilon=torch.tensor(EPSILON),
                                     trainingImages=images.squeeze(1),
                                     trainingLabels=labels)
constraint_loss = torch.stack(constraint_loss).mean()
task_loss = cross_entropy(model(images), labels)
total_loss = ALPHA * task_loss + (1 - ALPHA) * constraint_loss
total_loss.backward()
optimizer.step()
```

**After every epoch it exports the network** to `capucci-models-v028/capucci_v028_eNN.onnx`
and records the epoch's mean losses, training accuracy and the number of the fifty test
images classified correctly in `traces-v028/per_epoch.csv`.

The settings, all constants at the top of the file:

| setting | value | why |
| --- | --- | --- |
| `EPSILON` | 0.02 | the radius at which the vanilla network is genuinely vulnerable (part 1: 16 of 38 eligible images fail); at 0.005 there is nothing to improve |
| `ALPHA` | 0.4 | 40% cross-entropy, 60% constraint; the constraint carries slightly more weight |
| images | 1024, batch 64 | the same subset part 1 trained on; 16 steps per epoch |
| epochs | 10 | |
| optimiser | Adam, lr 1e-3 | as in part 1 |
| clamping, gradient clipping | none | |

Run it from the folder:

```bash
cd capucci-pdt
python3 pdt-Capucci-v028.py
```

It prints a line every four steps and one per epoch:

```plain
[16:38:31] vehicle_lang 0.28.0; torch 2.13.0+cpu; 4 threads
[16:38:31] loaded vanilla_e100.onnx: continuing that network, not training from scratch
[16:38:32] starting point classifies 38/50 of the test images correctly
[16:38:32] 10 epochs | qllAdditive logic | epsilon 0.02 | alpha 0.4 (task) / 0.6 (constraint) | 1024 images, 16 steps/epoch
[16:41:37]   epoch 1 step 4/16: constraint +0.6282 CE 0.0472
...
[16:52:19] epoch   1: constraint +0.5420 | CE 0.0592 | blended +0.3489 | train acc 98.7% | test 37/50 | 823s
...
[18:32:58] epoch  10: constraint +0.2880 | CE 0.0628 | blended +0.1979 | train acc 99.0% | test 40/50 | 637s
```

Each batch takes about 40 s, because the constraint loss runs an adversarial search
(ten random starts, five gradient steps each) for every image. An epoch is ten to
fourteen minutes and the whole run just under two hours on a CPU. The script pins PyTorch
to four threads; with more it was seen to run slower, not faster.

## Step 5: verify every snapshot

`verify_v028.py` waits for each snapshot to appear and runs Exercise #7's command on it,
so it can be started in a second terminal as soon as training begins:

```bash
cd capucci-pdt
python3 verify_v028.py
```

It is nothing more than a loop around

```bash
vehicle verify \
  --specification fashionRobustness-solution.vcl \
  --network classifier:capucci-models-v028/capucci_v028_e01.onnx \
  --parameter epsilon:0.02 \
  --dataset trainingImages:0-49Images.idx \
  --dataset trainingLabels:0-49Labels.idx \
  --solver Marabou \
  --solver-args --timeout=120
```

with three guards: a 120 s Marabou timeout per image, a two-hour wall clock per model, and
a 24 GB address-space cap, so that one hard query cannot stall the run or take the
machine down. It appends each verdict to `traces-v028/verify.csv` and writes the full
transcript to `marabou-outputs-v028/`. Verifying fifty images took between 12 and 42
minutes per snapshot and about 14 GB of memory; run one at a time.

To verify a single snapshot by hand, or with a different epsilon, run the command above
directly.

## Step 6: results

| epoch | constraint | cross-entropy | train acc | correct on 50 test | verified | genuinely non-robust | robust share of eligible | solver |
| ----: | ---------: | ------------: | --------: | -----------------: | -------: | -------------------: | -----------------------: | -----: |
| 0 (start) | -- | 0.0413 | 99.5% | 38/50 | **22/50** | 16 | 57.9% | 924 s |
| 1 | +0.5420 | 0.0592 | 98.7% | 37/50 | **27/50** | 10 | 73.0% | 2305 s |
| 2 | +0.4620 | 0.0595 | 98.9% | 39/50 | **24/50** | 14 | 61.5% | 2139 s |
| 3 | +0.4155 | 0.0578 | 98.7% | 39/50 | **26/50** | 13 | 66.7% | 1748 s |
| 4 | +0.3714 | 0.0584 | 98.5% | 38/50 | **24/50** | 14 | 63.2% | 720 s |
| 5 | +0.3340 | 0.0537 | 99.3% | 38/50 | **25/50** | 13 | 65.8% | 740 s |
| 6 | +0.3617 | 0.0631 | 98.8% | 38/50 | **25/50** | 13 | 65.8% | 721 s |
| 7 | +0.2918 | 0.0525 | 99.3% | 39/50 | **25/50** | 14 | 64.1% | 2505 s |
| 8 | +0.2896 | 0.0551 | 99.0% | 40/50 | **26/50** | 14 | 65.0% | 757 s |
| 9 | +0.2830 | 0.0587 | 98.9% | 38/50 | **27/50** | 11 | 71.1% | 836 s |
| 10 | +0.2880 | 0.0628 | 99.0% | 40/50 | **27/50** | 13 | 67.5% | 832 s |

The columns are the same as part 1's. `genuinely non-robust` is the falsified count less
the images the snapshot misclassifies, since a misclassified image fails at zero
perturbation for reasons that have nothing to do with robustness; `robust share of
eligible` is `verified` over `correct on 50 test`. No image errored; one query (image 6 of
the epoch-2 model) timed out and is counted as unverified.

**Property-driven training gained provable robustness, on every snapshot.** The starting
network proves 22 of the 50 images. All ten snapshots prove between 24 and 27, and the
final one proves 27. As a share of the images each network classifies correctly, the
baseline's 57.9% became 61.5% to 73.0%.

**It cost no accuracy.** The number of test images classified correctly never fell below
37 and finished at 40, two above the starting network; cross-entropy stayed between 0.052
and 0.063 throughout. The final snapshot is better than the starting network on both axes
at once. Part 1 showed that cross-entropy on its own erodes robustness once the data is
fitted; the constraint term reverses that without the task term having to give anything
up.

**Two things worth knowing beyond the headline.**

- The verified count plateaus after epoch 1 while the loss keeps falling. The ten counts
  were 27, 24, 26, 24, 25, 25, 25, 26, 27, 27. The loss is measured on the 1024 training
  images through an FGSM search, and the verification is exact on 50 held-out images, so
  later loss improvement is partly fitting rather than generalising, and partly tightening
  margins on perturbations the search can find but Marabou is not limited to.
- Consecutive snapshots differ by up to three images with no change in the objective.
  Twenty images are proved by all ten snapshots, sixteen by none, and fourteen flip. So a
  single snapshot carries an error bar of about plus or minus two on this test set. The
  evidence for the gain is that all ten sit above 22, not any one of them.

Total wall time was about four hours: two for training, and 3.7 hours of solver time for
the ten verifications, which overlapped with training.

## A note on Vehicle versions, and how the bug was found

This experiment was first run on 2026-08-28 under Vehicle 0.27.1, twice, and both runs
made the network *less* robust: one diverged to `nan` within three epochs, the other ran
to completion and halved the number of provably robust images. Their full records are in
[capucci-pdt/README.md](capucci-pdt/README.md), and the reproducers that isolated the
symptom are in [../exercises/temp/](../exercises/temp/README.md).

The symptom was that the compiled loss reported a property as *better* satisfied when
`epsilon` was widened and when the adversarial search was given more effort, which is the
opposite of what a `forall` requires. The cause, located by comparing the 0.27.1 and
0.28.0 Python packages, was one sign in the sampler's FGSM step: it descended the loss
instead of ascending it, so the search behind `forall perturbation` returned the least
violating perturbation rather than the most. Vehicle 0.28.0 (released 2026-09-10) fixes
it. Re-running the same sweep under 0.28.0, the loss increases with `epsilon` under all
three logics, and the run in this part is the result.

Two lessons carry over to any property-driven training project:

1. **Check the loss against the quantifier before training.** Evaluate the compiled loss
   on a fixed network at several values of `epsilon`. For a `forall` over the
   neighbourhood it must not decrease as the neighbourhood grows. This takes seconds and
   would have caught the bug on day one.
2. **Watch for the loss and the verifier disagreeing.** A constraint loss that falls while
   the verified count falls too is not a training difficulty; it means the surrogate and
   the property have come apart, and the pipeline, not the hyper-parameters, is where to
   look. The same habit found the normalisation mismatch recorded in
   [normalisation-options.md](normalisation-options.md).


