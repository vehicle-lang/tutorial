# The normalisation mismatch, and three ways to fix it

While running property-driven training on the vanilla network we found that the
training pipeline and the verification pipeline disagreed about what an input image
is. This file records what the defect was, how it was found, what it cost, and the
three available fixes with their trade-offs.

## The defect

Training normalised its inputs:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,))     # MEAN 0.2860, STD 0.3530
])
```

so the network learned to read pixels in roughly **[-0.810, 2.023]**. Verification
reads the `.idx` datasets, whose pixels are float64 in **[0.0, 1.0]**. Marabou was
therefore asked about a network operating outside the input scale it was trained for:
a pixel of full intensity should have arrived as 2.023 and arrived as 1.0.

Three separate things follow from that one line.

**1. The verified network is not the trained network's operating regime.** This
affects every result, including the pure cross-entropy baseline.

**2. `epsilon` names two different cubes.** Epsilon 0.005 applied in normalised units
during training is only `0.005 * 0.3530 = 0.00176` in raw pixel units, while
verification checks 0.005 in raw units. Training was enforcing a cube **2.83x
tighter** than the one being verified. This affects property-driven training only:
the vanilla runs use cross-entropy alone and have no training epsilon at all.

**3. The specification's validity precondition is false.**

```
validImage x = forall i j . 0 <= x ! i ! j <= 1
```

cannot hold of normalised pixels. In the Boolean reading the implication in
`robustAround` would simply be discharged as vacuously true; under a differentiable
logic it instead contributes a soft value driven by an antecedent that is
substantially false, throughout training.

## How it was found

Not by reading the code. Two symptoms pointed at it:

- Across ten epochs of constraint-loss-only training the **constraint loss rose**,
  from 0.6221 to 1.6688. Ten epochs of optimising an objective and nothing else had
  left the network further from satisfying it.
- The constraint loss and the verified property moved in **opposite directions**.
  Between epoch 5 and epoch 10 the loss worsened from 0.8378 to 1.6688 while provable
  robustness improved from 2/50 to 9/50. A faithful surrogate cannot do that.

The confirming test is cheap and worth keeping as a diagnostic: run each network on
the 50 verified images twice, once with raw pixels and once normalised, and see which
it prefers.

| Network | accuracy, raw [0,1] | accuracy, normalised | trained for |
| --- | ---: | ---: | --- |
| Chapter 3's `fashion1l32n` | **92.0%** | 40.0% | raw |
| our vanilla e200 | 60.0% | **76.0%** | normalised |

Chapter 3's supplied network was trained on raw pixels, so its 40/50 was correct all
along. Ours was not.

## What it cost

An image the network already misclassifies cannot be robust: `advises perturbedImage
label` fails at zero perturbation. So misclassifications appear in the falsified
column for reasons that have nothing to do with robustness.

| vanilla e200 | correct | verified | falsified | of which misclassified | genuinely non-robust |
| --- | ---: | ---: | ---: | ---: | ---: |
| as published (mismatched) | 30/50 | 29/50 | 21/50 | 20 | 1 |
| folded to raw inputs | 38/50 | **36/50** | 14/50 | 12 | 2 |

The published 29/50 was largely a misclassification count. Correcting the input space
lifts provable robustness to 36/50, and the corrected plateau across the three
checkpoints is 36/36/36 against accuracy pinned at 38/50 -- flat, where the published
figures were 28/29/29.

## The three fixes

### Option 1 -- drop `Normalize`, train on raw [0,1]

Delete the `transforms.Normalize(...)` line. The network then reads the same space the
specification and the datasets talk about.

**Pros**

- Fixes all three defects at once. The cubes agree because both sides are raw;
  `validImage` becomes true; the verified network is the trained network.
- Simplest possible change: it removes a line rather than adding a concept.
- Nothing to explain in the tutorial. The input space visibly matches the
  specification, which is exactly the property a teaching example should have.
- No weight surgery, no new ONNX operators, no second code path.

**Cons**

- Requires retraining. Every existing checkpoint and every published number has to be
  regenerated.
- Gives up whatever conditioning benefit normalisation provides. For a three-layer MLP
  on FashionMNIST this is small but not exactly zero.
- Departs from the standard torchvision recipe, so readers who have met `Normalize` in
  other MNIST tutorials may wonder why it is absent. Worth a sentence of explanation.

### Option 2 -- fold the normalisation into the first layer

Normalisation is affine and the first layer is affine, so they compose exactly:

```
W ((x - MEAN)/STD) + b  =  (W/STD) x + (b - (MEAN/STD) * rowsum(W))
```

Implemented in `fold_normalisation.py`. Validated on all three vanilla checkpoints:
maximum output deviation 7.9e-6, argmax agreement 100% over 512 random inputs, checked
through onnxruntime as well as PyTorch.

**Pros**

- Exactly equivalent, not approximately. The folded network *is* the same function
  over a rescaled input space.
- **The only option that salvages models already trained.** No retraining, so existing
  checkpoints and their training statistics stay valid.
- Adds no ONNX operators, so Marabou needs no capability it did not already have.
- If training also runs through the folded model, the inputs are raw and all three
  defects are fixed.

**Cons**

- Weight surgery has to be explained in a tutorial, which is a real pedagogical cost
  for a chapter whose subject is something else.
- Valid only because the first operation after `Flatten` is affine and the
  normalisation is a single scalar mean and standard deviation. It does not generalise
  to a per-pixel normalisation, or to a first layer that is not linear.
- If applied **only at export** and not during training, it fixes the input-space
  defect but leaves the epsilon cube mismatched, because training still perturbs
  normalised images. Applied at export alone it is a fix for verification, not for
  property-driven training.
- Two code paths -- trained model and exported model -- that must be kept consistent.

### Option 3 -- keep normalisation, scale the training epsilon

Set the training epsilon to `0.005 / 0.3530 = 0.014164` so the training cube matches
the verified one.

**Pros**

- One line, no retraining, no weight surgery.
- Keeps the familiar torchvision recipe intact.
- Makes the two cubes agree exactly, which is the defect it targets.

**Cons**

- **It fixes only the second defect, and that is the least damaging of the three.**
  The network is still trained on normalised inputs and verified on raw ones, so the
  misclassification problem that produced 20 of the 21 falsifications in the published
  baseline remains untouched.
- `validImage` stays false of every input the loss sees, so the soft-implication
  distortion remains.
- The epsilon in the code no longer equals the epsilon in the specification. A reader
  comparing the two sees different numbers for what is meant to be the same property --
  a poor state of affairs in a tutorial about specifications meaning what they say.

## Recommendation

Options 1 and 2 are **complementary, not alternatives**:

- **Option 1 for the chapter code going forward.** It is the fix that makes the
  example correct and needs no explanation.
- **Option 2 for the checkpoints that already exist**, so the corrected vanilla table
  can be produced without retraining, as it was here.
- **Option 3 is not sufficient.** It leaves the defect that did the most damage.

## Keeping the bug out

Two habits would have caught this immediately, and both are cheap enough to keep:

1. **Probe the input convention.** Run any network on a handful of the verification
   images twice, raw and normalised, and check which scaling it prefers. A network that
   scores 60% on the inputs its verifier supplies and 76% on inputs its verifier never
   sees is misconfigured, whatever the robustness count says.
2. **Watch for the loss and the solver disagreeing.** A constraint loss that rises
   while provable robustness improves, or vice versa, means the surrogate and the
   property have come apart. That is a pipeline bug, not a training difficulty.

## What fixing it exposed, and how that was resolved

Fixing the normalisation exposed something else. At epsilon 0.005 the number of provably
robust images is very nearly a restatement of accuracy: across the eight networks
measured, 82-97% of correctly-classified images were also proved robust, and on the
re-trained vanilla network only one of the 38 eligible images fails, at every checkpoint.
The robustness plateau in the corrected table is the accuracy plateau.

Epsilon 0.005 is therefore too small to make robustness a phenomenon distinct from
accuracy. That was an open question when this file was first written; it has since been
settled by measurement. Taking the 150-epoch network and the 37 images it classifies
correctly, and changing nothing but the radius:

| epsilon | provably robust | not robust |
| ------: | --------------: | ---------: |
| 0.005 | 36/37 | 1 |
| 0.02 | 21/37 | 16 |

A four-fold increase in the radius turns one failure into sixteen. Epsilon 0.02 is
consequently the radius at which the chapter's experiments can say anything: it leaves a
deficit large enough for a training method to improve on, where 0.005 leaves one image.

The two defects compound in a way worth noting. The original mismatch made epsilon mean a
cube 2.83x tighter during training than the one verified — and at these radii that factor
is the difference between a property that is nearly free and one that is substantially
violated. A scaling bug in the input space is not a small quantitative error; it can
change which regime the experiment is in.
