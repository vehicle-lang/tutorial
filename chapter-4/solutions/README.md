# Chapter 4 model solutions and expected results

These results were obtained with Vehicle 0.28.0 and Marabou 2.0.0 on a 12-core CPU
machine with 30 GB of memory. Trained networks and complete solver transcripts are
committed under `../chapter-code`, so every number below can be checked without
retraining.

## Exercise #1: running the chapter code

Expected outputs for each script are recorded in the chapter-code README, next to the
commands that produce them. In brief:

| script | what it produces | expected |
| --- | --- | --- |
| `vanilla_classifier.py` | `vanilla-experiment/onnx_models/vanilla_classifier.onnx` | 5 epochs, mean loss falling from about 2.1 to about 0.85 and training accuracy rising to about 69% (no seed, so runs vary by several points) |
| `pt_classifier.py` | `pdt-experiment/onnx_models/pdt_classifier.onnx` | task, constraint and total loss printed per step; task loss falling from 2.29 to about 0.6, constraint loss wandering between 0.03 and 0.23, training accuracy 79.5% after 5 epochs (seed 0, so reproducible); about 40 s per step with `OMP_NUM_THREADS=4` |
| `capucci-pdt/pdt-Capucci-v028.py` | ten snapshots in `capucci-models-v028/` | constraint loss falling from about 0.54 to about 0.29 over ten epochs, cross-entropy staying near 0.06 |
| `capucci-pdt/verify_v028.py` | `traces-v028/verify.csv` and ten transcripts | 24 to 27 of 50 images proved robust per snapshot |

The network `pt_classifier.py` exports verifies to 32/50 at `epsilon 0.005` and 29/50 at
`epsilon 0.02`, classifying 34 of the 50 test images correctly; the transcripts are in
`../chapter-code/pdt-experiment/marabou-outputs/`. If instead every image comes back
`errored` with `Onnx operation Div not currently supported by Marabou`, you are running
a version of the script from before 2026-09-12, which exported its normalisation layer as
operations Marabou cannot read; the chapter-code README explains the fix.

If `pt_classifier.py` reports a constraint loss that is exactly zero from the first step,
the property is trivially satisfied at that epsilon and the run says nothing; if it
reports a *negative* constraint loss, you are running a Vehicle version older than
0.28.0.

## Exercise #2: with and without the logical loss

The comparison the chapter makes, on the fifty held-out Chapter 3 images at
`epsilon 0.02`:

| network | correctly classified | provably robust | of the correctly classified |
| --- | ---: | ---: | ---: |
| `vanilla_e100.onnx`: cross-entropy only, 100 epochs | 38/50 | 22/50 | 57.9% |
| `capucci_v028_e10.onnx`: plus 10 epochs of the blended objective | 40/50 | **27/50** | 67.5% |

**Which is more robust?** The property-driven one, by five images, and by ten
percentage points of the images it could possibly be proved robust on. Every one of the
ten intermediate snapshots is also above the baseline (the full table is in the
chapter-code README), so this is not a lucky epoch.

**Which has the better task accuracy?** Also the property-driven one, marginally: 40
against 38 on the held-out images, and training accuracy within half a percent of the
vanilla network's 99.5% throughout. On this problem the blended objective did not trade
accuracy for robustness. Fifty images resolve differences of only a few images, so the
accuracy gain itself is within noise; what is not within noise is that accuracy did not
fall.

**Why?** Cross-entropy is minimised by making the network confident about the training
images, and once they are all classified correctly the only way to grow more confident is
to move the decision boundary closer to them. That erodes the margin, and the vanilla
experiment shows the erosion directly: robustness at `epsilon 0.02` falls from 25 to 22
to 21 across epochs 75, 100 and 150. The constraint loss, compiled from the
specification, is zero for an image whose whole `epsilon`-neighbourhood is classified
correctly and positive otherwise, so it exerts a force only on the images that are
breakable, pushing the boundary away from them. The task loss holds the classification in
place while that happens. Neither term alone does the job: cross-entropy alone erodes
robustness, and the constraint term alone would be perfectly satisfied by a network that
classifies everything the same way.

Two details of the comparison are worth knowing when you make your own:

- The specification the network was *trained* on states `advises` non-strictly (`>=`),
  because the loss compiler does not yet support the `j != label` guard of Chapter 3's
  strict form. Both networks were *verified* against Chapter 3's strict specification,
  and on this problem the two forms return identical verdicts for every image, so the
  comparison is like for like.
- The two networks here share their first 100 epochs. Comparing two networks trained
  from scratch, one with and one without the constraint term, is a fair variant, but the
  random initialisation and shuffling then add noise of their own; fixing a seed, as
  `pt_classifier.py` does, helps.

**The from-scratch variant, using the two chapter scripts as shipped** (five epochs each,
`pt_classifier.py` with the constraint at `epsilon 0.005`, `alpha 0.5`, default logic):

| network | correctly classified | robust at 0.005 | robust at 0.02 |
| --- | ---: | ---: | ---: |
| `vanilla_classifier.py` | 33/50 | 30/50 | 25/50 |
| `pt_classifier.py` | 34/50 | **32/50** | **29/50** |

Same answer, smaller margin: with the ceilings matched at 33 and 34, the property-driven
network proves two more images at the radius it was trained at and four more at the
wider one. Five epochs is early and the vanilla run is unseeded, so treat this as
consistent with the main result rather than a second proof of it. Both networks and all
four transcripts are committed under `../chapter-code`.

## Exercise #3: further experimentation

Only one configuration has been run to completion under Vehicle 0.28.0, the one in the
table above (Capucci `qllAdditive` logic, `ALPHA 0.4`, `EPSILON 0.02`), so this exercise
is genuinely open. What is known:

**The choice of logic changes the scale of the loss, not its direction.** Measured on an
untrained network and four random images, with gradients enabled:

| `epsilon` | Vehicle default | DL2 | `qllAdditive` |
| ------: | ------: | --: | ----------: |
| 0.0 | 0.62 | 0.62 | 1.20 |
| 0.02 | 0.90 | 0.90 | 1.24 |
| 0.5 | 4.57 | 4.57 | 1.71 |
| 2.0 | 6.07 | 6.07 | 1.94 |

All three increase with `epsilon`, as a worst-case quantifier must. The default and DL2
logics coincide on this specification. The Capucci logic is flatter in `epsilon`, which
is a consequence of its log-sum-exp conjunction with hardness `p = 2`; a larger `p`
approaches the hard `max` and steepens it. Which scale trains better is not settled, and
`ALPHA` interacts with it: the same `ALPHA` weights a loss of 1.2 quite differently from
a loss of 0.9.

**`ALPHA = 0` is a degenerate case.** With no task term, a constant classifier satisfies
robustness perfectly, and accuracy has nothing holding it up. An earlier run of exactly
this configuration is recorded in `../chapter-code/pure-pdt.md`; it was made under
Vehicle 0.27.1 with the inverted loss, so its numbers should not be quoted, but the
expectation that accuracy collapses stands.

**`ALPHA = 1` is the vanilla experiment**, whose results are in the chapter-code README.

**On multiple constraints.** Nothing has been measured. Chapter 3's Exercises #4 to #6
give alternative robustness definitions that compile to losses; note that a definition
Marabou cannot verify (standard robustness, Euclidean balls) can still be *trained*
against, since the loss compiler has no such restriction, but you then cannot check the
result with the chapter's command.

The specification with the custom logic, `../chapter-code/capucci-pdt/fashionRobustness-capucci.vcl`,
is the model solution for "use a different constraint loss function": the
`DifferentiableTensorLogic` block at its foot is all that is needed to define one.

## Exercise #4: normalisation inside the network

**What Marabou says.** With a `Normalize` layer exported as part of the network, every
one of the fifty images is reported as `errored`:

```plain
Caught a MarabouError error. Code: 105, Errno: 0, Message: Onnx operation Div not currently supported by Marabou.
```

The layer becomes ONNX `Sub` and `Div` nodes, and Marabou 2.0.0's ONNX reader accepts
neither. Rewriting it as a multiply and an add does not help: `Mul` is rejected too. The
operations it does accept include `Gemm`, `Relu` and `Reshape`, which is all a plain
multi-layer perceptron needs.

**The fix.** Normalisation is affine and so is the first linear layer, so they compose
into a single linear layer with weights `W / STD` and bias `b - (MEAN / STD) * rowsum(W)`.
`pt_classifier.py`'s `fold_normalisation` does this on a copy of the trained model just
before export, and asserts that the folded and original networks agree on 256 random
inputs before writing anything; the observed deviation was `8e-7`. The exported graph
is `Reshape, Gemm, Relu, Gemm, Relu, Gemm`, and verification then works: 34/50 correct,
32/50 robust at `epsilon 0.005`, 29/50 at `epsilon 0.02`.

**Why the fold is the right fix, and not only a workaround.** The folded network takes raw
`[0, 1]` pixels, which is the space the specification's `validImage` describes and the
space the `.idx` files live in, so `epsilon` means the same perturbation in training and
in verification. Normalising *outside* the network, with a `transforms.Normalize` step,
would have broken that: the chapter's earlier attempt did exactly this, and
`../chapter-code/normalisation-options.md` records what it cost and the three ways to
repair it.

## Exercise #5: a model from scratch

No model solution is provided; the point is to build your own. Expected results, from
the chapter's experience with the 784-64-32-10 network:

- Fashion MNIST is hard enough that a network trained on 1024 images misclassifies ten to
  thirteen of the fifty test images whatever you do, which caps provable robustness well
  below 50.
- At `epsilon 0.005` almost every correctly classified image is provably robust for any
  reasonable network, so comparisons need `epsilon 0.02` or wider.
- A smaller network has a smoother decision boundary and tends to be *easier* to prove
  robust per correctly classified image, but classifies fewer images correctly. The two
  effects pull the raw verified count in opposite directions, which is why the count
  should always be read against the number of eligible images.

## Exercise #6: properties beyond epsilon-ball robustness

No model solution is provided yet. The Iris properties are bounds on a four-dimensional
input space with conclusions that are not a fixed label, so this is the first exercise in
which the generality of property-driven training over adversarial training matters.
Contributions of a worked solution are welcome.
