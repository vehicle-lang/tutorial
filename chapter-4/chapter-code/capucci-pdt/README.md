# Capucci property-driven training experiment

Continuing the epoch-100 vanilla classifier with property-driven training, using the
Capucci (QLL) differentiable logic. Set up 2026-08-28.

## Starting point

| | |
| --- | --- |
| Model | `vanilla_e100.onnx` |
| Provenance | epoch 100 of the raw-pixel vanilla re-run of 2026-08-27 |
| Checksum | `md5 fa18b3a03881498dbbf82ab1c6f7a631` (byte-identical to the source) |
| Trained on | 1024 FashionMNIST training images, raw `[0,1]` pixels, cross-entropy only |
| Mean loss | 0.0413 |
| Train accuracy | 99.5% |

Its verified behaviour on the 50 images of Chapter 3 Exercise #7 --- the first fifty
FashionMNIST **test** images, held out of training:

| | correctly classified | provably robust |
| --- | ---: | ---: |
| `epsilon 0.005` | 38/50 | 37/50 |
| `epsilon 0.02` | 38/50 | 22/50 |

At `epsilon 0.02` this leaves **16 of the 38 correctly-classified images not provably
robust**. That is the headroom property-driven training has to work with, and the number
any result here should be compared against.

The model was confirmed to be the raw-pixel checkpoint and not one of the older
normalised ones: it scores 76.0% on raw `[0,1]` inputs against 36.0% on normalised.

## Training settings

| Setting | Value | Note |
| --- | --- | --- |
| Script | `pdt-Capucci.py` | in this folder; run it from here |
| Differentiable logic | `qllAdditive` | Capucci / QLL, selected by name |
| Hardness degree `p` | 2.0 | a plain definition, not an `@parameter` (see below) |
| `EPSILON` | 0.02 | the radius at which the vanilla network is genuinely vulnerable |
| `ALPHA` | 0.4 | weight on the **task** loss, so 40% cross-entropy / 60% constraint |
| Clamping | none | removed 2026-08-28 |
| Images | 1024, batch 64 | 16 steps per epoch |
| Epochs | 10 | about 5.6 min each, so roughly an hour |
| Optimiser | Adam, lr 1e-3 | |

`ALPHA` follows the $\lambda$ of the chapter's objective, which weights the task term. At
0.4 the constraint term therefore carries the larger share.

Clamping was removed on request. It had capped each image's constraint loss at 0 before
averaging; because every image's raw value was already negative, the clamp made the
constraint term contribute exactly nothing and the run reduced to cross-entropy at half
weight. Without it the constraint loss is unbounded below, since `trueElement` is
`-infinity` in this logic.

## Verification

The trained model is to be verified with Chapter 3 Exercise #7's command, unchanged, so
the result is directly comparable with the table above.

Everything needed is kept in this folder, so the experiment can be run and re-run without
reaching across the repository. `fashionRobustness-solution.vcl` and the two `.idx` files
are verbatim copies of Chapter 3's originals (checksums confirmed identical).

```bash
vehicle verify \
  --specification fashionRobustness-solution.vcl \
  --network classifier:<trained-model.onnx> \
  --parameter epsilon:0.02 \
  --dataset trainingImages:0-49Images.idx \
  --dataset trainingLabels:0-49Labels.idx \
  --solver Marabou
```

## The training specification, and how it differs from Exercise #7's

`fashionRobustness-capucci.vcl` is the training specification. It follows Exercise #7's
`fashionRobustness-solution.vcl` with two changes, both needed to compile a loss:

1. the `qllAdditive` logic declaration is appended, which Vehicle needs in order to
   translate the property into a loss function. It does not affect the property.

2. `advises` is stated in the **non-strict** form:

   | | |
   | --- | --- |
   | Exercise #7 (verification) | `forall j . j != label => classifier image ! label > classifier image ! j` |
   | here (training) | `forall j . classifier image ! label >= classifier image ! j` |

   The original does not compile to a loss: `j != label` compares two values of type
   `Index 10`, and the backend rejects it with *"Loss functions do not yet support
   compilation of 'CompareIndex'"*. The guard cannot simply be dropped while keeping `>`,
   because the case `j = label` would then demand a score strictly greater than itself.
   Weakening the comparison to `>=` makes that case trivially true and the guard
   unnecessary.

   The two forms therefore differ **only on ties**: Exercise #7's property is strict, this
   one admits a tie between the advised label and another. The chapter's other training
   specification, `fmnist-robustness.vcl`, uses the same workaround for the same reason,
   and says so in its own comment.

Confirmed working: the specification typechecks, compiles to a loss under the
`qllAdditive` logic, and produces gradients (constraint loss $-1.18$ and total gradient
magnitude $69$ on a randomly initialised network).

**Which specification to verify against is not yet decided.** Verifying with Exercise
#7's strict form keeps the result comparable with the baseline table above, at the cost of
training and verifying subtly different properties. Verifying with the non-strict form
makes the two identical, but the baseline would have to be re-measured under it.

## How the network is continued

`pdt-Capucci.py` does not train from scratch. It reads the weights out of
`vanilla_e100.onnx` --- the ONNX initialisers --- straight into the equivalent PyTorch
module, so training resumes from exactly that checkpoint. All six weight tensors were
checked to load **bit-for-bit identically**.

A forward-pass comparison between the reloaded PyTorch model and the ONNX file under
onnxruntime differs by at most $1.3 \times 10^{-5}$ with 100% agreement on the predicted
class. That residue is float32 accumulation order between the two runtimes, not a
difference in weights.

Confirmed by a short run (1 epoch, 128 images): the script loads the checkpoint, trains,
writes `per_epoch.csv` and exports a snapshot. Train accuracy stayed at 99.2%, against
the starting network's 99.5% --- unlike the earlier from-scratch attempts, which collapsed
to chance accuracy because the constraint term had no learned classifier to preserve.

## Outputs

Everything the run produces lands in `traces/`, which is gitignored:

| File | Contents |
| --- | --- |
| `per_epoch.csv` | epoch, constraint loss, cross-entropy, blended loss, train accuracy, seconds |
| `model_eNN.onnx` | the network after each epoch, so any epoch can be verified |
| `train_log.txt` | whatever the run printed, if stdout is redirected there |

## Run 1: `pdt-Capucci.py`, positive sign --- diverged at epoch 3

Launched 11:36 on 2026-08-28 with the settings above. Stopped after epoch 3.

| epoch | constraint | cross-entropy | blended | train acc | correct on 50 test | seconds |
| ----: | ---------: | ------------: | ------: | --------: | -----------------: | ------: |
| 0 (start) | -- | 0.0413 | -- | 99.5% | **38/50** | -- |
| 1 | -0.3372 | 0.1531 | -0.1411 | 95.1% | **38/50** | 262 |
| 2 | -2.4780 | 1.3693 | -0.9391 | 81.1% | 31/50 | 285 |
| 3 | `nan` | `nan` | `nan` | 31.2% | 3/50 | 300 |

**Epoch 1 is the encouraging row.** The constraint loss improved roughly 25-fold over the
starting network while held-out accuracy did not move at all --- still 38 of the 50 test
images classified correctly, the same as the baseline. Whether that corresponds to
*provable* robustness is a separate question that only Marabou can answer.

**Epoch 2 shows the trade beginning.** The constraint loss improved a further sevenfold,
but seven test images were lost. Since a misclassified image can never be proved robust,
that lowered the ceiling on provable robustness from 38 to 31.

**Epoch 3 diverged.** The weights overflowed to `nan`: 39,305 of the 52,652 parameters in
`capucci_e03.onnx` are non-finite. Because `nan` is absorbing --- every subsequent gradient
is `nan` too --- the remaining seven epochs could not have recovered, and the run was
stopped rather than left to produce six more unusable snapshots.

The cause is the unbounded constraint loss. `trueElement` is `-infinity` in this logic, so
there is no floor: with 60% of the objective's weight and nothing bounding it, the
optimiser drove the term toward `-infinity` until float32 overflowed. Taken with the
earlier clamped run, both ends are now measured:

| constraint term | outcome |
| --- | --- |
| clamped at 0 | contributes exactly `0.0000`; the run reduces to cross-entropy at reduced weight |
| unclamped | diverges to `nan` within three epochs |

Neither is a usable configuration. That is a finding about the loss, not about the choice
of hyper-parameters --- and the standard remedy is to bound the *step* rather than the loss,
with gradient clipping, which was not used here.

`capucci_e01.onnx` and `capucci_e02.onnx` have finite weights and are worth verifying;
`capucci_e03.onnx` is corrupt. The traces of this run are kept as
`traces/per_epoch.diverged-run.csv` and `traces/train_log.diverged-run.txt`.

## Run 2: `pdt-Capucci-neg.py`, negated sign --- completed 10 epochs

`pdt-Capucci-neg.py` is a copy of the script differing in one character:

    total = ALPHA * cross_entropy - (1 - ALPHA) * constraint_loss

The motivation is the quantifier defect reported upstream: Vehicle 0.27.1 compiles a
`forall` into a loss that reports the property as *better* satisfied when the input region
is widened and when the adversarial search is given more effort, which is the opposite of
what a worst-case quantifier requires. If the compiled loss is inverted with respect to
the property, minimising it trains away from robustness, and negating the coefficient
would compensate.

Note what the sign means on the logic's stated semantics: minimising the constraint term
drives it toward `trueElement` (`-infinity`), so *maximising* it --- what this variant does
--- drives it toward `falseElement` (`+infinity`). On those semantics this variant trains
the network to violate the property. It is worth running only because we have evidence
that the stated semantics do not match what the compiled loss measures. Whichever sign
gives the better verification result tells us about the defect, not about which objective
is principled.

Launched 11:53 on 2026-08-28 with settings otherwise identical to run 1, including no
gradient clipping, so both runs start from the same checkpoint and differ only in the
sign. Outputs are in `traces-neg/` and `capucci-models-neg/`.

It ran all ten epochs without diverging:

| epoch | constraint | cross-entropy | blended | train acc | correct on 50 test | seconds |
| ----: | ---------: | ------------: | ------: | --------: | -----------------: | ------: |
| 0 (start) | -- | 0.0413 | -- | 99.5% | **38/50** | -- |
| 1 | -0.5501 | 0.3351 | +0.4641 | 87.9% | 29/50 | 303 |
| 2 | -0.9270 | 0.7927 | +0.8733 | 73.1% | 29/50 | 243 |
| 3 | -0.7877 | 0.8372 | +0.8075 | 72.4% | 27/50 | 240 |
| 4 | -0.4810 | 0.7543 | +0.5903 | 74.6% | 24/50 | 253 |
| 5 | -0.4662 | 0.7941 | +0.5974 | 72.2% | 23/50 | 219 |
| 6 | -0.4491 | 0.8121 | +0.5943 | 72.1% | 26/50 | 291 |
| 7 | -0.4971 | 0.8655 | +0.6445 | 70.7% | 26/50 | 215 |
| 8 | -0.5391 | 0.9134 | +0.6888 | 67.2% | 18/50 | 184 |
| 9 | -0.6222 | 0.9973 | +0.7722 | 63.6% | 25/50 | 288 |
| 10 | -0.7886 | 1.1711 | +0.9416 | 56.7% | 16/50 | 306 |

### What this shows

**The sign of the coefficient does not control where the constraint loss goes.** This
variant *maximises* the constraint term, which on the logic's semantics should drive it
toward `falseElement` (`+infinity`). It never became positive. It fell to $-0.93$, came
back to $-0.45$, and drifted down again to $-0.79$, wandering with no relation to the
direction being pushed. Compare the two runs at epoch 1, where the only difference is the
sign and the starting weights are identical:

| epoch 1 | constraint | cross-entropy | train acc | correct on 50 test |
| --- | ---: | ---: | ---: | ---: |
| positive sign | -0.3372 | 0.1531 | 95.1% | **38/50** |
| negated sign | **-0.5501** | 0.3351 | 87.9% | 29/50 |

Negating the coefficient made the constraint loss *more* negative than adding it did. The
reported value and the gradient being followed are therefore decoupled: the optimiser is
descending a gradient that does not control the quantity it is nominally the gradient of.
That is consistent with the quantifier defect --- the compiled loss depends on which points
the adversarial search selects, and that selection moves as the weights move, so the value
can drift independently of the step.

**It did not diverge, but that is stability rather than success.** Run 1 overflowed to
`nan` at epoch 3 by descending toward `-infinity`; this variant pushes away from that
singularity, so it survived ten epochs with finite values. All ten snapshots are usable.

**The network degraded steadily anyway.** Held-out accuracy fell from 38/50 to 16/50 and
training accuracy from 99.5% to 56.7%, while cross-entropy rose 28-fold. The held-out
column is noisy --- 18 at epoch 8, 25 at epoch 9, 16 at epoch 10 --- which is the same
plus-or-minus-three wobble that 50 images gave in the vanilla experiment, so individual
figures should not be read too closely. The trend is not in doubt.

**Every snapshot has a lower ceiling than the starting checkpoint.** Since a misclassified
image can never be proved robust, the best conceivable verification result from this run is
29/50 (epochs 1 and 2), against the vanilla baseline's *achieved* 22/50. The headroom that
made the experiment worth running has largely been spent on lost accuracy.

### Conclusion across both runs

Neither sign produced a network worth preferring to the starting checkpoint, and neither
moved the constraint loss in the direction the objective specified. Taken with the earlier
finding that the compiled loss reports a property as better satisfied when the input region
is widened, the reasonable reading is that this loss is not yet a usable training signal
in Vehicle 0.27.1 --- not that the Capucci logic, the blending weight, or the sign was
chosen wrongly. The most useful output of these two runs is evidence for the upstream bug
report.

## Verification of the negated-sign snapshots (in progress)

Launched 14:47 on 2026-08-28. Two models, each verified against **both** specifications,
at `epsilon 0.02` on the same 50 FashionMNIST test images.

| Model | training accuracy | correct on 50 test | ceiling |
| --- | ---: | ---: | ---: |
| `capucci_neg_e01.onnx` | 87.9% | 29/50 | 29/50 |
| `capucci_neg_e02.onnx` | 73.1% | 29/50 | 29/50 |

The ceiling is the same as the correct count, because an image the network already
misclassifies fails `advises` at zero perturbation and can never be proved robust. So 29
is the arithmetic best case for either model.

### Why both specifications

| Pass | Specification | Comparable with |
| --- | --- | --- |
| 1 | `fashionRobustness-capucci.vcl` (the one trained against) | the property that was actually optimised |
| 2 | `fashionRobustness-solution.vcl` (Exercise #7) | the vanilla baseline of 22/50 |

The two differ **only on ties**: Exercise #7 requires the advised label to score strictly
higher than every other, while the training specification permits equality. The strict
property is therefore the stronger one, and its verified count can only be **lower than or
equal to** the non-strict count.

That makes the comparison worth having in its own right. If the two counts come back
equal, ties do not arise in practice on this problem and the `CompareIndex` limitation
costs nothing; if the strict count is lower, the gap is precisely the price of the
workaround described above.

### How it is run

`verify_neg.py` runs pass 1 and `verify_neg_ex7.py` pass 2. The second waits for the
first to finish before starting, because a 50-image run needs about 14 GB and two
concurrent solvers would risk the machine. Both use Chapter 3 Exercise #7's command with
only the specification and `epsilon` varying, and both apply the same guards:

| Guard | Value | Purpose |
| --- | --- | --- |
| `--solver-args --timeout=` | 120 s per image | a single hard query cannot stall the run |
| wall clock | 7200 s per model | backstop if the per-image cap misbehaves |
| `RLIMIT_AS` | 24 GB | Marabou dies rather than swapping the machine |

Results append to `traces-neg/verify.csv`, tagged `training` or `exercise7`. Full solver
transcripts go to `marabou-outputs-neg/` and `marabou-outputs-neg-ex7/`.

### Results so far

<!-- RESULTS TABLE START -->

| Model | spec | correct | verified | falsified | of which misclassified | genuinely non-robust | robust share of eligible | solver |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `capucci_neg_e01` | training | 29/50 | **15/50** | 35/50 | 21 | 14 | 51.7% | 781 s |
| `capucci_neg_e02` | training | 29/50 | _not yet run_ | | | | | |
| `capucci_neg_e01` | Exercise #7 | 29/50 | _not yet run_ | | | | | |
| `capucci_neg_e02` | Exercise #7 | 29/50 | _not yet run_ | | | | | |

<!-- RESULTS TABLE END -->

**This table may be incomplete.** The verifications run unattended and outlive any one
working session; `traces-neg/verify.csv` is the source of truth. To fold whatever has
finished into the table above, run:

    python3 record_results.py

It recomputes the decomposition from the raw counts and rewrites the table in place, so it
is safe to run more than once.

Set beside the network this run started from:

| | correct | verified | robust share of eligible |
| --- | ---: | ---: | ---: |
| `vanilla_e100` (baseline, Exercise #7 spec) | 38/50 | **22/50** | 57.9% |
| `capucci_neg_e01` (training spec) | 29/50 | **15/50** | 51.7% |

**Property-driven training made the network worse at the property, on both measures.**
Fewer images are provably robust in absolute terms --- 15 against 22 --- and a smaller
proportion of the images the network classifies correctly are robust, 51.7% against 57.9%.
So the loss of nine test images was not compensated by better behaviour on those that
remain.

The comparison is moreover **generous to the trained model**, because it is measured under
the non-strict training specification while the baseline was measured under Exercise #7's
strict one. The strict property can only score lower or equal, so the Exercise #7 pass
should return 15 or fewer.

No image timed out or errored, so the counts are decisive rather than an artefact of the
solver giving up. The run took 781 s against the baseline's 924 s, which is what a
falsification-heavy run looks like: Marabou stops at the first counterexample, so finding
more of them is faster than proving their absence.

### A gap to be aware of

The vanilla baseline has **not** been measured under the training specification --- the
22/50 figure comes from Exercise #7's strict property. So pass 2's numbers can be compared
with the baseline and pass 1's cannot. Comparing pass 1's count against 22/50 would flatter
the trained models, since the non-strict property is the weaker of the two. Measuring
`vanilla_e100.onnx` under the training specification would close this, and is one further
verification run.

## Files

| File | Role |
| --- | --- |
| `vanilla_e100.onnx` | the starting point: epoch-100 vanilla classifier |
| `fashionRobustness-solution.vcl` | Chapter 3 Exercise #7's specification, used for **verification** |
| `fashionRobustness-capucci.vcl` | the same property plus the `qllAdditive` logic, intended for **training** --- does not currently compile to a loss, see above |
| `0-49Images.idx`, `0-49Labels.idx` | the 50 held-out test images Exercise #7 checks |
| `pdt-Capucci.py` | the experiment: continues `vanilla_e100.onnx` with the blended objective |
| `data/` | FashionMNIST, downloaded on demand (gitignored) |
| `pdt-Capucci-neg.py` | as above, with the constraint loss subtracted rather than added |
| `capucci-models/` | snapshots from run 1 (positive sign) |
| `capucci-models-neg/` | snapshots from run 2 (negated sign) |
| `README.md` | this file |

`.vclo` files are Vehicle's compiled caches and are gitignored.
