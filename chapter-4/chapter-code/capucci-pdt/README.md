# Capucci property-driven training experiment

Continuing the epoch-100 vanilla classifier with property-driven training, using the
Capucci (QLL) differentiable logic. Set up 2026-08-28.

**Status.** Runs 1 and 2 (2026-08-28, Vehicle 0.27.1) failed because of a sampler bug in
Vehicle; run 3 (2026-09-11, Vehicle 0.28.0) repeats run 1 with the bug fixed and gains
provable robustness on every snapshot, from 22/50 to 24--27/50, at no cost in accuracy.
Sections below are in chronological order; the earlier ones describe 0.27.1 behaviour.

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
| `capucci_neg_e02` | training | 29/50 | **8/50** | 42/50 | 21 | 21 | 27.6% | 637 s |
| `capucci_neg_e01` | Exercise #7 | 29/50 | **15/50** | 35/50 | 21 | 14 | 51.7% | 890 s |
| `capucci_neg_e02` | Exercise #7 | 29/50 | **8/50** | 42/50 | 21 | 21 | 27.6% | 766 s |

<!-- RESULTS TABLE END -->

**This table may be incomplete.** The verifications run unattended and outlive any one
working session; `traces-neg/verify.csv` is the source of truth. To fold whatever has
finished into the table above, run:

    python3 record_results.py

It recomputes the decomposition from the raw counts and rewrites the table in place, so it
is safe to run more than once.

Set beside the network this run started from:

| | correct | verified | genuinely non-robust | robust share of eligible |
| --- | ---: | ---: | ---: | ---: |
| `vanilla_e100` (baseline, Exercise #7 spec) | 38/50 | **22/50** | 16 | 57.9% |
| `capucci_neg_e01` (training spec) | 29/50 | **15/50** | 14 | 51.7% |
| `capucci_neg_e02` (training spec) | 29/50 | **8/50** | 21 | 27.6% |

**Property-driven training made the network worse at the property, on every measure.**
Against the starting network there are fewer provably robust images in absolute terms ---
15, then 8, against 22 --- and a smaller proportion of the images the network classifies
correctly are robust: 51.7% and 27.6% against 57.9%. The nine test images lost by epoch 1
were not paid for by better behaviour on those that remain.

**The two snapshots isolate the effect, and this is the sharpest result of the
experiment.** They have the *identical* ceiling of 29/50, so accuracy is held constant
between them --- yet provable robustness halved, from 15 to 8, and the number of genuinely
non-robust images rose from 14 to 21 out of the same 29 eligible. The second epoch of
training cost no accuracy at all and destroyed robustness anyway.

That cannot be explained as an accuracy-for-robustness trade-off, which is the usual way
such a decline would be read. With accuracy fixed, the constraint term is damaging the
very property it names, and doing so faster than it damages the classifier.

**The two specifications turn out to agree.** `capucci_neg_e01` verifies 15/50 under both
--- the non-strict property it was trained on and Exercise #7's strict one --- with the same
35 falsifications. So ties do not arise in practice on this problem: no image is decided by
the difference between `>` and `>=`.

That is worth recording for two reasons. It means the `CompareIndex` limitation costs
nothing here, so training against the non-strict formulation is a sound stand-in for the
published property rather than a compromise. And it means the comparison against the
baseline's 22/50 is **fair rather than generous**, which is how it had to be described
while only the non-strict figure was available.



No image timed out or errored in either run, so the counts are decisive rather than an
artefact of the solver giving up. Both were faster than the baseline's 924 s --- 781 s and
637 s --- which is what falsification-heavy runs look like: Marabou stops at the first
counterexample, so finding more of them is quicker than proving their absence. The
progressively shorter time is itself a symptom of the progressively worse network.

### Conclusions from the verification experiment

**Property-driven training lost provable robustness rather than gaining it.** The network
it started from proves 22 of the 50 images; after one epoch the figure is 15, after two it
is 8. Measured as a share of the images each network classifies correctly --- the only
images that could possibly be proved robust --- the decline is 57.9%, 51.7%, 27.6%.

**The loss is not a trade for accuracy.** This is the sharpest finding. The two snapshots
have the *identical* ceiling of 29/50, so accuracy is held fixed between them, and provable
robustness still halved from 15 to 8, with genuinely non-robust images rising from 14 to 21
out of the same 29 eligible. A decline of this kind is normally explained as accuracy being
sacrificed for robustness; here accuracy was not sacrificed and robustness fell anyway. The
constraint term is damaging the property it names, and doing so faster than it damages the
classifier.

**The two specifications agree exactly, so the non-strict workaround is sound.** Both
models return the same counts under the strict published property and under the non-strict
one they were trained against --- 15/50 with 35 falsifications for `e01`, 8/50 with 42 for
`e02`. Not a single image on this problem is decided by the difference between `>` and
`>=`, on either network.

Two things follow. The `CompareIndex` limitation costs nothing here, so training against
the non-strict formulation is a faithful stand-in for the published property rather than a
compromise --- which also vindicates the same workaround in `fmnist-robustness.vcl`. And the
comparison against the baseline's 22/50 is like-for-like, so the decline is real and not an
artefact of measuring two different properties.

The strict property is more work for the solver even though it returns the same answer:
890 s against 781 s for `e01`, 766 s against 637 s for `e02`. A query cannot be discharged
by finding an equality, so Marabou has to do more to reach the same verdict.

**Read together with the training runs, the conclusion is about the toolchain, not the
hyper-parameters.** Every configuration tried has failed, and each in a different way:

| Configuration | Outcome |
| --- | --- |
| constraint loss clamped at 0 | term contributes exactly `0.0000`; run reduces to cross-entropy |
| unclamped, added | diverges to `nan` at epoch 3 |
| unclamped, subtracted | runs, but the loss still falls; robustness collapses |

Together with the finding that the compiled loss reports a property as *better* satisfied
when the input region is widened, and that the coefficient's sign does not control which
direction the loss travels, the reasonable reading is that this loss is not yet a usable
training signal in Vehicle 0.27.1. The Capucci logic, the blending weight and the sign were
not chosen wrongly; the signal they are applied to does not track the property.

The useful output of this experiment is therefore evidence for the upstream bug report,
not a trained network. The snapshots are kept so that the verification can be repeated once
the quantifier defect is fixed, which would give a clean before-and-after on the same
models.

## The bug, and its fix in Vehicle 0.28.0

Vehicle 0.28.0 was released on 2026-09-10 and installed here on 2026-09-11, replacing
0.27.1. Diffing the two Python wheels locates the defect: it was in the PyTorch sampler,
not the compiler. In `DefaultPyTorchSampler` the FGSM step was

    perturbation = epsilon * torch.sign(gradient)      # 0.27.1
    perturbation = -epsilon * torch.sign(gradient)     # 0.28.0

so the adversarial search behind `forall perturbation` walked *down* the loss and returned
the least violating perturbation rather than the most violating one. That is the whole of
the quantifier defect: a wider ball gave the search more room to find a benign point, and
a harder search found a more benign one. Two smaller changes came with it: the search now
enables gradients even when the loss is called under `torch.no_grad()`, and random starting
points are drawn per pixel rather than as one scalar.

Re-running the epsilon sweep from the bug notes, with gradients enabled, on an untrained
network and four random images:

| epsilon | default | DL2 | qllAdditive |
| ------: | ------: | --: | ----------: |
| 0.0 | 0.62 | 0.62 | 1.20 |
| 0.02 | 0.90 | 0.90 | 1.24 |
| 0.5 | 4.57 | 4.57 | 1.71 |
| 2.0 | 6.07 | 6.07 | 1.94 |

The loss now increases with epsilon under all three logics, and DL2 no longer returns
`+inf`. Issues 3 and 4 of the bug notes (an `@parameter` inside a logic, the de Bruijn
crash) have not been re-tested.

**The divergence in run 1 was a consequence of the bug.** On `vanilla_e100.onnx` with real
training images, the corrected `qllAdditive` loss is exactly 0 for images the network
classifies with a comfortable margin and positive for the rest, growing with epsilon (four
images at epsilon 0.02: `0.0, 0.0, 2.37, 2.26`). The `j = label` term of the `>=`
conjunction pins the log-sum-exp at or above zero, so the term is bounded below and there
is no `-infinity` to descend toward. Under 0.27.1 the same loss was negative everywhere.
The original positive-sign script, unclamped, is therefore the right configuration, and
`pdt-Capucci-neg.py` is obsolete.

The corrected loss is slower: about 3 s for four images against roughly 20 s per 64-image
batch before, so an epoch takes about twice as long. With eight threads a 64-image call was
seen to thrash; the run-3 script pins PyTorch to four.

## Run 3: `pdt-Capucci-v028.py`, run 1 repeated under Vehicle 0.28.0 --- completed

Launched 16:38 on 2026-09-11; training finished 18:33, the last verification 20:35. Every
setting is identical to run 1 --- same checkpoint, specification,
logic, epsilon 0.02, alpha 0.4, batch 64, 1024 images, Adam at 1e-3, 10 epochs, no
clamping, no gradient clipping --- so that any difference is attributable to the fix. Only
the output locations differ: `traces-v028/` and `capucci-models-v028/`, leaving run 1's
folders intact as the evidence for the bug report.

`verify_v028.py` runs alongside the trainer and verifies each snapshot as it appears,
against Exercise #7's strict specification at epsilon 0.02 with the same guards as the
run-2 scripts. Only the strict specification is used, since run 2 showed the two forms
agree on every image here. Transcripts go to `marabou-outputs-v028/`.

<!-- V028 RESULTS TABLE START -->

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

<!-- V028 RESULTS TABLE END -->

`traces-v028/per_epoch.csv` and `traces-v028/verify.csv` are the sources of truth; the
table above is regenerated from them by:

    python3 record_results_v028.py

Training took just under two hours (16 steps of about 40 s per epoch); the ten
verifications took 3.7 h of solver time in total, between 12 and 42 minutes each. No image
errored; one query timed out (image 6 of the epoch-2 model, which every other snapshot
proves).

### Per-image verdicts

Images 0--49 of the test set, one row per snapshot. `T` proved robust, `F` falsified,
`?` timed out. Images the snapshot misclassifies are falsified at zero perturbation and
appear as `F`.

    e01: FTTTTTTFFTFFFTFTTFTTTFTFTFFFTFTTTTTFTFTTFTFFFFTTFF
    e02: FFTTTT?FFTFFFTTTTFFTFFTFTFFFTFTTTTTFTFTTFTFFTFFTFF
    e03: TFTTTTTFFTFFFTFTFFTTFFTFTFFFTFTTTTTFTTTTFTFFTFFTFF
    e04: FFTTTTTTFTFFFTFTFFFTFFTFTFFFTFTTTTTFTFTTFTFFTFFTFF
    e05: FFTTTTTTFTFFFTFTFFFTFFTFTFFTTFTTTTTFTFTTFTFFTFFTFF
    e06: TFTTTTTTFTFFFTFTFFFFFFTFTFFFTFTTTTTFTTTTFTFFTFFTFF
    e07: FFTTTTTTFTFFFTFTFFFTFFTFTFFFTFTTTTTFTTTTFTFFTFFTFF
    e08: FTTTTTTTFTFFFTFTFFTFFFTFTFFFTFTTTTTFTTTTFTFFFFTTFF
    e09: FTTTTTTFFTFFFTFTTFFTFFTFTFFFTFTTTTTFTTTTFTFFTFTTFF
    e10: TFTTTTTTFTTFFTFTTFFTFFTFTFFFTFTTTTTFTTTTFTFFFFFTFF

| | images |
| --- | ---: |
| proved robust by all ten snapshots | 20 |
| proved by at least one snapshot | 34 |
| never proved by any snapshot | 16 |
| flip between snapshots | 14 |

The 16 never proved are the 10 to 13 images each snapshot misclassifies together with a
handful that are correctly classified but breakable in every snapshot. The 14 that flip
are the margin: images whose robustness at `epsilon 0.02` is decided by where exactly the
optimiser happens to be at the end of the epoch.

### Conclusions from run 3

**Property-driven training gained provable robustness, on every snapshot.** The starting
network proves 22 of the 50 test images. All ten snapshots prove between 24 and 27, a mean
of 25.6, and the final snapshot proves 27. Measured as a share of the images each network
classifies correctly, the only images that could be proved, the baseline's 57.9% became
61.5% to 73.0%.

**It cost no accuracy.** The ceiling never fell below 37/50 and finished at 40/50, two above
the starting network; cross-entropy stayed between 0.052 and 0.063 throughout (the start is
0.041); training accuracy stayed between 98.5% and 99.3%. The final snapshot is better than
the starting network on both axes at once: 40 correct against 38, and 27 provably robust
against 22. That is the trade run 2 could not make, where accuracy was held fixed between
two snapshots and robustness halved anyway.

**The fix is the whole explanation.** Same checkpoint, same specification, same logic, same
hyper-parameters, same absence of clamping and gradient clipping; the only difference from
run 1 is Vehicle 0.28.0 instead of 0.27.1. Run 1 diverged to `nan` at epoch 3; run 3
trained for ten epochs with a loss that fell monotonically apart from one uptick at epoch 6
and never approached zero from below. The constraint loss went from 0.54 to 0.29.

**Training loss and verified count decouple after the first epoch.** The loss halved over
epochs 1 to 10 while the verified count went 27, 24, 26, 24, 25, 25, 25, 26, 27, 27 --- a
band of plus or minus two around a flat line, with epoch 1 already at the top of it. Two
reasons are plausible and both probably contribute. The loss is measured on the 1024
training images and the verification on 50 held-out test images, so part of the later
improvement is fitting rather than generalising. And the loss sees only what the FGSM search
finds, which is a lower bound on the worst case, so the loss can approach zero on an image
that Marabou can still break. Longer training at this learning rate would be expected to
keep lowering the loss without moving the verified count much.

**The 14 flipping images are why a single snapshot is a noisy measurement.** Consecutive
epochs differ by up to three verified images without any change in what was being
optimised, so a one-snapshot comparison has an error bar of about plus or minus two on this
50-image set. The evidence that training helped is that all ten snapshots sit above the
baseline, not that any one of them does.

**What this means for the earlier conclusions.** The reading of runs 1 and 2 stands as a
description of Vehicle 0.27.1: the compiled loss did not track the property because the
sampler searched in the wrong direction, and no choice of sign, weight or clamping could
fix that. Under 0.28.0 the same loss is a usable training signal. The Capucci logic, alpha
0.4, epsilon 0.02 and the unclamped positive-sign objective are all vindicated as they were
originally chosen. The bug report can now cite a before-and-after on identical models.

## Files

| File | Role |
| --- | --- |
| `vanilla_e100.onnx` | the starting point: epoch-100 vanilla classifier |
| `fashionRobustness-solution.vcl` | Chapter 3 Exercise #7's specification, used for **verification** |
| `fashionRobustness-capucci.vcl` | the same property (non-strict `advises`) plus the `qllAdditive` logic, used for **training** |
| `0-49Images.idx`, `0-49Labels.idx` | the 50 held-out test images Exercise #7 checks |
| `pdt-Capucci.py` | the experiment: continues `vanilla_e100.onnx` with the blended objective |
| `data/` | FashionMNIST, downloaded on demand (gitignored) |
| `pdt-Capucci-neg.py` | as above, with the constraint loss subtracted rather than added (run 2; obsolete since 0.28.0, kept for the record) |
| `pdt-Capucci-v028.py` | run 3: `pdt-Capucci.py` unchanged in settings, writing to the `-v028` folders, under Vehicle 0.28.0 |
| `verify_neg.py`, `verify_neg_ex7.py` | run 2's verification against the training and Exercise #7 specifications |
| `verify_v028.py` | run 3's verification watcher: checks each snapshot against Exercise #7 as it appears |
| `record_results.py`, `record_results_v028.py` | regenerate the run 2 and run 3 results tables in this file |
| `capucci-models/` | snapshots from run 1 (positive sign, Vehicle 0.27.1) |
| `capucci-models-neg/` | snapshots from run 2 (negated sign, Vehicle 0.27.1) |
| `capucci-models-v028/` | snapshots from run 3 (Vehicle 0.28.0) |
| `marabou-outputs-neg/`, `marabou-outputs-neg-ex7/`, `marabou-outputs-v028/` | Marabou transcripts for runs 2 and 3 |
| `README.md` | this file |

`.vclo` files are Vehicle's compiled caches and are gitignored.
