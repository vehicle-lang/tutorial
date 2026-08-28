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

## Run 2: `pdt-Capucci-neg.py`, negated sign --- in progress

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
sign. Since the negated objective maximises a quantity that is unbounded *above*,
divergence at least as fast as run 1's is expected.

Outputs go to `traces-neg/` and `capucci-models-neg/` so this run cannot disturb run 1's.

**No epoch results yet at the time of writing.**

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
