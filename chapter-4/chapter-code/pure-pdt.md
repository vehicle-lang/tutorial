# Property-driven training with no task loss at all

A single experiment, run to answer one question: if the constraint loss compiled
from a Vehicle specification is what makes a network verifiable, what happens if we
train on *only* that, with the task loss switched off entirely?

The short answer is that it makes the network worse at both jobs — including the one
it was exclusively optimising. That is a more interesting result than the expected
accuracy-versus-robustness trade-off, and it is the empirical case for the blended
objective that the rest of the chapter builds.

## What was run

The starting point is the epoch-200 network from the vanilla experiment described in
[README.md](README.md): 100% training accuracy, cross-entropy 0.000281, and provable
robustness that had plateaued at 29/50. Its weights were reloaded exactly from
`model_e200.onnx` (checked against onnxruntime, agreeing to 7e-7), so this run
continues that network rather than starting afresh.

For the next ten epochs the only gradient signal is the constraint loss compiled from
[fmnist-robustness.vcl](fmnist-robustness.vcl):

```python
con = torch.stack(constraint_loss_fn(
          n=BATCH_SIZE, classifier=network, epsilon=torch.tensor(EPSILON),
          trainingImages=images.squeeze(1), trainingLabels=labels)).mean()
con.backward()                      # the ONLY gradient signal
optimizer.step()
with torch.no_grad():               # reported, never optimised
    logits = model(images)
    es += cross_entropy(logits, labels).item()
```

Cross-entropy is still computed every epoch, but purely for monitoring: it never
reaches `.backward()`. 1024 FashionMNIST training images, batch size 64 (16 steps per
epoch), epsilon 0.005, Adam at lr 1e-3.

## Training statistics

| Epoch | Constraint loss | Cross-entropy (monitored) | Train accuracy | Seconds |
|------:|----------------:|--------------------------:|---------------:|--------:|
| _start_ (vanilla e200) | – | 0.000281 | 100.00% | – |
| 1 | 0.6221 | 0.1373 | 94.92% | 183 |
| 2 | 0.9240 | 0.6482 | 79.10% | 236 |
| 3 | 0.9970 | 0.8944 | 72.17% | 202 |
| 4 | 1.0268 | 0.9257 | 70.61% | 215 |
| 5 | 0.8378 | 1.2471 | 62.79% | 230 |
| 6 | 0.9778 | 1.5360 | 58.69% | 250 |
| 7 | 1.0556 | 1.6809 | 58.50% | 220 |
| 8 | 1.2438 | 2.1979 | 54.69% | 228 |
| 9 | 1.5383 | 2.7288 | 50.10% | 221 |
| 10 | 1.6688 | 2.8094 | 50.98% | 273 |

Total training time 2258 s (37.6 min).

## Verification

Each epoch's weights were exported as `model_pdt_eNN.onnx` and verified with exactly
the command from Chapter 3, Exercise #7 — same specification, same 50 images, same
epsilon — so the numbers are directly comparable with that exercise and with the
vanilla baseline:

```bash
vehicle verify \
  --specification ../../chapter-3/exercises/FMNIST/fashionRobustness-solution.vcl \
  --network classifier:model_pdt_e01.onnx \
  --parameter epsilon:0.005 \
  --dataset trainingImages:../../chapter-3/exercises/FMNIST/idxdata/0-49Images.idx \
  --dataset trainingLabels:../../chapter-3/exercises/FMNIST/idxdata/0-49Labels.idx \
  --solver Marabou
```

| Model | Verified | Falsified | Solver time |
|-------|---------:|----------:|------------:|
| vanilla, epoch 200 (baseline) | **29/50** | 21/50 | 14 min 35 s |
| +1 epoch, constraint loss only | **14/50** | 36/50 | 39 min 50 s |
| +5 epochs, constraint loss only | _pending_ | _pending_ | – |
| +10 epochs, constraint loss only | _pending_ | _pending_ | – |

Full transcripts are in [marabou-outputs/](marabou-outputs/), one file per model.

## What the numbers say

**The constraint loss gets worse, not better.** It rises from 0.6221 after one epoch
to 1.6688 after ten, a factor of 2.7, with the epoch-5 dip (0.8378) the only
departure from an otherwise monotonic climb. Ten epochs of optimising this objective
and nothing else left the network further from satisfying it than when it started.

**Provable robustness halves immediately.** 29/50 down to 14/50 after a single epoch,
at which point training accuracy was still 94.9%. So this is not the accuracy
collapse dragging robustness down with it — the very first step away from the
cross-entropy optimum already costs fifteen provable images.

**Accuracy collapses too**, 100% to 50.98% on a ten-class problem, and monitored
cross-entropy rises to 2.8094. This part is unsurprising: nothing was optimising it.
It is the price of the experiment rather than its finding.

The finding is the first two points together. A blended objective is not merely a
nicety for preserving accuracy while robustness is pursued. Without the task loss to
hold the weights in a region where the classification is meaningful, the constraint
loss alone does not even minimise itself. This is the experimental case for the
`alpha` term that blends the two losses.

## Caveats that bear on interpretation

These matter, and anyone citing the numbers above should know them.

**The training and verification pipelines disagree about input scaling.** Training
normalises (`transforms.Normalize((0.2860,), (0.3530,))`), so the network sees pixels
in roughly [-0.810, 2.023]. Verification reads the `.idx` files, whose pixels are
float64 in [0.0, 1.0]. The network is therefore verified on inputs scaled differently
from those it was trained on. This is pre-existing — the vanilla baseline of 29/50 was
measured through the same mismatch — so the *relative* comparison here is sound, but
the absolute counts should not be read as this architecture's robustness ceiling.

**Consequently the two epsilons are not the same ball.** Epsilon 0.005 applied in
normalised units during training corresponds to 0.00176 in raw pixel units, while
verification checks 0.005 in raw units — about 2.8x larger. Training was enforcing a
noticeably tighter property than verification tested, which is its own reason not to
expect the training to pay off at verification time.

**The training spec's validity precondition is false in normalised space.**
`validImage x = forall i j . 0 <= x ! i ! j <= 1` cannot hold of normalised pixels.
Under a differentiable logic the implication in `robustAround` is not discharged as
vacuously true the way it would be in the Boolean reading; it contributes a soft value
driven by an antecedent that is substantially false. How much this distorts the
gradient has not been measured here, but it is a plausible mechanism for a constraint
loss that fails to descend.

**The run was stopped at ten epochs of a planned twenty**, once the direction was
clear. The numbers show degradation over ten epochs; they are not evidence that a much
longer run could never recover.

**One run, one seed.** No seed was fixed and the experiment was not repeated, so the
epoch-by-epoch figures carry unmeasured run-to-run variation. The one partial replicate
available — an aborted first attempt, preserved as
`per_epoch.aborted-50ep-run.csv` — agreed closely at epoch 1 (constraint 0.6357 versus
0.6221, accuracy 95.5% versus 94.9%), which is mildly reassuring but is not a
substitute for repetition.

Taken together, the first three caveats mean this experiment establishes that *this
pipeline*, trained this way, degrades on both objectives. Establishing the stronger and
more useful claim — that constraint-only training is intrinsically self-defeating —
would need the scaling mismatch fixed first, so that training and verification are
demonstrably talking about the same property.

## Reproducing it

The scripts live outside the repository, in `ch4-pdt-run/`:

- `continue_with_pdt.py` — reloads `model_e200.onnx`, trains ten epochs on the
  constraint loss alone, exports `model_pdt_eNN.onnx` after every epoch.
- `verify_each_epoch.py [epochs...]` — runs Exercise #7's command against each
  snapshot. With no arguments it verifies all ten; passing epoch numbers verifies a
  sample, which is usually what you want: one verification takes 15-40 minutes against
  roughly 4 minutes for an epoch of training.

Expect Marabou to reach about 14 GB of resident memory on a 50-image run, and to hold
it for the duration. That is normal for this problem size and it is released cleanly.

## Files

- `traces/per_epoch.csv`, `traces/per_epoch_table.md` — the training statistics above
- `traces/verify_per_epoch.csv` — the verification counts
- `traces/marabou-outputs/verify_pdt_eNN.txt` — full solver transcripts
- `traces/model_pdt_eNN.onnx` — all ten snapshots, so any epoch can be verified later
  without retraining
- `traces/train_log.txt`, `traces/verify_log.txt` — run logs
