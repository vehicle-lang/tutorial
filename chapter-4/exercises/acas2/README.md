# acas2: network N_{1,8} trained on the ten ACAS Xu properties, one per epoch

Second round of ACAS Xu property-driven training, set up 2026-09-17. The first round
(the former `acas` folder, summarised in the appendix at the end) trained three networks on
Chapter 2's Property 3 alone. This round uses the
**full upstream specification** with all ten Reluplex properties,
[vehicle-lang/vehicle, examples/acasXu/acasXu.vcl (dev)](https://github.com/vehicle-lang/vehicle/blob/dev/examples/acasXu/acasXu.vcl),
saved here as `acasXu-upstream.vcl`, and a single network, N_{1,8}.

The experiment, as specified:

1. verify N_{1,8} on the upstream specification, all ten properties;
2. split the specification into ten one-property training specifications, and check that
   each compiles independently;
3. train N_{1,8} for ten epochs, one specification per epoch: epoch k trains on property k
   only, starting from the network saved at the end of epoch k−1, and saves a snapshot;
4. verify the result on the original ten-property specification.

All results, training losses and Marabou verdicts, are in tables below.

> **Status (2026-09-17, 21:00).** Training is complete. All verifications are complete except the
> four long 9000 s re-runs of Properties 6, 7, 8 on the baseline and Property 6 on `e30`
> (`traces/verify_long_*.csv`), running at the time of writing. `python3 record-results.py`
> refreshes the tables in this file from the CSVs, `python3 record-results.py --baselines`
> prints the three-network table used in Exercise #6, and the table under "Re-running the
> undecided properties" still has to be filled in by hand from `traces/verify_long_*.csv`.
> This folder is then to be copied over `../../solutions/acas2/` and pruned to the starting
> material (see the chapter-4 README).

## Contents

| File | Role |
| --- | --- |
| `acasXu-upstream.vcl` | the upstream specification, ten properties, unchanged. Used for **verification**. |
| `acasXu.vcl`, `acasXu-training.vcl`, `acasXu-training-p2.vcl`, `acasXu_1_7.onnx`, `acasXu_1_9.onnx` | the first round's code and inputs, kept for the record (see the appendix); not used in this round's runs |
| `acasXu_1_8.onnx` | the network, byte-identical to Chapter 2's (md5 `3ecac92e…`) |
| `split-spec.py` | step 2: writes `specs/property01.vcl` … `specs/property10.vcl` |
| `capucci-logic.vcl` | the chapter's Capucci `qllAdditive` logic block, copied from `chapter-code/capucci-pdt/fashionRobustness-capucci.vcl` so that this folder is self-contained; `split-spec.py` appends it to each training spec |
| `specs/` | the ten one-property **training** specifications |
| `pdt-acas2.py` | step 3: the sequential training script |
| `verify-acas2.py` | steps 1 and 4: one `vehicle verify` per (network file, property), with a time limit |
| `specs-verify/` | the ten properties as one-property **verification** specifications: the upstream text unchanged, restricted to one property each (`split-spec.py --for-verification`); used for the long re-runs of Properties 6, 7, 8 |
| `record-results.py` | regenerates the combined Marabou table in this README from `traces/verify_*.csv` |
| `models/` | `acasXu_1_8_seq_eKK.onnx` (the ten-epoch tour), `*_cyc3_*` and `*_cyc3clip_*` (the two diverged 3-cycle runs), `*_cyc3reset_*` (the 3-cycle run that completed); `e00` is always the start, re-exported |
| `traces/` | `1_8_seq_per_epoch.csv`, `1_8_seq_steps.csv`, `1_8_seq_loss_matrix.csv`, `1_8_seq_train_log.txt`, `verify.csv`, verification logs |
| `marabou-outputs/` | one transcript per (network file, property) |
| `marabou-outputs-step0/` | the aborted first attempt at step 1 (see below) |

Software versions: Vehicle 0.28.0 (`vehicle --version`; `vehicle_lang` reports the same), Marabou as installed for Chapter 2, PyTorch 2.13.0 (CPU), Python 3.11, onnx and onnxruntime (the latter optional, used only to check the weight reload). Run on a 12-core CPU machine with PyTorch pinned to 4 threads.

## How the upstream specification differs from Chapter 2's

`diff acasXu.vcl acasXu-upstream.vcl`, ignoring renamed types and comments:

| | Chapter 2 (`acasXu.vcl`) | upstream (`acasXu-upstream.vcl`) |
| --- | --- | --- |
| properties | 3 only | 1 to 10 |
| the "network chooses i" predicate | `advises` (minimal score) | `minimalScore`, plus `maximalScore` for Property 2 |
| minimum ownship speed in `validInput` | 0 | 100 |
| Property 3 | identical apart from the predicate's name | |

Properties and the networks the Reluplex paper tested them on, from the comments in the file:

| Property | Says | Tested on |
| --- | --- | --- |
| 1 | intruder distant and slower ⇒ COC score ≤ a threshold | all 45 |
| 2 | intruder distant and slower ⇒ COC not maximal | N_{x,y}, x ≥ 2 |
| 3 | directly ahead, moving towards ⇒ COC not minimal | all except N_{1,7}, N_{1,8}, N_{1,9} |
| 4 | directly ahead, moving away, slower ⇒ COC not minimal | all except N_{1,7}, N_{1,8}, N_{1,9} |
| 5 | near, approaching from left ⇒ strong right | N_{1,1} |
| 6 | sufficiently far away ⇒ COC | N_{1,1} |
| 7 | large vertical separation ⇒ never strong left or strong right | N_{1,9} |
| 8 | large vertical separation, previous weak left ⇒ COC or weak left | N_{2,9} |
| 9 | previous weak right, nearby intruder ⇒ strong left | N_{3,3} |
| 10 | far away intruder ⇒ COC | N_{4,5} |

So for N_{1,8} only Property 1 is one the network was designed to satisfy. Properties 2 to 4
are known not to hold on it, and 5 to 10 were written for other networks with different
previous-advisory contexts, so there is no expectation that they hold. Training N_{1,8} on
all ten is therefore asking it to satisfy nine properties it was never meant to.

## Step 1: verifying N_{1,8} on all ten properties

**First attempt, aborted.** A single `vehicle verify` with no `--property` flag verifies all
properties in one call but prints nothing until the end. Run on the three networks, it had
completed properties 1 to 4 of N_{1,7} and was in the middle of Property 5 after ten
minutes; it was stopped when the experiment was narrowed to N_{1,8}, and its partial
transcript is in `marabou-outputs-step0/`. Lesson: verify one property per call, so that
each gets its own verdict and time, and put a time limit on each.

**The run.** `verify-acas2.py acasXu_1_8.onnx --timeout 900` runs, for k = 1 … 10,

```bash
vehicle verify --specification acasXu-upstream.vcl --solver Marabou \
  --network acasXu:acasXu_1_8.onnx --property propertyK
```

with a 15-minute limit per property.

Results (the same table is repeated under Step 4 with the trained network's columns):

| Property | N_{1,8} before training (step 1) | e10, after the ten-epoch tour (step 4) | snapshot e_k on its own property k |
|---:|---|---|---|
| 1 | ✓ verified (4 s) | ✓ verified (3 s) | e01: ✓ verified (6 s) |
| 2 | ✓ verified (302 s) | ✓ verified (20 s) | e02: ✓ verified (22 s) |
| 3 | ✗ falsified (2 s)<br>x = [1800, 0.06, 3.1, 980, 960] | ✓ verified (1 s) | e03: ✗ falsified (1 s)<br>x = [1800, -0.06, 3.1, 980, 960] |
| 4 | ✗ falsified (1 s)<br>x = [1800, 0.06, 0, 1000, 772] | ✗ falsified (2 s)<br>x = [1800, -0.06, 0, 1000, 769.7] | e04: ✓ verified (1 s) |
| 5 | ✗ falsified (1 s)<br>x = [400, 0.2, -3.137, 100, 44.44] | ✗ falsified (1 s)<br>x = [400, 0.2, -3.142, 100, 143.3] | e05: ✓ verified (2 s) |
| 6 | timeout (900 s) | ✗ falsified (106 s)<br>x = [3.205e+04, -3.142, -3.142, 606.2, 63.46] | e06: ✗ falsified (127 s)<br>x = [5.242e+04, -1.007, -3.137, 222.2, 246.6] |
| 7 | timeout (1588 s) | timeout (900 s) | e07: timeout (900 s) |
| 8 | timeout (1284 s) | timeout (900 s) | e08: timeout (900 s) |
| 9 | ✗ falsified (1 s)<br>x = [6940, -0.2133, -3.142, 109.2, 58.41] | ✗ falsified (2 s)<br>x = [7000, -0.14, -3.142, 125.2, 1.314] | e09: ✗ falsified (2 s)<br>x = [7000, -0.1438, -3.142, 103.2, 1.535] |
| 10 | ✓ verified (133 s) | ✓ verified (1113 s) | (e10, previous column) |

Reading the baseline column: Property 1 holds, as the Reluplex paper says it does on all
45 networks. Property 2 holds too, although it was only tested upstream on N_{x,y} with
x ≥ 2. Properties 3, 4, 5 are falsified in about a second each. Properties 6, 7 and 8 are
the whole-input-space properties and Marabou does not finish them inside the limit
(Properties 7 and 8 were only killed after 1588 s and 1284 s: the verifier's timeout was
not enforced promptly, see the note under Step 4). Property 9 is falsified in a second.
Property 10 holds (133 s), although upstream it was only tested on N_{4,5}. So before any
training N_{1,8} satisfies Properties 1, 2 and 10, violates 3, 4, 5 and 9, and 6, 7, 8 are
undecided within the limit.


## Step 2: splitting the specification into ten

`split-spec.py --p 5` writes `specs/property01.vcl` … `specs/property10.vcl`. Each file is:

- the upstream preamble (types, network, normalisation, `minimalScore`, `maximalScore`);
- the helper predicates of **all ten** property sections (`intruderDistantAndSlower`,
  `directlyAhead`, `movingTowards`, …), because Property 2 reuses Property 1's predicate
  and Property 4 reuses Property 3's; a first version that kept only the section's own
  helpers failed to compile `property02.vcl` with "`intruderDistantAndSlower` is not in
  scope";
- exactly **one** `@property` block, property k;
- the Capucci `qllAdditive` logic (`capucci-logic.vcl`, a local copy of the block at the end of
  `chapter-code/capucci-pdt/fashionRobustness-capucci.vcl`) with `p = 5`.

Two changes are made to the preamble, both needed for the loss compiler and both explained
in the appendix ("the guarded implication compiles to the constant zero"): the
`i != j =>` guard is dropped from `minimalScore` and `maximalScore` and the comparisons
become non-strict,

```vehicle
minimalScore i x = forall j . normAcasXu x ! i <= normAcasXu x ! j
maximalScore i x = forall j . normAcasXu x ! i >= normAcasXu x ! j
```

Verification (steps 1 and 4) uses the unchanged upstream file, so the verdicts are about the
original strict, guarded properties.

**Each file compiles independently.** Every spec was type-checked on its own
(`vehicle typecheck`, exit 0 for all ten) and compiled to a loss on its own, one
`vehicle --json compile loss --logic qllAdditive` process per file, then evaluated on the
starting network. Capucci value (smaller is truer, unbounded both ways; see the appendix for
why the values are negative) and gradient norm:

| Property | loss on N_{1,8} at start | gradient norm | compile time |
| ---: | ---: | ---: | ---: |
| 1 | −4.0113 | 1.000 | 3.7 s |
| 2 | −0.3225 | 0.895 | 3.3 s |
| 3 | −0.3203 | 0.898 | 3.8 s |
| 4 | −0.3198 | 0.899 | 3.3 s |
| 5 | +0.3228 | 0.899 | 3.3 s |
| 6 | +0.4599 | 0.894 | 3.7 s |
| 7 | −0.1834 | 0.548 | 3.4 s |
| 8 | +0.1826 | 0.550 | 3.1 s |
| 9 | +0.3153 | 1.024 | 3.2 s |
| 10 | +0.3213 | 0.894 | 3.1 s |

Property 1 is a plain threshold on one output and starts far in the true direction; the
`minimalScore`-shaped properties all start near ±`log(5)/p` = ±0.32, the Capucci soft-max
offset, positive where the network's minimal advisory is *not* the required one (5, 9, 10)
or where the required advisory is COC in a region where N_{1,8} gives it anyway (6 is the
exception, at +0.46: a disjunction in its antecedent). Property 7 and 8 have two
`minimalScore` terms each and a different offset.

Note that the sign of the starting loss is not the verdict; the soft-max offset moves the
"true" threshold away from zero. Marabou decides.

## Step 3: training, one property per epoch

`pdt-acas2.py 1_8` follows `pdt-acas.py`, the first round's script (itself
`chapter-code/capucci-pdt/pdt-Capucci-v028.py` adapted to ACAS Xu) with one structural
change: the loss function changes every epoch.

| Setting | Value |
| --- | --- |
| Starting network | `acasXu_1_8.onnx`, reloaded into PyTorch (agrees with onnxruntime to 5.7e-07) |
| Epoch k | 16 Adam steps on `specs/propertyKK.vcl`'s compiled loss, from the epoch k−1 weights |
| Task loss | none (`ALPHA = 0`): no ACAS Xu data |
| Logic | Capucci `qllAdditive`, `p = 5` |
| Adversarial search | Vehicle default: 10 random starts, 5 PGD steps, over each property's own region |
| Optimiser | Adam, lr 1e-3, one optimiser kept across the ten epochs; no clipping |
| Snapshots | `models/acasXu_1_8_seq_eKK.onnx` after every epoch; `e00` is the start |
| Monitoring | after every epoch, all ten losses are evaluated on the current network (`traces/1_8_seq_loss_matrix.csv`), plus agreement of advisories with the original network on 20,000 uniform inputs |

```bash
python3 split-spec.py --p 5
python3 pdt-acas2.py 1_8 --check       # compile the ten specs, print the table above
python3 pdt-acas2.py 1_8               # the ten epochs, about 5 minutes
```

The run took about three minutes (epoch times below; the first epoch's loss, a single
threshold comparison, is much cheaper to evaluate than the `minimalScore` ones).

### Per epoch: the property trained on, its loss, and drift from the original network

The loss is the Capucci value of the property being trained, mean and last over the 16
steps. "Agreement" is the fraction of 20,000 inputs drawn uniformly from the whole valid
input space on which the network gives the same advisory as the original N_{1,8}.

| Epoch | Trained on | loss mean | loss last | agreement with original | seconds | snapshot |
|---:|---|---:|---:|---:|---:|---|
| 0 | start |  |  | 100.00% |  | `acasXu_1_8_seq_e00.onnx` |
| 1 | property1 | -4.0188 | -4.0263 | 99.60% | 1 | `acasXu_1_8_seq_e01.onnx` |
| 2 | property2 | -0.3468 | -0.3594 | 99.60% | 7 | `acasXu_1_8_seq_e02.onnx` |
| 3 | property3 | -0.2823 | -0.2857 | 99.60% | 8 | `acasXu_1_8_seq_e03.onnx` |
| 4 | property4 | -0.3536 | -0.5390 | 52.59% | 6 | `acasXu_1_8_seq_e04.onnx` |
| 5 | property5 | +0.2097 | +0.0217 | 0.00% | 7 | `acasXu_1_8_seq_e05.onnx` |
| 6 | property6 | +0.8744 | +0.4452 | 0.00% | 14 | `acasXu_1_8_seq_e06.onnx` |
| 7 | property7 | -0.1848 | -0.1913 | 89.53% | 15 | `acasXu_1_8_seq_e07.onnx` |
| 8 | property8 | +0.1659 | +0.1598 | 91.69% | 20 | `acasXu_1_8_seq_e08.onnx` |
| 9 | property9 | +0.3650 | +0.3470 | 93.78% | 7 | `acasXu_1_8_seq_e09.onnx` |
| 10 | property10 | +0.2803 | +0.2737 | 95.31% | 8 | `acasXu_1_8_seq_e10.onnx` |

### All ten property losses after every epoch

Row = the network saved at the end of that epoch, column = property (from
`traces/1_8_seq_loss_matrix.csv`); smaller is truer; the property trained in that epoch is in
bold. This is the table that shows what training on one property does to the others.

| After epoch | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 | p10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (start) | -4.011 | -0.322 | -0.320 | -0.320 | +0.323 | +0.460 | -0.183 | +0.183 | +0.315 | +0.321 |
| 1 (p1) | **-4.027** | -0.335 | -0.308 | -0.307 | +0.326 | +0.447 | -0.187 | +0.178 | +0.321 | +0.309 |
| 2 (p2) | -4.042 | **-0.361** | -0.284 | -0.283 | +0.333 | +0.423 | -0.194 | +0.168 | +0.333 | +0.284 |
| 3 (p3) | -4.042 | -0.359 | **-0.287** | -0.287 | +0.332 | +0.425 | -0.193 | +0.169 | +0.330 | +0.287 |
| 4 (p4) | -4.030 | -0.318 | -0.557 | **-0.602** | +0.323 | +0.490 | -0.187 | +0.178 | +0.325 | +0.354 |
| 5 (p5) | -3.783 | -0.055 | -1.865 | -2.126 | **+0.011** | +1.608 | -0.050 | +0.357 | +0.775 | +1.564 |
| 6 (p6) | -4.031 | -0.339 | -0.662 | -0.606 | +0.066 | **+0.444** | -0.180 | +0.181 | +0.472 | +0.305 |
| 7 (p7) | -4.040 | -0.352 | -0.415 | -0.363 | +0.162 | +0.432 | **-0.192** | +0.172 | +0.420 | +0.294 |
| 8 (p8) | -4.047 | -0.361 | -0.376 | -0.328 | +0.189 | +0.424 | -0.206 | **+0.159** | +0.427 | +0.286 |
| 9 (p9) | -4.047 | -0.362 | -0.352 | -0.314 | +0.299 | +0.423 | -0.200 | +0.158 | **+0.345** | +0.284 |
| 10 (p10) | -4.055 | -0.375 | -0.329 | -0.284 | +0.339 | +0.411 | -0.197 | +0.156 | +0.332 | **+0.273** |

### What the snapshots advise

| Snapshot | advisories on 4,000 uniform inputs, whole valid space | output range |
|---|---|---|
| e00 | COC 99.8%, weakLeft 0.1%, weakRight 0.1% | [−0.021, 0.011] |
| e01 | COC 100% | [−0.037, 0.011] |
| e02 | COC 100% | [−0.052, 0.027] |
| e03 | COC 100% | [−0.051, 0.005] |
| e04 | COC 52.5%, weakRight 47.4% | [−0.247, 0.174] |
| e05 | strongRight 100% | [−1.987, 1.426] |
| e06 | strongRight 100% | [−0.573, 0.197] |
| e07 | COC 90.1%, strongRight 9.9% | [−0.301, 0.093] |
| e08 | COC 92.0%, strongRight 7.9%, weakLeft 0.1% | [−0.260, 0.112] |
| e09 | COC 94.2%, strongRight 5.6%, weakLeft 0.1% | [−0.233, 0.073] |
| e10 | COC 96.0%, strongRight 4.0%, weakLeft 0.1% | [−0.214, 0.090] |

### Reading the training tables

- **Epochs 1 to 3 barely move anything.** The trained property's own loss changes by a few
  hundredths, the other nine by less, and the network stays at 99.6% agreement with the
  original. Sixteen steps at the start of an Adam run are small steps; in the first round, sixteen
  steps on Property 3 alone also only moved its loss from −0.320 to −0.346, and yet that was
  enough for Marabou to verify it. The verification below says whether it was enough here.
- **Epochs 4 to 6 are the violent part.** Property 4 splits the input space between COC and
  weakRight. Property 5, which demands strongRight in its region, turns the whole network
  into a constant strongRight classifier within sixteen steps (agreement 0%, outputs
  spanning [−2.0, 1.4]) and drags every other property with it: Properties 3 and 4 become
  "very true" (−1.9, −2.1) because COC is no longer minimal anywhere, and Properties 6, 9
  and 10, which demand COC or strongLeft, become much worse. Property 6 then cannot pull the
  network back inside its sixteen steps.
- **Epochs 7 to 10 walk the network most of the way home.** Properties 7, 8, 9, 10 each
  dislike strongRight in their regions, and by epoch 10 the network agrees with the original
  on 95.8% of inputs, with every property's loss within a few hundredths of its starting
  value. The ten-epoch tour has roughly cancelled out, apart from a residual 4% of the space
  that now advises strongRight and an output scale ten times larger than the original's.
- **One property per epoch is a form of catastrophic forgetting.** Each epoch optimises one
  loss with no memory of the previous nine; Property 5 undid what Properties 1 to 4 had done,
  and Properties 7 to 10 undid Property 5. This is the case for summing the ten losses into
  one objective (Chapter 4, Exercise #3, "several constraints at once") rather than
  visiting them in turn.


## Step 4: verifying the trained network on the ten-property specification

```bash
python3 verify-acas2.py models/acasXu_1_8_seq_e10.onnx --timeout 900
```

Step 4 was run three ways, one `verify-acas2.py` per snapshot set, all against the unchanged
ten-property `acasXu-upstream.vcl`, all with a 900 s limit per (snapshot, property):

- the final network `e10` on all ten properties (`traces/verify_e10.csv`);
- each intermediate snapshot `e01` … `e09` on the property it had just been trained on
  (`traces/verify_own_property.csv`), to ask whether one epoch on property k makes
  property k hold *immediately*;
- (the baseline column is step 1, `traces/verify_baseline.csv`).

| Property | N_{1,8} before training (step 1) | e10, after the ten-epoch tour (step 4) | snapshot e_k on its own property k |
|---:|---|---|---|
| 1 | ✓ verified (4 s) | ✓ verified (3 s) | e01: ✓ verified (6 s) |
| 2 | ✓ verified (302 s) | ✓ verified (20 s) | e02: ✓ verified (22 s) |
| 3 | ✗ falsified (2 s)<br>x = [1800, 0.06, 3.1, 980, 960] | ✓ verified (1 s) | e03: ✗ falsified (1 s)<br>x = [1800, -0.06, 3.1, 980, 960] |
| 4 | ✗ falsified (1 s)<br>x = [1800, 0.06, 0, 1000, 772] | ✗ falsified (2 s)<br>x = [1800, -0.06, 0, 1000, 769.7] | e04: ✓ verified (1 s) |
| 5 | ✗ falsified (1 s)<br>x = [400, 0.2, -3.137, 100, 44.44] | ✗ falsified (1 s)<br>x = [400, 0.2, -3.142, 100, 143.3] | e05: ✓ verified (2 s) |
| 6 | timeout (900 s) | ✗ falsified (106 s)<br>x = [3.205e+04, -3.142, -3.142, 606.2, 63.46] | e06: ✗ falsified (127 s)<br>x = [5.242e+04, -1.007, -3.137, 222.2, 246.6] |
| 7 | timeout (1588 s) | timeout (900 s) | e07: timeout (900 s) |
| 8 | timeout (1284 s) | timeout (900 s) | e08: timeout (900 s) |
| 9 | ✗ falsified (1 s)<br>x = [6940, -0.2133, -3.142, 109.2, 58.41] | ✗ falsified (2 s)<br>x = [7000, -0.14, -3.142, 125.2, 1.314] | e09: ✗ falsified (2 s)<br>x = [7000, -0.1438, -3.142, 103.2, 1.535] |
| 10 | ✓ verified (133 s) | ✓ verified (1113 s) | (e10, previous column) |

**A note on the time limit.** `verify-acas2.py` first used `Popen.communicate(timeout=…)`.
That worked on short tests but overran on long solver runs: Property 7 of the baseline was
reported as a timeout only after 1588 s, and Property 10 of `e10` ran to completion at
1113 s and returned `verified`. The script now kills the solver's process group from a
watchdog thread at the deadline, which does not depend on how the output is read; later
runs use that version. Results marked `timeout` are undecided, not falsified.

### Reading the verification tables

- **Property 3, the one the first round was about, is verified on `e10`.** It is falsified on
  the baseline and, notably, also falsified on `e03`, the snapshot trained on it: sixteen
  steps on Property 3 alone did not suffice here, whereas in the first round they did (with a
  differently-scaled loss and a fresh optimiser). What made it hold were the later epochs,
  4 to 6, which pushed clear-of-conflict out of the minimal position across the input space;
  by `e10` the network still advises strongRight on 4% of the space, evidently including the
  Property 3 region.
- **Properties 4 and 5 hold on the snapshot trained on them and are lost again by `e10`.**
  `e04` verifies Property 4 and `e05` verifies Property 5; `e10` falsifies both, with a
  counterexample for Property 4 at the same corner as the baseline's. This is the forgetting
  the loss matrix predicted: each later epoch optimises another property with no memory of
  these two.
- **Property 10 holds before and after**, so the last epoch's training on it preserved rather
  than created it. `e10` needs 1113 s to prove it against the baseline's 133 s, the opposite
  of Property 2's speed-up: the trained network's margins in Property 10's region are thinner.
- **Properties 1 and 2 were never at risk**, verified before and after, though `e10` proves
  Property 2 fifteen times faster than the baseline does (20 s against 302 s), a sign that
  the trained network's margins there are wider.
- **Net effect of the ten-epoch tour on N_{1,8}:** verified properties went from {1, 2, 10}
  to {1, 2, 3, 10}; Property 3 was gained, nothing that held was lost, and 4, 5, 9 remain
  falsified while 7 and 8 remain undecided (6 went from undecided to falsified).
- **Properties 6, 7, 8 could not be decided on any snapshot** within the limit: their
  regions cover most of the input space and Marabou needs longer than 15 minutes. `e06`,
  trained on Property 6, is falsified on it in 127 s, and `e10` in 106 s.
- **Property 9 is falsified everywhere**, including on `e09` right after training on it. Its
  region asks for strongLeft at low speeds, which no snapshot ever advises.

## Step 5: repeating the ten-epoch tour three times (30 epochs)

Requested after the results above: loop the tour, so that epoch e trains on property
((e−1) mod 10)+1, for three cycles. `pdt-acas2.py 1_8 --cycles 3 --tag cyc3`, everything
else as in Step 3, snapshots `models/acasXu_1_8_cyc3_eKK.onnx` after every epoch (so the
networks after 10, 20 and 30 epochs are `e10`, `e20`, `e30`).

### First attempt, without gradient clipping: diverged in the second tour

Epochs 1 to 10 reproduce Step 3 exactly (same seed, same starting weights). Epochs 11 and
12 improve Properties 1 and 2 a little further. Epoch 13, the second visit to Property 3,
runs away, and the weights overflow to NaN early in epoch 14. The run was stopped at epoch
26; its traces are kept as `traces/1_8_cyc3_*` and its snapshots as `models/*_cyc3_*`
(`e00` to `e13` are usable, `e14` onwards contain NaN).

| Epoch | Trained on | loss mean | loss last | agreement with original |
|---:|---|---:|---:|---:|
| 1 | property1 | -4.0188 | -4.0263 | 99.6% |
| 2 | property2 | -0.3468 | -0.3594 | 99.6% |
| 3 | property3 | -0.2823 | -0.2857 | 99.6% |
| 4 | property4 | -0.3536 | -0.5390 | 52.6% |
| 5 | property5 | +0.2097 | +0.0217 | 0.0% |
| 6 | property6 | +0.8744 | +0.4452 | 0.0% |
| 7 | property7 | -0.1848 | -0.1913 | 89.5% |
| 8 | property8 | +0.1659 | +0.1598 | 91.7% |
| 9 | property9 | +0.3650 | +0.3470 | 93.8% |
| 10 | property10 | +0.2803 | +0.2737 | 95.3% |
| 11 | property1 | -4.0637 | -4.0729 | 96.2% |
| 12 | property2 | -0.4069 | -0.4194 | 96.9% |
| 13 | property3 | -0.8048 | -2.3100 | 60.0% |
| 14 to 25 | property4 … property5 | nan | nan | (nan weights; the 99.6% recorded is an artefact of `argmin` on NaN outputs) |

Inside epoch 13 the loss went from −0.287 to −2.310 in sixteen steps while the gradient norm
climbed from 3.3 to 35; four steps into epoch 14 the loss was −3.76 at gradient norm 44
(`traces/1_8_cyc3_steps.csv`). All ten losses around the divergence:

| After epoch | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 | p10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 (p10) | -4.055 | -0.375 | -0.329 | -0.284 | +0.339 | +0.411 | -0.197 | +0.156 | +0.332 | **+0.273** |
| 11 (p1) | **-4.074** | -0.396 | -0.310 | -0.269 | +0.343 | +0.394 | -0.201 | +0.150 | +0.334 | +0.255 |
| 12 (p2) | -4.094 | **-0.421** | -0.287 | -0.250 | +0.351 | +0.373 | -0.207 | +0.141 | +0.340 | +0.235 |
| 13 (p3) | -4.093 | -0.419 | **-2.846** | -1.902 | +0.075 | +0.375 | -0.201 | +0.144 | +0.340 | +0.236 |
| 14 (p4) | nan | nan | nan | **nan** | nan | nan | nan | nan | nan | nan |

This is the failure mode of run 1 in `chapter-code/capucci-pdt/README.md`: the `qllAdditive`
loss has no floor (`trueElement = −∞`), so once a property's margin starts to grow,
minimising further only inflates the weights until float32 overflows. Two things made the
second visit worse than the first: Adam's moment estimates were carried over from the first
tour, so its effective step on Property 3 was larger, and Property 3's loss was already in
the region where it accelerates (compare the first round, where N_{1,7} and N_{1,9} reached
the same tail in their third epoch). The chapter's remedy is to bound the step, not the
loss: gradient clipping.

### Second attempt, with the gradient norm clipped at 1.0

`pdt-acas2.py 1_8 --cycles 3 --clip 1.0 --tag cyc3clip`. Identical to the first attempt
except that `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` is applied before
every optimiser step. Traces `traces/1_8_cyc3clip_*`, snapshots `models/*_cyc3clip_*`.

**Diverged at the same point.** Gradient clipping changed almost nothing: epochs 1 to 12 match
the unclipped run to within a few thousandths, epoch 13 (Property 3's second visit) runs
from −0.29 to −3.72 with the *unclipped* gradient norm reaching 45, and the weights are NaN
by the end of epoch 14. The full 30 epochs ran to completion but epochs 14 to 30 are
meaningless.

| Epoch | Trained on | loss mean | loss last | agreement with original |
|---:|---|---:|---:|---:|
| 1 | property1 | -4.0188 | -4.0263 | 99.6% |
| 2 | property2 | -0.3468 | -0.3594 | 99.6% |
| 3 | property3 | -0.2823 | -0.2857 | 99.6% |
| 4 | property4 | -0.3536 | -0.5396 | 60.9% |
| 5 | property5 | +0.2059 | +0.0216 | 0.0% |
| 6 | property6 | +1.0011 | +0.4498 | 0.0% |
| 7 | property7 | -0.1870 | -0.1954 | 85.7% |
| 8 | property8 | +0.1648 | +0.1572 | 87.3% |
| 9 | property9 | +0.3615 | +0.3453 | 91.0% |
| 10 | property10 | +0.2851 | +0.2769 | 93.4% |
| 11 | property1 | -4.0609 | -4.0713 | 94.1% |
| 12 | property2 | -0.4094 | -0.4236 | 94.5% |
| 13 | property3 | -1.5136 | -3.7212 | 61.8% |
| 14 to 30 | property4 … property10 | nan | nan | (nan weights) |

Why clipping did not help: Adam divides each parameter's step by a running estimate of its
gradient magnitude, so the size of the step it takes is set by the learning rate and almost
independent of the gradient's norm. Clipping the norm rescales the gradient, Adam undoes the
rescaling. The remedy for the carried-over momentum is a fresh optimiser per epoch.

Where the NaN actually comes from: at the end of epoch 13 the outputs span [−2.6, 1.4]; four
steps into epoch 14 the loss is −4.5, twelve steps in it is −13.3 (`traces/1_8_cyc3clip_steps.csv`).
A Capucci margin of 13 means `exp(p · 13) = exp(65)`, still inside float32; by step 16 the
margin passes 17.7, `exp(88.7)` overflows to `inf`, and `log(inf) − inf` is `nan`. The
compiled `qllAdditive` conjunction `(1/p)·log(Σ exp(p·xᵢ))` is evaluated literally, without
the usual log-sum-exp stabilisation, so it overflows as soon as any margin exceeds about
`88/p`. That is a separate, practical limit on how far this loss can be pushed at `p = 5`.

### Third attempt: a fresh Adam optimiser every epoch

`pdt-acas2.py 1_8 --cycles 3 --reset-adam --tag cyc3reset`, otherwise as the first attempt:
the optimiser is re-created at the start of every epoch, so no momentum or second-moment
estimate carries over from one property to the next.

**It survived all 30 epochs.** Per-epoch losses (three columns = the three tours):


| Epoch | Trained on | loss mean | loss last | agreement | | Epoch | Trained on | loss mean | loss last | agreement | | Epoch | Trained on | loss mean | loss last | agreement |
|---:|---|---:|---:|---:|---|---:|---|---:|---:|---:|---|---:|---|---:|---:|---:|
| 1 | p1 | -4.019 | -4.026 | 99.6% | | 11 | p1 | -4.051 | -4.058 | 99.6% | | 21 | p1 | -4.083 | -4.090 | 99.6% |
| 2 | p2 | -0.348 | -0.360 | 99.6% | | 12 | p2 | -0.409 | -0.422 | 99.6% | | 22 | p2 | -0.474 | -0.488 | 99.6% |
| 3 | p3 | -0.295 | -0.307 | 99.6% | | 13 | p3 | -0.243 | -0.253 | 99.6% | | 23 | p3 | -0.262 | -0.408 | 97.5% |
| 4 | p4 | -0.321 | -0.333 | 0.3% | | 14 | p4 | -0.271 | -0.295 | 99.5% | | 24 | p4 | -0.988 | -2.334 | 13.7% |
| 5 | p5 | +0.307 | +0.295 | 0.0% | | 15 | p5 | +0.323 | +0.299 | 78.6% | | 25 | p5 | +0.124 | +0.009 | 0.0% |
| 6 | p6 | +0.468 | +0.456 | 0.0% | | 16 | p6 | +0.415 | +0.401 | 99.6% | | 26 | p6 | +0.574 | +0.380 | 60.5% |
| 7 | p7 | -0.186 | -0.195 | 99.5% | | 17 | p7 | -0.212 | -0.222 | 99.6% | | 27 | p7 | -0.202 | -0.253 | 88.1% |
| 8 | p8 | +0.164 | +0.156 | 99.6% | | 18 | p8 | +0.134 | +0.126 | 99.6% | | 28 | p8 | +0.106 | +0.100 | 92.4% |
| 9 | p9 | +0.340 | +0.292 | 99.6% | | 19 | p9 | +0.322 | +0.183 | 99.6% | | 29 | p9 | +0.228 | +0.090 | 89.0% |
| 10 | p10 | +0.279 | +0.268 | 99.6% | | 20 | p10 | +0.229 | +0.219 | 99.6% | | 30 | p10 | +0.182 | +0.173 | 90.1% |

All ten losses after each tour (rows: start, after 10, 20 and 30 epochs):

| After epoch | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 | p10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -4.011 | -0.322 | -0.320 | -0.320 | +0.323 | +0.460 | -0.183 | +0.183 | +0.315 | +0.321 |
| 10 | -4.043 | -0.382 | -0.267 | -0.267 | +0.352 | +0.405 | -0.208 | +0.151 | +0.345 | +0.267 |
| 20 | -4.075 | -0.446 | -0.225 | -0.227 | +0.384 | +0.356 | -0.239 | +0.123 | +0.166 | +0.217 |
| 30 | -4.113 | -0.517 | -1.238 | -1.268 | +0.207 | +0.311 | -0.276 | +0.101 | +0.061 | +0.172 |

### Reading the run

- **With a fresh Adam per epoch the tours are stable and Properties 1, 2, 7, 8, 10 improve
  monotonically across the three tours** (p2: −0.32 → −0.38 → −0.45 → −0.52; p8: +0.18 →
  +0.15 → +0.12 → +0.10; p10: +0.32 → +0.27 → +0.22 → +0.17), with agreement back at 99.6%
  after every tour's tenth epoch. Resetting the optimiser removes the carried-over momentum
  that made the second visit to Property 3 diverge in the first two attempts. The third tour
  is more violent (epoch 24 on Property 4 takes its loss to −2.3 and agreement to 14%), and
  after 30 epochs Properties 3 and 4 sit at −1.24 and −1.27, well into the tail, at 90%
  agreement.
- **Properties 5 and 6 are the disruptive ones in every run.** They demand strongRight and COC
  in regions where N_{1,8} does something else, and each visit to them costs most of the
  agreement, which the following epochs then rebuild.

### Marabou verdicts for the 3-cycle snapshots, beside the earlier networks

Each of the three snapshots (after 10, 20 and 30 epochs of the fresh-Adam run) was verified
on all ten properties of the unchanged `acasXu-upstream.vcl`, one `verify-acas2.py` per
snapshot running in parallel, 900 s per property. The first two columns repeat Steps 1 and 4.
`python3 record-results.py` regenerates this table from `traces/verify_*.csv`; cells marked
*running* had no verdict when the README was last regenerated. ✓ verified, ✗ falsified with
Marabou's counterexample [distance, angle, heading, speed, intruderSpeed], timeout = undecided.

<!-- COMPACT VERIFY TABLE START -->
| Network | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | verified |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N_{1,8} baseline | ✓ (4 s) | ✓ (302 s) | ✗ (2 s) | ✗ (1 s) | ✗ (1 s) | timeout (900 s) | timeout (1588 s) | timeout (1284 s) | ✗ (1 s) | ✓ (133 s) | {1, 2, 10} |
| tour e10 | ✓ (3 s) | ✓ (20 s) | ✓ (1 s) | ✗ (2 s) | ✗ (1 s) | ✗ (106 s) | timeout (900 s) | timeout (900 s) | ✗ (2 s) | ✓ (1113 s) | {1, 2, 3, 10} |
| fresh Adam e10 | ✓ (12 s) | ✓ (42 s) | ✗ (2 s) | ✗ (2 s) | ✗ (3 s) | timeout (7964 s) | timeout (900 s) | timeout (900 s) | ✗ (2 s) | ✓ (64 s) | {1, 2, 10} |
| fresh Adam e20 | ✓ (32 s) | ✓ (36 s) | ✗ (2 s) | ✗ (2 s) | ✗ (2 s) | timeout (7964 s) | ✗ (3 s) | timeout (900 s) | ✗ (26 s) | ✓ (71 s) | {1, 2, 10} |
| fresh Adam e30 | ✓ (24 s) | ✓ (25 s) | ✓ (2 s) | ✓ (1 s) | ✗ (24 s) | timeout (7964 s) | ✗ (2 s) | ✗ (4 s) | ✓ (28 s) | ✓ (49 s) | {1, 2, 3, 4, 9, 10} |

† from the re-run with a one-property specification and a 9000 s limit (see below); the original 900 s verdict is in the detailed table.
<!-- COMPACT VERIFY TABLE END -->

The same verdicts in full, with counterexamples:

<!-- COMBINED VERIFY TABLE START -->
| Property | N_{1,8} baseline | tour e10 | fresh Adam e10 | fresh Adam e20 | fresh Adam e30 |
|---:|---|---|---|---|---|
| 1 | ✓ (4 s) | ✓ (3 s) | ✓ (12 s) | ✓ (32 s) | ✓ (24 s) |
| 2 | ✓ (302 s) | ✓ (20 s) | ✓ (42 s) | ✓ (36 s) | ✓ (25 s) |
| 3 | ✗ (2 s)<br><small>[1800, 0.06, 3.1, 980, 960]</small> | ✓ (1 s) | ✗ (2 s)<br><small>[1800, 0.06, 3.1, 980, 960]</small> | ✗ (2 s)<br><small>[1800, 0.06, 3.1, 980, 960]</small> | ✓ (2 s) |
| 4 | ✗ (1 s)<br><small>[1800, 0.06, 0, 1000, 772]</small> | ✗ (2 s)<br><small>[1800, -0.06, 0, 1000, 769.7]</small> | ✗ (2 s)<br><small>[1800, -0.06, 0, 1000, 742.2]</small> | ✗ (2 s)<br><small>[1800, 0.06, 0, 1001, 711.3]</small> | ✓ (1 s) |
| 5 | ✗ (1 s)<br><small>[400, 0.2, -3.137, 100, 44.44]</small> | ✗ (1 s)<br><small>[400, 0.2, -3.142, 100, 143.3]</small> | ✗ (3 s)<br><small>[400, 0.2891, -3.137, 103.2, 56]</small> | ✗ (2 s)<br><small>[400, 0.2, -3.137, 100, 134.2]</small> | ✗ (24 s)<br><small>[353.8, 0.3182, -3.137, 100.7, 38.4]</small> |
| 6 | timeout (900 s) | ✗ (106 s)<br><small>[3.205e+04, -3.142, -3.142, 606.2, 63.46]</small> | timeout (7964 s) | timeout (7964 s) | timeout (7964 s) |
| 7 | timeout (1588 s) | timeout (900 s) | timeout (900 s) | ✗ (3 s)<br><small>[6814, -0.02704, -2.129, 130.2, 89.31]</small> | ✗ (2 s)<br><small>[1.204e+04, 0.4116, -0.2451, 102.5, 121.6]</small> |
| 8 | timeout (1284 s) | timeout (900 s) | timeout (900 s) | timeout (900 s) | ✗ (4 s)<br><small>[3110, -3.142, 0.1, 632.1, 675.5]</small> |
| 9 | ✗ (1 s)<br><small>[6940, -0.2133, -3.142, 109.2, 58.41]</small> | ✗ (2 s)<br><small>[7000, -0.14, -3.142, 125.2, 1.314]</small> | ✗ (2 s)<br><small>[7000, -0.1644, -3.142, 100.1, 35.27]</small> | ✗ (26 s)<br><small>[4500, -0.3814, -3.141, 121.8, 75]</small> | ✓ (28 s) |
| 10 | ✓ (133 s) | ✓ (1113 s) | ✓ (64 s) | ✓ (71 s) | ✓ (49 s) |

Properties verified per network:

| Network | properties verified |
|---|---|
| N_{1,8} baseline | {1, 2, 10} |
| tour e10 | {1, 2, 3, 10} |
| fresh Adam e10 | {1, 2, 10} |
| fresh Adam e20 | {1, 2, 10} |
| fresh Adam e30 | {1, 2, 3, 4, 9, 10} |
<!-- COMBINED VERIFY TABLE END -->

**A second note on the time limit.** The watchdog version of `verify-acas2.py` also failed to
stop Property 6 on the three fresh-Adam snapshots at 900 s: all three ran for 7964 s before
the timer fired, although the same watchdog killed a test run at 4 s and at 40 s. The three
verifier processes each showed two threads, the main one blocked reading the solver's pipe
and the timer thread waiting, for the whole two hours; the timers fired together when the
solver next produced output. The mechanism is not understood. For future runs the robust
choice is Marabou's own per-query limit, which the chapter's scripts use:
`vehicle verify ... --solver-args --timeout=900`. Whatever the cause, a `timeout` verdict
still means only "undecided", and none of the decided verdicts are affected.

### Re-running the undecided properties with one-property specifications and a 9000 s limit

Requested after the table above: Properties 6, 7 and 8 on the baseline, and Property 6 on
the fresh-Adam `e30`, each verified against a specification containing **only that property**
(`specs-verify/property06.vcl` etc., the upstream text unchanged apart from dropping the
other nine `@property` blocks, so the strict guarded `minimalScore` is kept), with the limit
raised tenfold to 9000 s. The four runs went in parallel:

```bash
python3 verify-acas2.py acasXu_1_8.onnx --properties 6 --spec specs-verify/property06.vcl --timeout 9000 --csv traces/verify_long_baseline_p6.csv
python3 verify-acas2.py acasXu_1_8.onnx --properties 7 --spec specs-verify/property07.vcl --timeout 9000 --csv traces/verify_long_baseline_p7.csv
python3 verify-acas2.py acasXu_1_8.onnx --properties 8 --spec specs-verify/property08.vcl --timeout 9000 --csv traces/verify_long_baseline_p8.csv
python3 verify-acas2.py models/acasXu_1_8_cyc3reset_e30.onnx --properties 6 --spec specs-verify/property06.vcl --timeout 9000 --csv traces/verify_long_e30_p6.csv
```

`verify-acas2.py` now also wraps each call in GNU `timeout -s KILL`, a second layer under the
watchdog; both were tested to kill a run at 5 s.

**Split-and-conquer did not help either.** Marabou's parallel mode was tried on the baseline's
Property 7 with its one-property specification, `--solver-args "--snc --num-workers=4"`
(passed through by `verify-acas2.py --solver-args`), with a 600 s cap: neither of the
property's two queries finished (`traces/verify_snc_baseline_p7.csv`,
`marabou-outputs/acasXu_1_8_property7_property07_snc.txt`). The pass-through itself works:
Property 1 verifies under the same flags in 17 s against 4 s without, the difference being
the workers' start-up.

<!-- LONG RERUN TABLE -->

### Does the 30-epoch network avoid the trivial-network problem from acas?

**Yes.** Advisories over 4,000 uniform inputs across the whole valid input space, and over the
Property 3 region:

| Network | whole input space | agreement with original | Property 3 region |
|---|---|---:|---|
| original N_{1,8} | COC 99.8%, weakLeft 0.1%, weakRight 0.1% | 100% | COC 100% |
| fresh Adam e10 | COC 100% | 99.8% | COC 100% |
| fresh Adam e20 | COC 99.9%, strongLeft 0.1% | 99.7% | COC 100% |
| **fresh Adam e30** | **COC 90.8%, strongRight 5.5%, strongLeft 3.7%** | **90.5%** | strongRight 100% |
| acas round, N_{1,8} after 1 epoch on P3 | strongRight 86.5%, no COC anywhere | 0.5% | strongRight 100% |

In the acas round (the preliminary single-property round, see the appendix) the trained
network never advised COC. Here, after 30 epochs, COC is still the advisory on 91% of the
input space and the network agrees with the original on 90.5% of inputs, while inside the
Property 3 region it now advises strongRight instead of COC, which is what Properties 3 and 4
require. The change is localised rather than global. The price is a wider output range, from
about ±0.02 to [−0.34, 1.10].

Two reasons this differs from the acas round: the ten properties pull in different
directions, and the ones that demand COC (6 and 10) or forbid strong turns (7) push back
against the ones that forbid COC (3, 4), so no single property gets to reshape the whole
network; and each property only gets sixteen steps before the next one takes over, whereas
in the first round Property 3 had all 48 steps to itself. The cost is an output scale about fifty
times the original's, which is the unbounded Capucci loss at work, and a 9% slice of the
space whose advisory has changed with no data to say whether that is acceptable.

### Conclusions from acas2

- **On N_{1,8} the ten-property sequential tour works better repeated than done once.**
  Verified properties: baseline {1, 2, 10}; one tour {1, 2, 3, 10}; three tours with a fresh
  optimiser per epoch {1, 2, 3, 4, 9, 10}. Nothing that held was lost in either case.
  Properties 5, 7 and 8 remain falsified (5 and 7, 8 demand behaviour N_{1,8} was never
  trained for) and 6 remains undecided.
- **One property per epoch is catastrophic forgetting in slow motion, and the cure is
  repetition.** In the single tour, Properties 4 and 5 held on the snapshot trained on them
  and were lost by the end; the loss matrix shows each epoch undoing part of the previous
  ones. Over three tours the per-property losses improve monotonically from visit to visit,
  and the verified set grows.
- **Carrying Adam's state across the tours makes the run diverge; resetting it makes the run
  stable.** Two attempts with a persistent optimiser (with and without gradient clipping)
  overflowed to NaN at the second visit to Property 3; clipping cannot help under Adam. A
  fresh optimiser per epoch completed all 30 epochs.
- **The compiled Capucci loss has two practical limits worth knowing**: it is unbounded, so
  any property that is already satisfied keeps pulling the weights outward, and its
  conjunction is evaluated as a literal `log(Σ exp(p·x))`, which overflows float32 once a
  margin exceeds about `88/p`.
- **The two Vehicle 0.28.0 issues found in the first round carry over**: the guarded `i != j =>` compiles
  to a constant loss (worked around in `split-spec.py`), and `forall` keeps the least
  violating of the sampler's trajectories.

---

## Appendix: the preliminary single-property round (the former `acas` folder)

Before this round, a first experiment trained each of N_{1,7}, N_{1,8} and N_{1,9} on
Chapter 2's `acasXu.vcl`, which has **Property 3 only**, for three epochs of 16 steps with the
property loss alone (Capucci logic, `p = 5`, and again with `p = 2`). Its folder has been
removed because the result was not a usable network; its code and inputs are kept here
(`pdt-acas.py`, `verify-acas.py`, `acasXu.vcl`, `acasXu-training.vcl`, `acasXu-training-p2.vcl`,
and the three ONNX files), and the parts of its record that this round builds on follow.

**Why this round trains on all ten properties.** In that round, one epoch on Property 3 alone
was enough for Marabou to verify Property 3 on all three networks, and also enough to make
every network stop advising clear-of-conflict *anywhere* in the input space: the optimiser
satisfied "COC is not minimal in this region" by making COC never minimal at all. With no
task loss, a single property is satisfied most cheaply by a trivial, near-constant network.
Running several epochs of training over **all ten properties** is this round's answer to that
extreme over-fitting: the properties pull in different directions (6 and 10 demand COC,
7 forbids strong turns, 3 and 4 forbid COC), so no single one gets to reshape the whole
network, and each gets only sixteen steps before the next takes over. The section "Does the
30-epoch network avoid the trivial-network problem" above shows that this worked: after 30
epochs the network still agrees with the original on 90% of inputs while satisfying six
properties instead of three.

**Result of that round, for the record.** Verification with Chapter 2's command against
`acasXu.vcl`, Capucci `p = 5` (the `p = 2` run gave identical verdicts):

| Network | epoch 0 (start, re-exported) | epoch 1 | epoch 2 | epoch 3 |
|---|---|---|---|---|
| N_{1,7} | ✗ counterexample [1799.99, 0.0600, 3.0999997, 980.0, 1058.63] | ✓ verified | ✓ verified | ✓ verified |
| N_{1,8} | ✗ counterexample [1799.99, 0.0570, 3.0999997, 980.0, 960.0] | ✓ verified | ✓ verified | ✓ verified |
| N_{1,9} | ✗ counterexample [1799.99, 0.0600, 3.0999997, 980.0, 960.0] | ✓ verified | ✓ verified | ✓ verified |

Losses (Capucci value at the last step of each epoch; smaller is truer; see below for why they
are negative): N_{1,7} −0.320 → −0.381 → −0.877 → −4.139; N_{1,8} −0.320 → −0.346 → −0.373
→ −0.400; N_{1,9} −0.321 → −0.348 → −0.482 → −2.807. Violation rate in the Property 3 region
100% → 0% after epoch 1 on all three; agreement with the original network under 1% after epoch
1 on all three.

### What the verified networks actually did

Verified is not the same as useful. The advisory each snapshot gives, over 4,000 inputs
drawn uniformly from the whole valid input space and 4,000 from the Property 3 region:

| Network | epoch | whole input space | Property 3 region |
|---|---:|---|---|
| N_{1,7} | 0 | COC 97.3%, weakLeft 1.4%, weakRight 1.2%, strongRight 0.1% | COC 100% |
| | 1 | strongRight 96.0%, strongLeft 2.4%, weakRight 1.0%, weakLeft 0.6%, **COC 0%** | strongRight 100% |
| | 3 | strongLeft 90.5%, strongRight 9.5%, **COC 0%** | strongLeft 100% |
| N_{1,8} | 0 | COC 99.7%, weakLeft 0.1%, weakRight 0.1% | COC 100% |
| | 1 | strongRight 86.6%, weakLeft 5.1%, strongLeft 4.9%, weakRight 3.4%, **COC 0%** | strongRight 100% |
| | 3 | strongRight 88.6%, strongLeft 5.5%, weakLeft 3.2%, weakRight 2.7%, **COC 0%** | strongRight 100% |
| N_{1,9} | 0 | COC 99.9%, weakLeft 0.1%, weakRight 0.1% | COC 100% |
| | 1 | strongLeft 87.6%, weakRight 9.6%, weakLeft 2.1%, strongRight 0.8%, **COC 0%** | weakRight 100% |
| | 3 | weakRight 91.8%, weakLeft 8.0%, strongRight 0.1%, **COC 0%** | weakRight 100% |

Two things follow.

**The property was trained in by making clear-of-conflict unreachable everywhere.** Property
3 only constrains a thin slice of the input space, but the loss acts on the weights, and the
cheapest way to raise the clear-of-conflict score in the slice is to raise it globally.
After one epoch none of the networks advises clear-of-conflict for *any* sampled input,
including the 97–99.9% of the space where the original networks did. Which alternative
advisory wins is arbitrary and changes between epochs (N_{1,7} flips from strongRight to
strongLeft between epochs 1 and 3). These are constant-ish classifiers that happen to
satisfy Property 3, not collision-avoidance controllers.

**This is the `ALPHA = 0` result the chapter warns about, seen on a real property.**
`chapter-code/pure-pdt.md` argues that a constant classifier satisfies robustness perfectly,
so constraint-only training must be expected to destroy accuracy; the chapter's
Exercise #3 asks the reader to think about what `ALPHA = 0` must do before running it. The
ACAS Xu case is sharper still, because the target property is satisfied by any network that
never advises clear-of-conflict, and the optimiser finds that solution in 16 steps. A
note of caution on the agreement proxy: because the original networks advise
clear-of-conflict on almost all *uniformly sampled* inputs, "agreement" here mostly measures
whether clear-of-conflict is still advised; it does not tell us how the trained networks
behave on the realistic encounter distribution, for which there is no data here.

### Compiling the specification to a loss

### The problem: the guarded implication compiles to the constant zero

Chapter 2's `advises` is

```vehicle
advises i x = forall j . i != j => normAcasXu x ! i < normAcasXu x ! j
```

This type-checks and verifies correctly, and under Vehicle 0.28.0 it also *compiles* to a
loss without complaint (the `CompareIndex` limitation reported in the chapter-code README
no longer applies). But the loss it produces is identically zero. On the starting networks,
under both the default Vehicle logic and the Capucci logic:

```
property3 loss = 0.0000e+00    grad norm = 0.000e+00
advises(clearOfConflict, <Marabou's counterexample>) = 0.0000e+00
```

while the network advises clear-of-conflict on 100% of the region. Forty Adam steps on this
loss changed nothing: loss 0, violation rate 100%, agreement with the original 100%.

Dumping the compiled program (`vehicle --json compile loss --logic qllAdditive
--specification acasXu.vcl`) shows why. The body of `advises` becomes

```
reduceConjunction( foreach k .
  WhereTensor( pointwiseLessThan(out ! i, out ! k),      -- value if condition holds
               BoolNot(BoolCompareIndex(Ne, i, k)),      -- condition:  NOT (i != k)
               trueElement ) )                           -- value otherwise
```

and the PyTorch back-end evaluates `WhereTensor(a, cond, b)` as `where(cond, a, b)`. The
condition is `not (i != k)`, i.e. `i == k`, so the real comparison is kept **only** for
`k = i` (where it is `s - s = 0`) and every `k != i` receives the logic's true element. The
guard's polarity is inverted: the compiler produces "if `i == k` then compare else true"
instead of "if `i != k` then compare else true". The reduction is therefore over
`[0, true, true, true, true]`, which is 0 under either logic, and so is `property3`, with
zero gradient. The verifier back-end is unaffected, which is why verification works.

### The workaround: drop the guard and make the comparison non-strict

`acasXu-training.vcl` (kept in this folder) uses the form that Chapter 4 already uses for Fashion MNIST
(`chapter-code/fmnist-robustness.vcl`, `capucci-pdt/fashionRobustness-capucci.vcl`):

```vehicle
advises i x = forall j . normAcasXu x ! i <= normAcasXu x ! j
```

The `j = i` case is now trivially true (`s <= s`) and needs no guard. The two forms differ
only on ties. Under the training form, `not (advises clearOfConflict x)` asks that some
other advisory scores *strictly* below clear-of-conflict, which is exactly what Marabou
checks after its own strict-to-non-strict conversion. This is the only change to the
property; the `qllAdditive` logic block appended at the end of the file is Vehicle's
mechanism for choosing the loss, not part of the property.

With this change the compiled loss is no longer trivial. At the Marabou counterexample of
N_{1,7}, where the outputs are `[-0.0203, -0.0188, -0.0189, -0.0178, -0.0178]`:

| | guarded `i != j =>`, `<` | unguarded, `<=` |
| --- | ---: | ---: |
| `advises(COC, x)`, Capucci `p = 2` | 0.0000 | +0.8032 |
| `property3` loss, Capucci `p = 2` | 0.0000 | −0.8032 |
| `property3` loss, Capucci `p = 5` | 0.0000 | −0.3203 |
| gradient norm w.r.t. the weights | 0.000 | 0.896 |

(For reference, with all five margins near zero the Capucci conjunction is
`(1/p) · log(Σ exp(p · margin)) ≈ log(5)/p`, which is 0.80 at `p = 2` and 0.32 at `p = 5`;
the numbers above are that quantity, so the loss is measuring what it should.)

**To be clear about what made the difference: the Capucci logic was not the cause.**
Removing the guard and making the comparison non-strict is what produced the non-trivial
loss, not the logic choice. The guarded form gives exactly zero under the default Vehicle
logic and under the Capucci logic alike; the unguarded form is non-trivial under both. The
logic only shapes the loss once there is one to shape. (The Capucci logic is used here
because the chapter uses it, and at `p = 5` at the chapter author's request; the chapter's
own experiment uses `p = 2`.)

### Two further observations about the compiled loss

Neither affects this experiment, but both are worth knowing.

1. **The `forall x` is compiled with the right bounds.** Vehicle extracts the search box
   for `x` from the antecedent `validInput x and directlyAhead x and movingTowards x`:
   lower `[1500, max(-π, -0.06), max(-π, 3.10), 980, 960]`, upper
   `[1800, 0.06, π, 1200, 1200]`, in problem-space units, with `normalise` applied inside
   the body. The antecedent itself is then dropped from the body, so it never masks the
   consequent. This is the reason no data set is needed: the sampler draws its own inputs.

2. **Across the ten adversarial trajectories, the least violating one is kept.** The
   compiled `property3` is `neg( ReduceMax( sampler( search_lambda = neg(neg(advises)) ) ) )`.
   The sampler minimises `search_lambda = advises` along each trajectory, which correctly
   seeks the most violating point; but `ReduceMax` then takes the *largest* `advises` value
   over the ten end-points, i.e. the least violating trajectory, and negates it. A
   worst-case quantifier should take the smallest. Here all ten trajectories agreed to four
   decimals, so it made no difference, but on a property that is violated only in part of
   its region it would make the loss optimistic.

### A practical note on compiling from Python

`vehicle_lang` runs the Vehicle compiler in-process (a Haskell runtime loaded once per
Python process). Twice during this work `load_specification` returned empty output and
failed with `JSONDecodeError: Expecting value`, and on both occasions another Vehicle
process (a `vehicle verify`, or a second compile in the same process) was running at the
same time. Run one Vehicle invocation at a time. `pdt-acas.py` compiles the specification
once per run, and retries if the compile fails.

**Why the losses are negative.** In the `qllAdditive` logic `trueElement` is −∞ and
`falseElement` is +∞: smaller is truer, and zero has no special meaning. A comparison
`a <= b` is interpreted as the raw margin `a − b`, negation as `x ↦ −x`, and conjunction as
the soft maximum `(1/p)·log(Σ exp(p·xᵢ))`; nothing is clamped at zero. So `advises COC x` is
the soft maximum of the five margins `out_COC − out_k`, which at the start are all within a
few thousandths of zero, giving roughly `log(5)/p` (the soft maximum exceeds the true
maximum by up to `log(n)/p`, and with five nearly equal margins it sits at that ceiling).
Property 3 is `not (advises COC x)`, i.e. `−advises`, hence −0.32 at `p = 5` and −0.80 at
`p = 2` before training. Training makes the property truer, so the loss falls; once some
other advisory scores far below clear-of-conflict, one margin becomes large and positive and
the loss heads towards −∞. Because of the soft-max offset there is no fixed value at which
the property "becomes true"; read the loss relatively, and use Marabou for the verdict.

