# Exercise #6: the ACAS Xu benchmark, starting material

This folder holds what you start Exercise #6 from. The complete worked solution, with the
training script, every snapshot, all solver transcripts and a README that records each step,
is in [`../../solutions/acas2/`](../../solutions/acas2/README.md).

| File | Role |
| --- | --- |
| `acasXu-upstream.vcl` | the full ACAS Xu specification, ten properties, from [vehicle-lang/vehicle, examples/acasXu/acasXu.vcl (dev)](https://github.com/vehicle-lang/vehicle/blob/dev/examples/acasXu/acasXu.vcl); used for **verification**, unchanged |
| `acasXu_1_7.onnx`, `acasXu_1_8.onnx`, `acasXu_1_9.onnx` | the three networks, identical to Chapter 2's |
| `split-spec.py` | writes the ten one-property specifications: `specs/propertyKK.vcl` for **training** (with the changes the loss compiler needs), or `specs-verify/propertyKK.vcl` for verifying one property at a time |
| `capucci-logic.vcl` | the chapter's Capucci `qllAdditive` logic block, appended to each training specification by `split-spec.py` |
| `verify-acas2.py` | runs Chapter 2's verification, one `vehicle verify` per (network, property), with a time limit and a transcript per call |

## What to do

1. **Verify the shipped networks.** Reproduce the table in the exercise text:

   ```bash
   python3 verify-acas2.py acasXu_1_8.onnx --timeout 900
   ```

   Verdicts and times go to `traces/verify.csv`, transcripts to `marabou-outputs/`. Expect
   Properties 1, 2 and 10 to verify, 3, 4, 5 and 9 to be falsified in seconds, and 6, 7 and 8
   to hit the cap.

2. **Split the specification.** `python3 split-spec.py --p 5` writes the ten training
   specifications with the Capucci logic at hardness 5 (the chapter's own experiment uses 2).
   Look at what changed in `minimalScore` and `maximalScore`, and why: the guarded implication
   `i != j => ...` compiles to a constant loss, so the training form drops the guard and uses a
   non-strict comparison. `python3 split-spec.py --for-verification` writes the same ten
   properties unchanged, one per file, for verifying a hard property on its own.

3. **Train, one property per epoch.** Write the training loop yourself, starting from
   `chapter-code/capucci-pdt/pdt-Capucci-v028.py`: reload the ONNX weights into PyTorch,
   compile `specs/propertyKK.vcl` to a loss with `load_specification(..., logic=
   vcl.CustomDifferentiableLogic("qllAdditive"))`, and in epoch k take 16 Adam steps on
   property k's loss alone. There is no data set, so there is no task term. Save a snapshot
   after every epoch. Then repeat the tour of the ten properties two more times, and start a
   fresh optimiser each epoch, or the second visit to a property will diverge.

4. **Verify what you trained**, with the same command as in step 1 pointed at your snapshots.
   Which properties did you gain, which did you lose, and does the network still advise
   clear-of-conflict where the original did?

Run one Vehicle compile at a time, and cap every Marabou call: Properties 6, 7 and 8 do not
finish in fifteen minutes on any of these networks, and did not finish in two and a half hours
on the shipped one.
