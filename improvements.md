# Improvements for Vehicle Tutorial
This file is for making miscellaneous notes on how to improve the Vehicle tutorial pages. These may be anything from big ideas to minor changes.

**General note**: at some point we should decide where to talk about network normalisation. Matthew says this should take place within the network itself (i.e., in the ONNX file, viewable using [this site](https://netron.app/)), rather than inside of a specification.

## Ch. 4: Property-Driven Training
My overall vision for this chapter is to introduce a concrete example to aid with the explanation of how to use Vehicle specifications/properties to generate logical loss functions, with the task loss function to train a neural network.

I will use the MNIST Fashion dataset to train a (very simple) neural network and use (some/various definition(s) of) robustness to generate a logical loss function. I will compare this network with one that hasn't been trained using a logical loss function (just the standard task loss function) and show the difference in both task accuracy and satisfaction of the robustness property.

It may also be interesting to investigate these metrics (accuracy, robustness) with different types of differentiable logics, weighting schemes, etc. Using multiple robustness properties is also a good opportunity to demonstrate how they can be implemented in Vehicle, and involving them in the statistical analysis may provide insights into their respective differences and use cases.

Whilst the theoritical background is useful in providing a motivation for generating logical loss functions, it is not entirely necessary for learning to generate/use logical loss functions. I may cut this down significantly.

Current roadblocks:
- Logical loss function generation is currently broken: see [here](https://github.com/vehicle-lang/vehicle/issues/1183).
- For a large number of images (which I would like to use for more accurate statistics), the `vehicle compile queries` command takes a _lot_ of memory: see [here](https://github.com/vehicle-lang/vehicle/issues/1184). I need to learn how to perform this in batches (and this might also have to feature in the tutorial so that it can be replicated by those learning the language).

**Note on batch processing:** I am currently working on a script that performs batch query compilation. This will take a large idx file and generate $n$ batches of queries. The number of queries depends on the property being validated, but in general the number of queries in each batch is:

$$
    k=\frac{\text{number of images}\times Q}{n}
$$

where $Q$ is the constant number of queries that a property generates for a single data point.

## Resolved: `requirements.txt` and CI on the `exercises` branch

**Raised 2026-08-26; decided the same day — option 2, correct rather than
remove.** Both `requirements.txt` and `.github/workflows/ci.yml` were repaired
on the `exercises` branch rather than deleted. What was done is recorded under
"What was implemented" below; the findings that motivated it are kept for the
record.

The question was whether `requirements.txt` (and the `ci.yml` that uses it)
should be deleted from the `exercises` branch, or repaired.

Findings:

- `requirements.txt` exists at the root of the `exercises` branch and pins
  `vehicle-lang ==0.17.0`, `maraboupy ==1.0.0`, `agda ==2.7.0.1`. The pinned
  Vehicle is far behind the current release (0.27.1), which is what
  `pip install vehicle-lang` gives readers of Chapter 1.
- Its only reference is `.github/workflows/ci.yml`, line 31
  (`pip install -r requirements.txt`).
- That workflow is already dead on `exercises`: it triggers only on pushes to
  the `tutorial` branch, and its steps `cd` into `examples/acasXu`,
  `examples/mnist-robustness` and `examples/hyperrec`, which exist on the
  `tutorial` branch but not on `exercises`. It also runs `vehicle check`, which
  is not a valid subcommand in 0.27.1 (`Invalid argument 'check'`).
- `ci.yml` is byte-identical on both branches, and the `tutorial` branch has its
  own `requirements.txt` and `ci.yml`, so the pair on `exercises` looks like a
  leftover copy. Nothing on the `pages` branch references a `requirements.txt`.

Two proposed actions, either of which is self-consistent:

1. **Delete both** `requirements.txt` and `.github/workflows/ci.yml` from
   `exercises`. The `tutorial` branch keeps its working copies. Consequence:
   no CI on the branch that holds the live code — honest, since it cannot pass
   there today. Deleting only `requirements.txt` is worse: it leaves the
   workflow referencing a missing file.
2. **Repair instead:** retarget the trigger to `exercises`, point the steps at
   `chapter-N/chapter-code`, replace `vehicle check` with `vehicle typecheck`,
   and unpin `vehicle-lang`. This would have caught the `--verifier` →
   `--solver` breakage that went unnoticed across Chapters 2, 3 and 5, so it has
   real value.

**Version/command coupling (important for whoever picks this up).** The command
name and the pinned version have to be changed together, or CI fails either way:

- `vehicle check` was renamed to `vehicle typecheck` in the compiler by commit
  `72e97b1c` ("Rename `check` command to `typecheck` and update docs", #1014).
  `git tag --contains 72e97b1c` shows it shipped in **v0.23.0 and later only**;
  `git merge-base --is-ancestor 72e97b1c v0.17.0` confirms it is *not* in
  v0.17.0.
- So with `requirements.txt` pinning `vehicle-lang ==0.17.0`, the workflow's new
  `vehicle typecheck` will fail with `Invalid argument 'typecheck'`. Before the
  edit it used `check`, which 0.17.0 has but 0.27.1 does not.

To make CI actually pass on `exercises`, option 2 needs all four of:

1. raise or drop the `vehicle-lang` pin (0.27.1, or unpinned),
2. keep `vehicle typecheck` (already done in `7a5349f`),
3. retarget the trigger from `push: branches: [tutorial]` to `exercises`,
4. point the steps at `chapter-N/chapter-code` instead of `examples/*`.

Related: the `tutorial` branch (last commit 2026-08-03) and `source-donotuse`
are both still public, and the stale `tutorial` branch is what keeps this CI
alive. Retiring it would remove the ambiguity. Chapter 3's links used to point
into `tutorial`; they now point at `exercises`.

### What was implemented

`requirements.txt`:

- `vehicle-lang` is now **unpinned**, not raised to a fixed version. Chapter 1
  tells readers to run `pip install vehicle-lang`, so CI now tests against
  whatever that gives them. Pinning is precisely how this file drifted to
  0.17.0 while the chapters documented a newer command line; an exact pin would
  keep CI green while readers hit failures. Revisit if reproducibility matters
  more than drift detection.
- `maraboupy` raised `==1.0.0` -> `==2.0.0` (current, and what is installed
  locally).
- `idx2numpy ==1.2.3` and `numpy` added: `chapter-3/exercises/create_dataset.py`
  needs both and neither was declared anywhere.
- `agda ==2.7.0.1` left alone; it is still the current release.

`.github/workflows/ci.yml`:

- trigger changed from `push: branches: [tutorial]` to `exercises`;
- the three hard-coded `examples/*` steps replaced by a loop over
  `git ls-files '*.vcl'`, so all 16 specifications are checked and new ones are
  picked up automatically. Specs containing `your answer here` are skipped, as
  they are meant not to compile until the reader completes them. That rule is
  exact: it matches only `resources-by-dataset/mnist/mnist-robustness-incomplete.vcl`,
  the single spec that legitimately fails to typecheck, while
  `chapter-2/exercises/acasXu-incomplete.vcl` has no placeholders and passes;
- **a `vehicle verify` step was added.** This corrects the claim made earlier in
  this note that a working CI "would have caught the `--verifier` breakage".
  A typecheck-only workflow would *not* have: `--verifier` is a `verify`-only
  flag. The new step runs one real verification (Chapter 2's ACAS Xu property 3,
  a single query) so the `verify` command line is actually exercised. Note that
  property 3 is false for network 1_7, so a counterexample is the expected
  result and `vehicle verify` still exits 0.

Verified locally by running both steps under `sh` exactly as CI does: 15 specs
pass, 1 is skipped, exit status 0; the verification completes and exits 0.

**Not yet verified:** the workflow has never run on GitHub, so dependency
resolution on a clean Ubuntu runner is untested — in particular whether
`agda ==2.7.0.1` installs there. Watch the first run on `exercises`.
