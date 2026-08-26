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

## Open question: `requirements.txt` and CI on the `exercises` branch

**Raised 2026-08-26, parked pending input from other developers.** One partial
change has since been made: commit `7a5349f` on the `exercises` branch replaced
`vehicle check` with `vehicle typecheck` in `ci.yml`. The rest is untouched, and
that change alone does not make CI pass — see "Version/command coupling" below.

Should `requirements.txt` (and the `ci.yml` that uses it) be deleted from the
`exercises` branch, or repaired?

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
