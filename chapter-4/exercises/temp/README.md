# Vehicle 0.27.1: reproducers for four issues found on 2026-08-27

> **Where to run these.** The Python scripts in `loss-quantifier/` load their
> specification by relative path, so they must be run from
> `chapter-4/chapter-code/`, not from this directory:
>
> ```bash
> cd ../../chapter-code
> python3 ../exercises/temp/loss-quantifier/B-default-and-dl2-rows.py
> ```
>
> They also need FashionMNIST only for the training logs, not for the epsilon
> sweeps — those use random tensors and an untrained network, so they run anywhere
> the spec resolves. The `.vcl` files in `specs/` are self-contained and can be
> typechecked in place with `vehicle typecheck --specification specs/<file>`.
>
> This is a scratch folder: the material is working evidence for bug reports, not
> tutorial content.

Found while preparing Chapter 4 of the tutorial (property-driven training). None of these
are filed yet. Listed worst-first.

## 1. `forall` is inverted at training time (most serious, fails silently)

A `forall` quantifier requires the worst case, so a wider search space and a harder search
must both make the compiled loss *worse*. Both do the opposite.

Reproducer: `loss-quantifier/B-default-and-dl2-rows.py`, but **remove the
`with torch.no_grad():`** — see the caveat below. Measured with gradients enabled, using
`fmnist-robustness.vcl` and Vehicle's default logic, on an untrained network and 4 random
images (0 = true, positive = false):

| epsilon | loss   |     | search effort            | loss   |
| ------: | -----: | --- | ------------------------ | -----: |
| 0.0     | +0.1985|     | 1 sample,  1 step        | +0.1537|
| 0.02    | +0.1467|     | 1 sample,  5 steps       | +0.1478|
| 0.5     | +0.0848|     | 10 samples, 5 steps      | +0.1467|
| 2.0     | +0.0596|     | 40 samples, 5 steps      | +0.1460|
|         |        |     | 10 samples, 20 steps     | +0.1439|

Widening the ball 100x makes the property look 3.3x *truer*; searching 20x harder makes it
truer still. Marabou meanwhile falsifies 16 of 37 correctly-classified images at
epsilon 0.02 on the same network family.

Consequence: property-driven training cannot train epsilon-robustness with this
specification, and the failure is invisible from the training curve. Affects
`pt_classifier.py` and `tf_classifier.py` in the tutorial, and all three logics tested
(Vehicle default, DL2, and a custom one).

**Measurement caveat that cost me an hour:** `DefaultPyTorchSampler` only takes FGSM steps
when `loss.requires_grad` is true, else it substitutes a zero gradient. Calling the loss
under `torch.no_grad()` therefore silently disables the adversarial search and understates
the effect (drift of 14% rather than 3.3x). The saved scripts contain the `no_grad` version
as run; delete that line to reproduce the real behaviour.

**The search is definitely running** — this is not a case of the sampler being skipped.
Instrumenting an unmodified `pt_classifier.py` (batch of 64, `SUBSET_SIZE` reduced to 64,
one epoch) by counting calls into `DefaultPyTorchSampler.get_loss` and
`torch.autograd.grad`:

    sampler configured with num_samples=10, num_steps=5
    sampler get_loss calls        : 64      (one per image in the batch)
    torch.autograd.grad calls     : 3200    (= 64 x 10 x 5, exactly as expected)
    loss requires_grad True       : 64
    loss requires_grad False      : 0

So FGSM runs at full effort on every image and still reports the property as truer when
the ball widens and truer still when the search works harder. Probe script:
`loss-quantifier/probe-pt-classifier.py`.

Behaviour established conclusively; cause in the compiler NOT established. Report as
symptoms.

## 2. DL2 logic yields `+inf` on this specification

`DL2DifferentiableLogic` returns `+inf` at every epsilon for `fmnist-robustness.vcl`, so it
provides no gradient at all. Same reproducer, second loop iteration.

## 3. `@parameter` inside a `DifferentiableTensorLogic` fails to compile

`specs/test-capucci.vcl` typechecks but `load_specification` dies:

    Internal scoping error: declaration 'User.p' not found in scope
      internalScopingError, Vehicle/Data/Variable/Free/Context/Core.hs:21
      lookupInFreeCtx, Vehicle/Compile/FunctionaliseResources

`specs/test-capucci-const.vcl` is the same file with `p : Real ; p = 1.0` as a plain
definition instead, and works. So a custom logic cannot currently take a runtime parameter.

## 4. de Bruijn index error compiling a `foreach` inside a `forall`

`specs/novacuity.vcl` typechecks (exit 0) but the loss backend dies:

    Internal scoping error during lookupIxInBoundCtx,
      Vehicle/Data/Code/ForcedValue.hs:128
    the bound context of length '3' is smaller than the found DB index 𝓲4

The spec quantifies `forall (delta : Image)` and builds the perturbed image with
`foreach i j . clamp01 (image ! i ! j - epsilon * squash (delta ! i ! j))`. It was written
to test whether vacuous satisfaction explained issue 1 — that test never ran because of
this crash.

`specs/novalid.vcl` is the fallback test that did run (original spec with the
`validImage` antecedent removed). Removing it made the wrong-direction drift *stronger*,
32% versus 14%, which refuted the vacuity hypothesis.

## Files

- `loss-quantifier/A-capucci-row.py` — epsilon sweep, custom qllAdditive logic
- `loss-quantifier/B-default-and-dl2-rows.py` — epsilon sweep, Vehicle default and DL2
- `loss-quantifier/out.log`, `eps-sens2` output — the sweeps as run (under `no_grad`)
- `loss-quantifier/samples2.log` — the search-effort sweep WITH gradients (the good data)
- `loss-quantifier/train-capucci*.log` — three training runs: alpha 0, alpha 0.5, and
  alpha 0.5 with the constraint loss clamped at 0
- `specs/` — the four specifications referenced above
