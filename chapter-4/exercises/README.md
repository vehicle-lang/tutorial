# Chapter 4 exercises

All the exercises use the material in `../chapter-code`; there is nothing extra to
download. Read that folder's README first: it walks through both of the chapter's
experiments, and Exercises #1 and #2 are essentially those experiments.

Before starting, check the Vehicle version:

```bash
vehicle --version
```

It must report **0.28.0 or later**. Property-driven training does not work in 0.27.1
and earlier (the compiled loss searches in the wrong direction), and the symptom is
subtle: training runs, the loss falls, and the verified network is *less* robust than
before. The chapter-code README explains how this was found.

Two practical notes that apply throughout:

- **Training time.** The constraint loss runs an adversarial search for every image in
  every batch, so an epoch on 1024 images takes ten to fourteen minutes on a CPU. Budget
  about two hours for ten epochs.
- **Verification time and memory.** Verifying fifty images against the Chapter 3
  specification takes between twelve and forty minutes per network, and Marabou uses
  about 14 GB of memory while it runs. Run one verification at a time.

## Exercise #1 (⭑): Run the chapter code

Download the required materials (or produce them yourself) and repeat the steps
described in this chapter.

Everything is in `../chapter-code`. Follow its README in order:

1. `vanilla_classifier.py` trains a network on cross-entropy alone.
2. `pt_classifier.py` (or `tf_classifier.py`) trains one on the blended objective, using
   the specification `fmnist-robustness.vcl` compiled to a loss with Vehicle's default
   differentiable logic.
3. `capucci-pdt/pdt-Capucci-v028.py` continues the vanilla network with the blended
   objective, and `capucci-pdt/verify_v028.py` verifies every snapshot it produces.

The README records the output each step produced on the maintainers' machine, so you
can check yours against it. Your numbers will differ a little, since only
`pt_classifier.py` fixes a random seed.

## Exercise #2 (⭑): Verifying and comparing networks

Use a Vehicle specification (either the one provided, or your own) to verify a property
(e.g., robustness) of a network trained *with* a logical loss function, and compare this
to one trained *without* a logical loss function. Which is more robust, which has the
better task accuracy, and why?

The chapter-code README's worked experiment is exactly this comparison: the
100-epoch vanilla network against the same network after ten epochs of property-driven
training. You can reproduce it in full (about four hours) or, more cheaply, verify only
the two endpoints:

```bash
cd ../chapter-code/capucci-pdt

# the network trained without a logical loss
vehicle verify \
  --specification fashionRobustness-solution.vcl \
  --network classifier:vanilla_e100.onnx \
  --parameter epsilon:0.02 \
  --dataset trainingImages:0-49Images.idx \
  --dataset trainingLabels:0-49Labels.idx \
  --solver Marabou

# the same network after ten epochs of property-driven training
vehicle verify \
  --specification fashionRobustness-solution.vcl \
  --network classifier:capucci-models-v028/capucci_v028_e10.onnx \
  --parameter epsilon:0.02 \
  --dataset trainingImages:0-49Images.idx \
  --dataset trainingLabels:0-49Labels.idx \
  --solver Marabou
```

Both networks are committed, so this needs no training at all. When you compare the
counts, remember that an image the network misclassifies can never be proved robust, so
read the verified count against the number of correctly classified images, not
against 50. The chapter-code README explains how to get that number.

## Exercise #3 (⭑⭑): Further experimentation

Try various combinations of task loss functions, constraint loss functions, and alpha
values. How do these affect each other? Is there a combination that makes the network
more robust? Is there a combination that makes the network more accurate? What happens
when you use multiple constraint loss functions simultaneously?

The knobs are all near the top of `capucci-pdt/pdt-Capucci-v028.py`:

- `ALPHA` weights the task loss; `1 - ALPHA` weights the constraint loss. The chapter
  uses 0.4. Try 0.2 and 0.8, and think about what `ALPHA = 0` must do to a classifier
  before you run it.
- `EPSILON` is the radius the loss is trained at. The chapter uses 0.02 because at 0.005
  the vanilla network is already robust on all but one eligible image, leaving nothing
  to improve. What happens if you train at 0.02 and verify at 0.005, or the reverse?
- The **differentiable logic** is chosen where the specification is loaded:

  ```python
  spec = loss_pt.load_specification(path, logic=vcl.CustomDifferentiableLogic("qllAdditive"))
  ```

  `vcl.VehicleDifferentiableLogic()` and `vcl.DL2DifferentiableLogic()` are built in,
  and any `DifferentiableTensorLogic` declared in the specification can be named. The
  Capucci logic used in the chapter is declared at the bottom of
  `capucci-pdt/fashionRobustness-capucci.vcl`; its hardness `p` is a plain definition
  there, so changing it means editing that line.
- **Several constraints at once** means several `@property` declarations in the
  specification, each loaded from `spec[...]` and summed into the objective with its own
  weight. Chapter 3's exercises are a source of alternative robustness definitions.

Write your snapshots to a new folder so that the committed ones stay as a reference, and
keep the verification command unchanged so that your counts are comparable with the
README's tables.

## Exercise #4 (⭑⭑⭑): Training a model from scratch

Finally, try creating your own model from scratch and repeat the experiments and
comparisons described above. Explore the relationship between how complex a model is
and to what degree it can satisfy robustness, and the effect robustness training can
have on this. Hint: a simple model is worse at spotting the difference between two
different images. Does this make it more or less likely to be robust?

`vanilla_classifier.py` and `pt_classifier.py` are the templates. Keep the pixels in
`[0, 1]` and do not add a `Normalize` transform: the specification and the `.idx` files
both use that range, and a normalised network is verified on a different input space
from the one it was trained on. `chapter-code/normalisation-options.md` records what
went wrong when this chapter's authors did exactly that.

Two things to keep constant so that your networks stay comparable: the verification
command from Chapter 3 Exercise #7, and the fifty test images it uses. Marabou can run
out of memory on wider or deeper networks; Vehicle then reports those images as
`errored`, which is expected rather than a mistake on your part.

Sample solutions and expected results are in `../solutions`.

The `temp/` folder here is not exercise material. It holds the reproducers for the
Vehicle 0.27.1 bug mentioned above, kept as the record behind the bug report.
