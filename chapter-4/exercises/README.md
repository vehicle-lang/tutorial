# Chapter 4 exercises

Exercises #1 to #5 and #7 use the material in `../chapter-code`; Exercise #6 uses the
ACAS Xu material in `acas2/` here. There is nothing extra to download. Read the chapter-code
README first: it walks through both of the chapter's experiments, and Exercises #1 and #2
are essentially those experiments.

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

Both notes are about the Fashion MNIST exercises (#1 to #5); Exercise #6 has its own,
very different, timings.

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

## Exercise #4 (⭑⭑): Normalisation inside the network

Rebuild the training-verification pipeline *with* normalisation, but placed inside the
network as its first layer, so that the inputs the specification and the verifier see are
still raw pixels. Train, export, and verify. What does Marabou say about the exported
network, and why? Fix it by folding the normalisation into the first linear layer before
exporting, check that the folded network computes the same function as the trained one,
and verify again.

Start from `pt_classifier.py`, which already normalises inside the network with a small
`Normalize` module. To see the problem, export `model` directly instead of `export_model`
and run the verification command from the chapter-code README; then put the fold back.
The identity you need is

    W ((x - MEAN)/STD) + b  =  (W/STD) x + (b - (MEAN/STD) * rowsum(W))

and `chapter-code/vanilla-experiment/folded-vanilla-models/fold_normalisation.py` applies
it to a saved ONNX file after the fact, which is the other place the fold can live.

## Exercise #5 (⭑⭑⭑): Training a model from scratch

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

## Exercise #6 (⭑⭑): The ACAS Xu benchmark

This is the first exercise about properties more complex than epsilon-ball robustness.
In Chapter 2 you verified the ten ACAS Xu properties on the networks N_{1,7}, N_{1,8} and
N_{1,9}, and saw that most of them fail: N_{1,8}, for instance, satisfies only Properties 1,
2 and 10, and Properties 6, 7 and 8 are too hard for Marabou to decide within fifteen
minutes. Using the property-driven training of this chapter, re-train one of these networks
and see whether more properties become verifiable.

Two things make this harder than Fashion MNIST. There is **no data set**, so there is no task
loss to hold the network to its job (`ALPHA = 0`), and with a single property that is fatal:
training N_{1,8} on Property 3 alone makes the property hold after one epoch by producing a
network that *never* advises clear-of-conflict anywhere: the degenerate solution the
chapter's objective section warns about, with no task term to rule it out. And the ten
properties are stated with a guarded implication `i != j => ...` that the loss compiler turns
into a constant loss, so the training specification has to state `minimalScore` non-strictly
without the guard, as `fmnist-robustness.vcl` does for the same reason.

The hint that makes it work: split the specification into ten one-property files and train
on them **one property per epoch, repeatedly**, so that the properties counterbalance each
other and no single one gets to reshape the whole network. Start a fresh optimiser each
epoch, or the second visit to a property will diverge.

Everything needed is in `acas2/`: the upstream ten-property specification, the three
networks, `split-spec.py` (the one-property specifications), `pdt-acas2.py` (the sequential
trainer) and `verify-acas2.py` (one Marabou call per property, with a time limit). Its README
records every step of the worked run, including the two failed attempts: N_{1,8} goes from
three verified properties to six after three tours of the ten, while still agreeing with the
original network on 90% of inputs.

**Note 1.** Timings here are unlike the Fashion MNIST exercises: an epoch takes about ten
seconds and Marabou proves or refutes most ACAS Xu properties in seconds, but Properties 6,
7 and 8 do not finish in fifteen minutes and need a cap, and a property left uncapped can run
for hours.

**Note 2.** Properties 6, 7 and 8 are the whole-input-space properties. Their regions cover
nearly the entire valid box (Property 7's is literally all of it), so Marabou has to
case-split over most of the network's 300 ReLUs instead of the few that are active in a
narrow slice like Property 3's. Vehicle also turns each into several queries (Property 6 into
8, because of the disjunction in its antecedent), each of which is a full Marabou run. They
were the hard ones in the original Reluplex paper too, which is why the spec's comments say
7 was tested on N_{1,9} only and 8 on N_{2,9} only.

These are the verdicts you should have obtained in Chapter 2's exercises, for the three
shipped networks on the ten properties, with a 15-minute cap per property (✓ verified,
✗ falsified, t/o = undecided within the cap):

<!-- EXERCISE 6 BASELINE TABLE START -->
| NN | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | summary |
|:----|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:------------------------|
| N_{1,7} | ✓ | ✓ | ✗ | ✗ | ✗ | t/o | t/o | ✗ | ✗ | ✓ | verified {1, 2, 10}; falsified 3, 4, 5, 8, 9; undecided 6, 7 |
| N_{1,8} | ✓ | ✓ | ✗ | ✗ | ✗ | t/o | t/o | t/o | ✗ | ✓ | verified {1, 2, 10}; falsified 3, 4, 5, 9; undecided 6, 7, 8 |
| N_{1,9} | ✓ | ✓ | ✗ | ✗ | ✗ | t/o | t/o | t/o | ✗ | ✓ | verified {1, 2, 10}; falsified 3, 4, 5, 9; undecided 6, 7, 8 |
<!-- EXERCISE 6 BASELINE TABLE END -->

Note that some properties remain unverified for all three networks: Properties 3, 4, 5 and 9
are falsified on every one, 6 and 7 are undecided on every one, and 8 is falsified on one
network and undecided on the other two. This may suggest that
these properties are difficult or impossible to infer from data: the networks were trained on
a data set sampled from a controller, and a property that no network has picked up from the
data is a property the data does not exhibit clearly enough, or a property the network
architecture cannot represent, which property-driven training may be able to supply.

## Exercise #7 (⭑⭑⭑): Training for properties more complex than epsilon-ball robustness: the Iris data set

Recall the code and Vehicle specifications from Chapter 2, Exercise #4, the Iris model.
For any of your properties that Marabou falsified, run property-driven training and
measure whether they become verifiable as a result.

The material is in `../../resources-by-dataset/iris/`: `iris_model.onnx`, the test-set
files `iris_test_data.idx` and `iris_test_label.idx`, the notebook `Iris.ipynb` that
trained the model, and `iris-possible-answers.vcl` with candidate properties. The
training loop from `pt_classifier.py` transfers directly: load the Iris specification
with `load_specification`, name the property you want as the constraint loss, and blend it
with the task loss. Keep the heuristics from this chapter in mind, in particular the
choice of `ALPHA`, of the hardness `p` if you use the Capucci logic, and the handling of
normalisation.

Sample solutions and expected results are in `../solutions`.

The `temp/` folder here is not exercise material. It holds the reproducers for the
Vehicle 0.27.1 bug mentioned above, kept as the record behind the bug report.
