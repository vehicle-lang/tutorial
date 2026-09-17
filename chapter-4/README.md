# Property-driven training

This chapter reuses the data sets introduced in the previous two chapters, so there
is no new data to describe. Training uses **Fashion MNIST**, the ten-class clothing
counterpart to MNIST described in the
[Chapter 3 README](../chapter-3/README.md), and the property being trained for is the
same classification robustness verified there. The one exception is Exercise #6, which
returns to Chapter 2's ACAS Xu networks and trains for their ten collision-avoidance
properties, a setting with no data set at all.

What changes is the direction of travel. Chapters 2 and 3 took a fixed network and
asked whether it satisfied a specification. Here the specification is compiled into a
*loss function* instead, so that the network is trained to satisfy it in the first
place. The verification step at the end is the same as in Chapter 3, applied to a
network we have just trained ourselves.

## The argument of the chapter, in numbers

The chapter code carries out two experiments on the same small network, and the
second answers the question the first raises.

1. **Train the ordinary way, then verify.** A network trained on cross-entropy alone
   reaches 99.5% training accuracy, yet at `epsilon 0.02` Marabou can prove only 22 of
   the 50 held-out Chapter 3 images robust, and training it for longer makes that number
   *fall*. Once the training data is fitted, optimising cross-entropy further only
   narrows the margins around the training points.

2. **Continue that network with the specification in the objective.** Ten epochs of
   property-driven training, blending cross-entropy with the loss Vehicle compiles from
   the robustness specification, raise the count to 27 of 50, with all ten intermediate
   snapshots above the baseline and no loss of accuracy: the final network classifies 40
   of the 50 images correctly against 38 before.

| | correctly classified | provably robust at `epsilon 0.02` |
| --- | ---: | ---: |
| cross-entropy only, 100 epochs | 38/50 | 22/50 |
| plus 10 epochs of property-driven training | 40/50 | **27/50** |

Both results, and everything needed to reproduce them, are in `chapter-code`.

## Software versions

The property-driven training in this chapter needs **Vehicle 0.28.0 or later**. Versions
up to and including 0.27.1 have a bug in the adversarial search behind `forall` that
inverts the compiled loss, so training with them makes networks *less* robust; the
chapter code records what that looked like. Check with `vehicle --version`, and upgrade
with `pip install --upgrade vehicle-lang` if necessary. Verification uses Marabou 2.0.0,
as in Chapter 3.

## Where things are

- `chapter-code` - the specifications, the training and verification scripts, the
  trained networks and the solver transcripts, with a README that walks through both
  experiments step by step and lists the Python libraries required.
- `exercises` - the exercises for this chapter and what you need to attempt them.
  `exercises/acas2` holds the starting material for Exercise #6, the ACAS Xu benchmark:
  the ten-property specification, three networks, and the scripts that split the
  specification and run Marabou.
- `solutions` - sample solutions and expected results. `solutions/acas2` is the complete,
  self-contained worked solution of Exercise #6: training script, one-property
  specifications, every snapshot, all traces and solver transcripts, and a README that
  records each step.
