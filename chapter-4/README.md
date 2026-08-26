# Property-driven training

This chapter reuses the data sets introduced in the previous two chapters, so there
is no new data to describe. Training uses **Fashion MNIST**, the ten-class clothing
counterpart to MNIST described in the
[Chapter 3 README](../chapter-3/README.md), and the property being trained for is the
same classification robustness verified there.

What changes is the direction of travel. Chapters 2 and 3 took a fixed network and
asked whether it satisfied a specification. Here the specification is compiled into a
*loss function* instead, so that the network is trained to satisfy it in the first
place. The verification step at the end is the same as in Chapter 3, applied to a
network we have just trained ourselves.

## Where things are

- `chapter-code` - the specification, the training scripts, and the commands to run
  them. See its README, which also lists the Python libraries required.
- `exercises` - the exercises for this chapter.
- `solutions` - sample solutions.
