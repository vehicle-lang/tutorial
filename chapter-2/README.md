# ACAS Xu

ACAS Xu is a collection of 45 neural networks that together make up a collision
avoidance system for autonomous unmanned aircraft. Each network takes five
measurements of the aircraft's position and speed relative to an intruder, and
scores five possible advisories: clear-of-conflict, weak left, weak right,
strong left and strong right. The advisory with the lowest score is the one the
system issues.

The partial verification of the system was first described in the seminal
[Reluplex paper](https://arxiv.org/abs/1702.01135), which lists ten properties
the networks are expected to satisfy. This example demonstrates how the entire
specification, consisting of all 10 properties, can be written in a single file.
Unlike the equivalent low-level Marabou queries, the specification is written at
a high level and is understandable by a non-expert.

## Where things are

- `chapter-code` - the specification, the networks, and the commands to run
  them. See its README.
- `exercises` - the exercises for this chapter, and what you need to solve them.
- `solutions` - sample solutions.
