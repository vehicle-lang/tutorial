# Chapter 5 exercises

The exercises use Vehicle 0.26.1. Run commands from this directory so the
relative path to the model remains valid.

Exercises 3 and 4 are independent: begin each one from a fresh copy of
`windController-exercises.vcl` rather than carrying the record refactor from
Exercise 3 into Exercise 4.

## Exercises 1 and 2

Use the clean files and commands in `../chapter-code`. These exercises generate
Rocq files locally; no generated files are supplied here.

## Exercise 3: records

Copy `windController-exercises.vcl`, then replace the raw input and output
tensors with `@tensor` records. Keep the existing `normalise` calculation
unchanged: this exercise changes the Vehicle data representation, not the ONNX
model or its preprocessing.

Verify the refactored specification with:

```bash
vehicle verify \
  --specification windController-exercises.vcl \
  --network controller:../chapter-code/controller.onnx \
  --verifier Marabou \
  --cache controller-result
```

## Exercise 4: centred sensor stability

Start again from the tensor-based working copy. Define `sensorNearCenter` so
that:

- the current sensor reading is between `-0.2` and `0.2` metres;
- the difference between the current and previous readings is between `-0.2`
  and `0.2` metres; and
- the controller's velocity adjustment is between `-0.5` and `0.5`
  metres/second.

The change constraint describes something different from the position bound.
For example, a previous reading of `-0.2` and a current reading of `0.2` are both
near the centre, but the reading has jumped by `0.4` metres. That is not a
stable pair under the `0.2`-metre change limit. Because the controller receives
both readings, it may react more strongly to such a jump than to two readings
that are close together.

You must also write `-0.4 <= previousSensor <= 0.4` in the premise passed to
Vehicle. This is not an extra assumption. If the current reading is at most
`0.2` metres from zero and it differs from the previous reading by at most
`0.2` metres, then the previous reading can be at most `0.2 + 0.2 = 0.4`
metres from zero:

```text
previousSensor = currentSensor - (currentSensor - previousSensor)
|previousSensor| <= |currentSensor| + |currentSensor - previousSensor|
                 <= 0.2 + 0.2 = 0.4
```

Vehicle 0.26.1 needs this finite bound written in this instance for each
network input, even when it follows from the other conditions.

After adding the property, run:

```bash
vehicle verify \
  --specification windController-exercises.vcl \
  --property sensorNearCenter \
  --network controller:../chapter-code/controller.onnx \
  --verifier Marabou \
  --cache controller-result

vehicle export \
  --target Rocq \
  --cache controller-result \
  --output WindControllerSpec.v
```

Model solutions and the focused Rocq proof are in `../solutions`.
The optional parameterised challenge also has a model solution there, together
with parameter values that have been checked to pass and fail on the supplied
network.
