# Chapter 5 model solutions

These solutions target Vehicle 0.26.1 and use the model in `../chapter-code`.
Generated caches and ITP files are intentionally excluded because they contain
machine-specific paths and can be reproduced from the commands below.

## Exercise 1

After cache-backed export, `WindControllerSpec.v` should contain `safe` as a
`Lemma` whose proof invokes `vehicle_validate` with the verification cache.

## Exercise 2

The direct `vehicle compile itp` output in `WindControllerSpecAxiom.v` should
instead declare `safe` as an `Axiom`. This lets a system proof be developed
before neural-network verification, but the result does not carry the
cache-backed validation used by the lemma from Exercise 1.

## Exercise 3

`windController-records-solution.vcl` is a record-based refactor of the original
tensor specification. It uses the same local `controller.onnx` as the chapter
code, so its `normalise` function is unchanged; only tensor indexing is replaced
by named record fields.

```bash
vehicle verify \
  --specification windController-records-solution.vcl \
  --network controller:../chapter-code/controller.onnx \
  --verifier Marabou \
  --cache records-result
```

## Exercise 4

`windController-centred-stability-solution.vcl` adds the concrete
`sensorNearCenter` property. It expresses two meaningful assumptions: the
current reading is within `0.2` metres of the centre and the reading changed by
at most `0.2` metres. These imply the tool-facing bound
`|previousSensor| <= 0.4` by the triangle inequality. The bound is written in
the VCL file because Vehicle 0.26.1 needs an explicit finite range for both
network inputs, note it is not an additional physical assumption.

```bash
vehicle verify \
  --specification windController-centred-stability-solution.vcl \
  --property sensorNearCenter \
  --network controller:../chapter-code/controller.onnx \
  --verifier Marabou \
  --cache controller-result
```

The `rocq` directory contains the focused `controllerCenterBound` proof and the
commands required to generate its imported `WindControllerSpec.v` file.

## Optional parameterised challenge

`windController-parameterised-stability-solution.vcl` replaces the three fixed
bounds with `maxDistance`, `maxSensorChange`, and `maxSpeed`. Its automatically
implied previous-reading bound is:

```text
|previousSensor| <= maxDistance + maxSensorChange
```

For example, the following values reproduce the verified Exercise 4 property:

```bash
vehicle verify \
  --specification windController-parameterised-stability-solution.vcl \
  --property sensorNearCenter \
  --network controller:../chapter-code/controller.onnx \
  --parameter maxDistance:0.2 \
  --parameter maxSensorChange:0.2 \
  --parameter maxSpeed:0.5 \
  --verifier Marabou
```

With Vehicle 0.26.1 and the supplied network this verifies both generated
queries. Tightening only `maxSpeed` to `0.1` does not hold: Marabou returns the
counterexample `x = [-0.1, -0.3]`. 
