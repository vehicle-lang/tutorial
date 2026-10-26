# Running the Wind Controller example

The controller and its verification problem are described in the
[chapter README](../README.md). This file covers the files in this directory and
the Vehicle 0.26.1 commands used by Chapter 5.

## Input files

- `controller.onnx` is the neural network controller.
- `windController.vcl` is the tensor-based safety specification.

The specification accepts current and previous sensor readings in metres,
normalises them to the network's input range, and checks the controller's
velocity adjustment.

## Verify with Marabou

Run these commands from this directory:

```bash
vehicle verify \
  --specification windController.vcl \
  --network controller:controller.onnx \
  --verifier Marabou \
  --cache controller-result
```

The expected result is that Vehicle verifies the property `safe` and writes a
verification cache to `controller-result`.

## Export to Rocq

Exporting from the cache produces a validated lemma:

```bash
vehicle export \
  --target Rocq \
  --cache controller-result \
  --output WindControllerSpec.v
```

Compiling directly from the specification instead produces an axiom:

```bash
vehicle compile itp \
  --target Rocq \
  --specification windController.vcl \
  --network controller:controller.onnx \
  --output WindControllerSpecAxiom.v
```

The cache and generated Rocq files are local build products and are not part of
the supplied chapter code.

## Full system-proof references

The chapter shows how the verified neural-network property can be used inside a
larger proof that the car remains on the road. The focused Exercise 4 proof is
provided in `../solutions/rocq`; the complete system-level developments remain
in their original repositories:

- [Rocq proof](https://github.com/vehicle-lang/vehicle/blob/v0.26.1/examples/windController/rocqProof/SafetyProof.v)
- [Isabelle proof](https://github.com/vehicle-lang/vehicle/blob/v0.26.1/examples/windController/isabelleProof/SafetyProof.thy)
- [Agda proof](https://github.com/vehicle-lang/vehicle/blob/v0.26.1/examples/windController-newStyle/agdaProof/SafetyProof.agda)
- [Imandra proof](https://github.com/imandra-ai/imandra-vehicle)

These links are references to the original proof shown in the chapter rather than solutions to
the exercises.
