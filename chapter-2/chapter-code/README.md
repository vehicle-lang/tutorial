# Running the ACAS Xu example

The data set and the verification problem are described in the
[chapter README](../README.md). This file covers the files in this folder and
how to run them.

## Input files

- `acasXu.vcl` - the specification describing the desired behaviour.

- `acasXu_1_7.onnx`, `acasXu_1_8.onnx`, `acasXu_1_9.onnx` - 3 out of the 45
  networks. The remainder can be found
  [here](https://github.com/NeuralNetworkVerification/Marabou/tree/master/resources/onnx/acasxu).

## Verifying using Marabou

The following command verifies `property3` for the network `acasXu_1_7.onnx`:

```bash
vehicle verify \
  --specification acasXu.vcl \
  --solver Marabou \
  --network acasXu:acasXu_1_7.onnx \
  --property property3
```

The same property can be verified for the other two networks in the folder. The
remaining properties apply to other network components.

Omitting `--property` verifies every property in the specification.

## Output files

The outputs of the above Vehicle commands can be found in the test suite:

- [Automatically generated Marabou queries](https://github.com/vehicle-lang/vehicle/tree/dev/vehicle/tests/golden/specifications/acasXu/Marabou.queries)
