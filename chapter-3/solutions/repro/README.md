# Compiler bug reproducer — not an exercise solution

These files are **not** a solution to any exercise. They are the minimal
reproducer for a Vehicle bug found while writing the Exercise #6 solution
(`../fmnist-robustness-euclidean.vcl`, Euclidean instead of L-infinity balls).

`tiny.onnx` is a two-input, one-output network (a single `MatMul`). The four
specifications are identical apart from one constraint, and **all four
type-check**. They differ only when compiled to queries:

| file | constraint | `vehicle compile queries` |
| --- | --- | --- |
| `linear.vcl` | `x ! 0 <= epsilon` | compiles |
| `reduceadd-linear.vcl` | `reduceAdd x <= epsilon` | compiles |
| `quadratic.vcl` | `x ! 0 * x ! 0 <= epsilon` | clean "non-linear constraint" error |
| `reduceadd.vcl` | `reduceAdd (x * x) <= epsilon` | `developerError` crash |

So `reduceAdd` is fine on its own, and a non-linearity on its own is diagnosed
properly. Only a non-linearity *inside* a tensor reduction crashes, in
`Vehicle.Compile.Unblock` via `PurifyAssertion`. Reproduce with:

```sh
vehicle compile queries --format VNNLibQueries \
  --specification reduceadd.vcl \
  --network f:tiny.onnx \
  --parameter epsilon:0.1 \
  --output out
```

Observed with Vehicle 0.27.1. The crash is solver-independent: it happens during
query compilation, before any solver runs.
