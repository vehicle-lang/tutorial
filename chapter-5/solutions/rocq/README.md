# Focused Rocq solution

`CentredStabilityProof.v` proves the exercise theorem only. It deliberately
does not duplicate the full Wind Controller system proof from the Vehicle
repository.

The generated `WindControllerSpec.v`, verification cache, `Makefile`, and Rocq
build products are not included. Generate them locally so that
`vehicle_validate` records paths that are valid on your machine.

## Prerequisites

- Vehicle 0.26.1;
- Marabou 2.0.0;
- Rocq 9 or later and Dune 3.13 or later; and
- the `vehicle-rocq` package from the Vehicle 0.26.1 source tree, installed in
  a dedicated opam switch. Its opam file pins the compatible MathComp,
  Finmap, and Analysis commits.

For example, from the Vehicle source tree:

```bash
opam switch create vehicle-0.26.1 5.4.1
eval "$(opam env --switch=vehicle-0.26.1)"
opam repo add rocq-released https://rocq-prover.org/opam/released
opam install -y ./vehicle-rocq
```

## Generate and check the proof

Run the following commands from this `solutions/rocq` directory. Keeping the
same working directory for verification and Rocq compilation lets
`vehicle_validate` resolve the cache resources correctly.

```bash
vehicle verify \
  --specification ../windController-centred-stability-solution.vcl \
  --property sensorNearCenter \
  --network controller:../../chapter-code/controller.onnx \
  --verifier Marabou \
  --cache controller-result

vehicle export \
  --target Rocq \
  --cache controller-result \
  --output WindControllerSpec.v

rocq makefile -f _CoqProject -o Makefile
make
```

The exported `sensorNearCenter` lemma includes
`|previousSensor| <= 0.4` because Vehicle 0.26.1 needs an explicit finite bound
for both network inputs. The exercise theorem should require only the two
meaningful stability assumptions. Therefore `controllerCenterBound` first
proves the tool-facing bound from `|currentSensor| <= 0.2` and
`|currentSensor - previousSensor| <= 0.2`, and then applies the generated
lemma. The proof step shows that no third physical assumption has been added.
