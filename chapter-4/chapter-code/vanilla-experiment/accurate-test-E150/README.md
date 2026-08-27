# accurate-test-E150

The subset of Chapter 3's 50-image FashionMNIST test set that the 150-epoch vanilla
classifier **classifies correctly** -- 37 of the 50 images.

| File | Contents |
| --- | --- |
| `accurateImages.idx` | 37 images, float64 (`0x0E`), dims (37, 28, 28) |
| `accurateLabels.idx` | 37 labels, uint8 (`0x08`), dims (37,) |
| `accurate-indices.txt` | which indices of the original `0-49Images.idx` were kept |

Headers and dtypes match `chapter-3/exercises/FMNIST/idxdata/0-49Images.idx` exactly, so
Vehicle reads them identically. Dropped indices: 8, 11, 17, 21, 23, 25, 26, 29, 40, 42,
43, 48, 49.

## Why this set exists

On the full 50 images a verified count conflates two different things. An image the
network already misclassifies cannot be robust -- `advises perturbedImage label` fails at
zero perturbation -- so it is falsified for reasons that have nothing to do with
robustness. In the earlier experiments most of the falsified column turned out to be
misclassifications rather than robustness failures.

On this set every image is classified correctly, so the ceiling is the whole dataset and
`verified / 37` is a **pure robustness measurement**: the fraction of correctly-classified
images that are provably robust.

## Two limits on how far it can be pushed

**It is specific to the 150-epoch model.** A different checkpoint classifies a different
subset correctly, so using this set to compare checkpoints judges other models on images
chosen to suit epoch 150. To compare checkpoints, either take the intersection of their
correct sets, or keep using the full 50 images and report the decomposition
(correct / verified / misclassified / genuinely non-robust) for each.

**It is easier by construction.** Because the images were selected as ones the model
gets right, `verified / 37` is not comparable with any published `x / 50` figure.

## Using it

```bash
vehicle verify \
  --specification ../../../chapter-3/exercises/FMNIST/fashionRobustness-solution.vcl \
  --network classifier:<model.onnx> \
  --parameter epsilon:0.005 \
  --dataset trainingImages:accurateImages.idx \
  --dataset trainingLabels:accurateLabels.idx \
  --solver Marabou
```

`n` is inferred from the datasets, so nothing else needs changing. Expect roughly
12 minutes and about 14 GB of resident memory for 37 images.
