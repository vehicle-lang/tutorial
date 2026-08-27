# Folded vanilla models

These are the networks from the vanilla experiment, rewritten to take raw `[0,1]`
pixels. They exist to repair a defect, and they are a historical artefact rather than
something the current chapter code produces.

## Why they exist

The chapter's training scripts used to normalise their inputs,
`transforms.Normalize((0.2860,), (0.3530,))`, so the networks learned to read pixels
in roughly `[-0.810, 2.023]`. Verification reads the `.idx` datasets, whose pixels are
in `[0.0, 1.0]`. Every published verification result was therefore measured on a
network operating outside the input scale it had been trained for. See
[normalisation-options.md](../../normalisation-options.md) for the full account.

The scripts no longer normalise, so **models trained from now on do not need
folding**. These files convert the checkpoints that already existed, so the corrected
results could be obtained without retraining.

## What folding does

Normalisation is affine and the first layer is affine, so the two compose exactly:

```
W ((x - MEAN)/STD) + b  =  (W/STD) x + (b - (MEAN/STD) * rowsum(W))
```

The folded network is therefore the *same function* as the original, over a rescaled
input space. It is not an approximation and not a retrained model. Architecture and
ONNX operators are unchanged, so Marabou needs no capability it did not already have.

`fold_normalisation.py` performs the rewrite and checks it. For each model here the
maximum output deviation from the original was at most 7.9e-6, with 100% argmax
agreement over 512 random inputs, verified through onnxruntime as well as PyTorch.

```bash
python3 fold_normalisation.py <normalised-model.onnx> <folded-model.onnx>
```

## The files

| File | Folded from | Notes |
| --- | --- | --- |
| `vanilla_e100_folded.onnx` | epoch-100 checkpoint | 1024 images, cross-entropy only |
| `vanilla_e200_folded.onnx` | epoch-200 checkpoint | " |
| `vanilla_e300_folded.onnx` | epoch-300 checkpoint | " |
| `vanilla_classifier_folded.onnx` | [`../onnx_models/vanilla_classifier.onnx`](../onnx_models/vanilla_classifier.onnx) | the 5-epoch model the chapter script used to produce |

## What they show

Verified with Chapter 3, Exercise #7's command -- epsilon 0.005, the 50 images in
`0-49Images.idx`, which are the first 50 FashionMNIST **test** images. Transcripts are
in [marabou-outputs/](marabou-outputs/).

| epochs | mean loss | train accuracy | correct on the 50 test images | provably robust |
| -----: | --------: | -------------: | ----------------------------: | --------------: |
| 100 | 0.00199 | 100% | 38/50 | 36/50 |
| 200 | 0.00028 | 100% | 38/50 | 36/50 |
| 300 | 0.00008 | 100% | 38/50 | 36/50 |

Two things to take from this.

**The published figures understated these networks.** The same three checkpoints,
measured through the mismatch, scored 28/50, 29/50 and 29/50. Most of what was counted
as a robustness failure was a misclassification: at epoch 200 the mismatched network
classified only 30 of the 50 images correctly, and was proved robust around 29 of those
30. Folded, it classifies 38 correctly and is proved robust around 36.

**The plateau is real, and is now cleaner.** Between epoch 100 and epoch 300 the
training loss falls by a factor of twenty-six while neither accuracy nor provable
robustness moves by a single image. That supports the chapter's claim -- once
cross-entropy is satisfied, further optimisation of it is uninformative about
robustness -- more crisply than the published 28/29/29, which had a one-image wobble.

**But note what the plateau is.** Accuracy sits at 38/50 and provable robustness two
images below it throughout. At epsilon 0.005 the robustness count is very nearly a
restatement of accuracy: only 2 of the 38 eligible images are genuinely non-robust.

That is a property of the radius rather than of these networks. On the re-trained vanilla
network, widening the ball to epsilon 0.02 takes the failures among correctly-classified
images from 1 of 37 to 16 of 37. The chapter's later experiments therefore use 0.02; see
[the chapter-code README](../../README.md) for those figures.

## A caveat on the comparison with Chapter 3's network

Chapter 3's `fashion1l32n.onnx` was trained on raw pixels, so its 40/50 was correct all
along and needs no folding. Comparing it against these models is not like-for-like: it
classifies 46 of the 50 correctly against 38 here, having been trained on far more data.
Per eligible image the vanilla network is the more robust of the two, 36/38 (94.7%)
against 40/46 (87.0%).
