"""Rewrite a checkpoint so it takes raw [0,1] pixels instead of normalised ones.

The training pipeline normalised its inputs, (x - MEAN)/STD, while the .idx files
handed to Marabou hold raw pixels in [0,1]. That made 'epsilon' mean two different
sized cubes either side of the pipeline, and made the specification's
`validImage x = forall i j . 0 <= x ! i ! j <= 1` false of every input the loss saw.

Normalisation is affine and the first layer is affine, so the two compose exactly:

    W ((x - MEAN)/STD) + b  =  (W/STD) x + (b - (MEAN/STD) sum_j W_ij)

Folding it in therefore costs nothing in accuracy, adds no ONNX operators (so
Marabou needs no new support), and leaves a network whose input space is the same
[0,1] space the specification and the datasets talk about.
"""
import sys, numpy as np, onnx, torch, torch.nn as nn
from onnx import numpy_helper

MEAN, STD = 0.2860, 0.3530

def build():
    return nn.Sequential(nn.Flatten(), nn.Linear(784,64), nn.ReLU(),
                         nn.Linear(64,32), nn.ReLU(), nn.Linear(32,10))

def load(path):
    w = {i.name: numpy_helper.to_array(i) for i in onnx.load(path).graph.initializer}
    m = build(); sd = m.state_dict()
    for k in ("1.weight","1.bias","3.weight","3.bias","5.weight","5.bias"):
        sd[k] = torch.tensor(w[k])
    m.load_state_dict(sd); m.eval(); return m

def fold(m):
    """Return an equivalent network over raw [0,1] inputs."""
    f = build(); f.load_state_dict(m.state_dict())
    with torch.no_grad():
        W = m[1].weight.detach().clone(); b = m[1].bias.detach().clone()
        f[1].weight.copy_(W / STD)
        f[1].bias.copy_(b - (MEAN / STD) * W.sum(dim=1))
    f.eval(); return f

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    orig   = load(src)
    folded = fold(orig)

    # equivalence check: folded(raw) must equal orig(normalised(raw)) for raw in [0,1]
    torch.manual_seed(0)
    raw  = torch.rand(512,1,28,28)                 # the range Marabou will explore
    norm = (raw - MEAN) / STD
    with torch.no_grad():
        a = orig(norm); c = folded(raw)
    err = (a - c).abs().max().item()
    agree = (a.argmax(1) == c.argmax(1)).float().mean().item()
    print(f"  max |orig(normalised) - folded(raw)| = {err:.3e}")
    print(f"  argmax agreement                    = {100*agree:.2f}%")
    assert err < 1e-4, "fold is not equivalent"

    torch.onnx.export(folded, torch.randn(1,1,28,28), dst,
                      input_names=["input"], output_names=["output"], external_data=False)
    print(f"  wrote {dst}")
