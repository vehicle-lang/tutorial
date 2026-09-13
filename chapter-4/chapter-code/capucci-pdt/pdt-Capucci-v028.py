"""Run 3: repeat run 1 unchanged, under Vehicle 0.28.0.

This is pdt-Capucci.py with three differences, none of them to the experiment itself:

  * outputs go to traces-v028/ and capucci-models-v028/, so run 1's folders -- the
    evidence for the upstream bug report -- are left untouched;
  * snapshots are named capucci_v028_eNN.onnx for the same reason;
  * PyTorch is pinned to four threads, because the corrected sampler is slower per image
    and was seen to thrash at eight.

Every setting that defines the experiment is identical to run 1: the same starting
checkpoint, specification, logic, epsilon, alpha, batch size, image subset, optimiser,
epoch count, and no clamping or gradient clipping. The only thing that has changed is the
Vehicle version, so any difference in outcome is attributable to the fix.

What the fix was: in 0.27.1 the PyTorch sampler's FGSM step moved in the *descent*
direction, so the search behind `forall perturbation` found the least violating
perturbation instead of the most violating one. 0.28.0 reverses the step. The compiled
constraint loss now increases with epsilon, as a worst-case quantifier must, and on the
starting network it is zero for images that satisfy the property and positive for those
that do not. The -infinity divergence seen in run 1 was a consequence of the bug.

    total = ALPHA * cross_entropy + (1 - ALPHA) * constraint_loss

Outputs (traces-v028/ is gitignored, the snapshots are kept):

    traces-v028/per_epoch.csv            epoch, constraint loss, cross-entropy, accuracy, seconds
    capucci-models-v028/capucci_v028_eNN.onnx   the network after each epoch
    traces-v028/train_log.txt            whatever the run printed
"""

import csv
import os
import time

import numpy as np
import onnx
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from onnx import numpy_helper
from torch.utils.data import DataLoader, Subset

import vehicle_lang as vcl
from vehicle_lang.loss import pytorch as loss_pt

torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
TRACES = os.path.join(HERE, "traces-v028")
MODELS = os.path.join(HERE, "capucci-models-v028")

START_MODEL = "vanilla_e100.onnx"
SPEC = "fashionRobustness-capucci.vcl"
TEST_IMAGES = "0-49Images.idx"
TEST_LABELS = "0-49Labels.idx"

BATCH_SIZE = 64
SUBSET_SIZE = 1024  # ensure SUBSET_SIZE mod BATCH_SIZE == 0
EPSILON = 0.02      # the radius at which the starting network is genuinely vulnerable:
                    # it proves 22 of the 38 images it classifies correctly, leaving 16
ALPHA = 0.4         # weight on the task loss; (1 - ALPHA) weights the constraint loss,
                    # matching the alpha of the objective in the chapter. Unchanged from
                    # run 1 so the two runs differ only in the Vehicle version.
NUM_EPOCHS = 10

def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_idx(path):
    """Minimal IDX reader, so no extra dependency is needed just to score 50 images."""
    import struct

    data = open(path, "rb").read()
    _, dtype_code, ndim = struct.unpack(">HBB", data[:4])
    dims = struct.unpack(">" + "I" * ndim, data[4 : 4 + 4 * ndim])
    dtype = {0x08: np.uint8, 0x0B: np.int16, 0x0C: np.int32,
             0x0D: np.float32, 0x0E: np.float64}[dtype_code]
    payload = np.frombuffer(data[4 + 4 * ndim :], dtype=np.dtype(dtype).newbyteorder(">"))
    return payload.reshape(dims)

# --- the network, and the epoch-100 weights ---------------------------------------
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)

weights = {
    init.name: numpy_helper.to_array(init)
    for init in onnx.load(os.path.join(HERE, START_MODEL)).graph.initializer
}
state = model.state_dict()
for key in ("1.weight", "1.bias", "3.weight", "3.bias", "5.weight", "5.bias"):
    assert key in weights, f"{START_MODEL} has no initialiser named {key}"
    assert weights[key].shape == tuple(state[key].shape), f"shape mismatch for {key}"
    state[key] = torch.tensor(weights[key])
model.load_state_dict(state)
log(f"vehicle_lang {vcl.VERSION}; torch {torch.__version__}; {torch.get_num_threads()} threads")
log(f"loaded {START_MODEL}: continuing that network, not training from scratch")

# --- data ------------------------------------------------------------------------
# Raw [0,1] pixels, no normalisation: the specification and the .idx datasets handed to
# the verifier both live in that range, so epsilon means the same thing on both sides.
transform = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.FashionMNIST(
    root=os.path.join(HERE, "data"), train=True, download=True, transform=transform
)
train_loader = DataLoader(
    Subset(train_data, range(SUBSET_SIZE)), batch_size=BATCH_SIZE, shuffle=True
)

# --- the loss compiled from the specification -------------------------------------
spec = loss_pt.load_specification(
    os.path.join(HERE, SPEC),
    logic=vcl.CustomDifferentiableLogic("qllAdditive"),
)
constraint_loss_fn = spec["robust"]

def network(x: torch.Tensor) -> torch.Tensor:
    return model(x.reshape(1, 1, 28, 28)).reshape(10)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
cross_entropy = nn.CrossEntropyLoss()

# The 50 images the specification is checked against: the first fifty FashionMNIST *test*
# images, held out of the 1024 trained on. The count classified correctly is a hard ceiling
# on how many can be proved robust.
test_images = torch.tensor(read_idx(os.path.join(HERE, TEST_IMAGES)).astype(np.float32))
test_images = test_images.reshape(-1, 1, 28, 28)
test_labels = torch.tensor(read_idx(os.path.join(HERE, TEST_LABELS)).astype(np.int64))


def correct_on_test():
    was_training = model.training
    model.eval()
    with torch.no_grad():
        n_correct = int((model(test_images).argmax(1) == test_labels).sum())
    if was_training:
        model.train()
    return n_correct

def export(epoch: int) -> str:
    """Snapshot the weights as ONNX so any epoch can be verified afterwards.

    Written to a .part file and renamed, so the verification watcher never sees a
    half-written model: existence of the final name means the file is complete."""
    dst = os.path.join(MODELS, f"capucci_v028_e{epoch:02d}.onnx")
    was_training = model.training
    model.eval()
    torch.onnx.export(
        model,
        torch.randn(1, 1, 28, 28),
        dst + ".part",
        input_names=["input"],
        output_names=["output"],
        external_data=False,  # required for Marabou verification
    )
    os.replace(dst + ".part", dst)
    if was_training:
        model.train()
    return dst

os.makedirs(TRACES, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)
per_epoch = open(os.path.join(TRACES, "per_epoch.csv"), "w", newline="")
writer = csv.writer(per_epoch)
writer.writerow(
    ["epoch", "constraint_loss", "cross_entropy", "blended_loss",
     "train_accuracy", "correct_on_50_test", "seconds"]
)
writer.writerow([0, "", "", "", "", correct_on_test(), ""])
per_epoch.flush()
log(f"starting point classifies {correct_on_test()}/50 of the test images correctly")

log(
    f"{NUM_EPOCHS} epochs | qllAdditive logic | epsilon {EPSILON} | alpha {ALPHA} "
    f"(task) / {1 - ALPHA:.1f} (constraint) | {SUBSET_SIZE} images, "
    f"{SUBSET_SIZE // BATCH_SIZE} steps/epoch"
)

model.train()
for epoch in range(1, NUM_EPOCHS + 1):
    started = time.time()
    con_total = ce_total = blended_total = 0.0
    correct = seen = steps = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        constraint_loss = constraint_loss_fn(
            n=BATCH_SIZE,
            classifier=network,
            epsilon=torch.tensor(EPSILON),
            trainingImages=images.squeeze(1),
            trainingLabels=labels,
        )
        constraint_loss = torch.stack(constraint_loss).mean()

        logits = model(images)
        task_loss = cross_entropy(logits, labels)

        total_loss = ALPHA * task_loss + (1 - ALPHA) * constraint_loss
        total_loss.backward()
        optimizer.step()

        con_total += constraint_loss.item()
        ce_total += task_loss.item()
        blended_total += total_loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.numel()
        steps += 1

        if steps % 4 == 0:
            log(f"  epoch {epoch} step {steps}/{SUBSET_SIZE // BATCH_SIZE}: "
                f"constraint {constraint_loss.item():+.4f} CE {task_loss.item():.4f}")

    seconds = time.time() - started
    on_test = correct_on_test()
    writer.writerow(
        [
            epoch,
            f"{con_total / steps:.6f}",
            f"{ce_total / steps:.6f}",
            f"{blended_total / steps:.6f}",
            f"{correct / seen:.4f}",
            on_test,
            f"{seconds:.1f}",
        ]
    )
    per_epoch.flush()
    export(epoch)
    log(
        f"epoch {epoch:3}: constraint {con_total / steps:+.4f} | "
        f"CE {ce_total / steps:.4f} | blended {blended_total / steps:+.4f} | "
        f"train acc {100 * correct / seen:.1f}% | test {on_test}/50 | {seconds:.0f}s"
    )

per_epoch.close()
log(f"done; {NUM_EPOCHS} snapshots in capucci-models-v028/, statistics in traces-v028/per_epoch.csv")
