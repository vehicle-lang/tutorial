"""Property-driven training with the additive QLL / Capucci differentiable logic.

A copy of pt_classifier.py that swaps Vehicle's default logic for a custom one. The
logic itself is declared in fmnist-robustness-capucci.vcl as `qllAdditive`; Vehicle
only accepts a DifferentiableTensorLogic from a specification, so it is selected here
by name with CustomDifferentiableLogic rather than written in Python.

Two things differ from pt_classifier.py beyond the logic:

  * epsilon is 0.02 rather than 0.005. At 0.005 the vanilla network is already
    provably robust around 36 of the 37 images it classifies correctly, leaving one
    image of headroom -- too little to show that any training method helps. At 0.02
    the same network manages only 21 of 37. See the chapter-code README.
  * losses are reported once per epoch rather than once per step.
"""
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.onnx
import vehicle_lang as vcl
from vehicle_lang.loss import pytorch as loss_pt

BATCH_SIZE = 64
SUBSET_SIZE = 1024 # ensure SUBSET_SIZE mod BATCH_SIZE = 0
EPSILON = 0.02
ALPHA = 0.5        # weight on the task loss; (1 - ALPHA) weights the constraint loss.
                   # Both terms are needed. At ALPHA = 0 the constraint loss is the only
                   # signal, and since a constant classifier is trivially robust the
                   # optimiser takes that route: three epochs of it drove training
                   # accuracy to 7.8% while the constraint loss kept improving. The task
                   # loss is what rules out that degenerate solution.
CLAMP_CONSTRAINT = True   # clamp each image's constraint loss at 0 before averaging.
                          # In this logic trueElement is -infinity, so the constraint
                          # loss is unbounded below and an already-satisfied image keeps
                          # yielding gradient for being "more true". A constant classifier
                          # is perfectly robust, so that gradient points straight at the
                          # degenerate solution: without the clamp, training collapses to
                          # chance accuracy at ALPHA = 0 and at ALPHA = 0.5 alike.
                          # Clamping at 0 makes a satisfied constraint contribute nothing,
                          # leaving only violated images to push the weights.
OUT_DIR = "pdt-experiment/onnx_models"

# ToTensor alone puts pixels in [0.0, 1.0]. We deliberately do not normalise:
# the specification says `validImage x = forall i j . 0 <= x ! i ! j <= 1`, and
# the .idx datasets handed to the verifier hold pixels in that same range. Adding
# a Normalize step here would train the network on a different input space from
# the one it is later verified on, so `epsilon` would denote a different sized
# perturbation on each side of the pipeline.
transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
train_subset = Subset(train_data, range(SUBSET_SIZE))
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

# Load the specification, interpreting its connectives with the custom logic declared
# in the .vcl file. The string must match the name of the DifferentiableTensorLogic
# declaration exactly.
spec = loss_pt.load_specification(
    "fmnist-robustness-capucci.vcl",
    logic=vcl.CustomDifferentiableLogic("qllAdditive")
)

constraint_loss_fn = spec["robust"]

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

def network(x: torch.Tensor) -> torch.Tensor:
    return model(x.reshape(1, 1, 28, 28)).reshape(10)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
cross_entropy = nn.CrossEntropyLoss()

num_epochs = 5

print(f"property-driven training with the qllAdditive logic | epsilon {EPSILON} | alpha {ALPHA}")

for epoch in range(num_epochs):
    task_total = constraint_total = blended_total = 0.0
    correct = seen = steps = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        logits = model(images)
        task_loss = cross_entropy(logits, labels)

        constraint_loss = constraint_loss_fn(
            n=BATCH_SIZE,
            classifier=network,
            epsilon=torch.tensor(EPSILON),
            trainingImages=images.squeeze(1),
            trainingLabels=labels
        )
        constraint_loss = torch.stack(constraint_loss)
        if CLAMP_CONSTRAINT:
            # per-image, not after the mean: otherwise one deeply-satisfied image can
            # offset a violated one and the batch looks compliant when it is not
            constraint_loss = constraint_loss.clamp(min=0.0)
        constraint_loss = constraint_loss.mean()

        total_loss = ALPHA * task_loss + (1 - ALPHA) * constraint_loss
        total_loss.backward()
        optimizer.step()

        task_total += task_loss.item()
        constraint_total += constraint_loss.item()
        blended_total += total_loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.numel()
        steps += 1

    print(
        f"Epoch: {epoch + 1}, "
        f"task loss: {task_total / steps:.4f}, "
        f"constraint loss: {constraint_total / steps:.4f}, "
        f"blended: {blended_total / steps:.4f}, "
        f"train accuracy: {100 * correct / seen:.1f}%"
    )

model.eval()
os.makedirs(OUT_DIR, exist_ok=True)
onnx_path = os.path.join(OUT_DIR, "pdt_capucci_classifier.onnx")

torch.onnx.export(
    model,
    torch.randn(1, 1, 28, 28),
    onnx_path,
    input_names= ["input"],
    output_names=["output"],
    external_data=False, # required for Marabou verification
)

print(f"Saved to {onnx_path}")
