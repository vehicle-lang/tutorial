import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.onnx
import vehicle_lang as vcl
from vehicle_lang.loss import pytorch as loss_pt

SEED = 0
torch.manual_seed(SEED)
generator = torch.Generator().manual_seed(SEED)

MEAN, STD = 0.2860, 0.3530 # mean and std dev of Fashion MNIST
BATCH_SIZE = 64
SUBSET_SIZE = 1024 # ensure SUBSET_SIZE mod BATCH_SIZE = 0

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
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

# load Vehicle specification + loss function

spec = loss_pt.load_specification(
    "fmnist-robustness.vcl",
    logic=vcl.VehicleDifferentiableLogic()
)

constraint_loss_fn = spec["robust"]

class Normalize(nn.Module):
    def __init__(self, mean: float, std: float):
        super().__init__()
        self.mean = mean
        self.std = std
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
 
model = nn.Sequential(
    Normalize(MEAN, STD),
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
alpha = 0.5

for epoch in range(num_epochs):
    running_total_loss, correct, seen = 0.0, 0, 0

    for step, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(images)
        task_loss = cross_entropy(logits, labels)

        constraint_loss = constraint_loss_fn(
            n=BATCH_SIZE,
            classifier=network,
            epsilon=torch.tensor(0.005),
            trainingImages=images.squeeze(1),
            trainingLabels=labels
        )

        constraint_loss = torch.stack(constraint_loss).mean()
        total_loss = alpha * task_loss + (1 - alpha) * constraint_loss
        print(
            f"Step {step}:\n\t"
            f"task loss:        {task_loss.item():.4f}\n\t"
            f"constraint loss:  {constraint_loss.item():.4f}\n\t"
            f"total loss:       {total_loss.item():.4f}"
        )
        total_loss.backward()
        optimizer.step()

        running_total_loss += total_loss.item() * labels.numel()
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.numel()

    print(
        f"Epoch: {epoch + 1}, "
        f"mean total loss: {running_total_loss / seen:.4f}, "
        f"train accuracy: {100 * correct / seen:.1f}%"
    )

model.eval()
input_tensor = torch.randn(1,1,28,28)

path = "pdt-experiment/onnx_models/pdt_classifier.onnx"
os.makedirs(os.path.dirname(path), exist_ok=True)

torch.onnx.export(
    model,
    input_tensor,
    path,
    input_names= ["input"],
    output_names=["output"],
    external_data=False, # required for Marabou verification
)

print("Saved to " + path)