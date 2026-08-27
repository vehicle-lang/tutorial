import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.onnx

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
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
cross_entropy = nn.CrossEntropyLoss()

num_epochs = 5

for epoch in range(num_epochs):
    running_loss, correct, seen = 0.0, 0, 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.numel()
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.numel()

    print(
        f"Epoch: {epoch + 1}, "
        f"mean loss: {running_loss / seen:.4f}, "
        f"train accuracy: {100 * correct / seen:.1f}%"
    )

model.eval()
input_tensor = torch.randn(1,1,28,28)

torch.onnx.export(
    model,
    input_tensor,
    "onnx_models/vanilla_classifier.onnx",
    input_names= ["input"],
    output_names=["output"],
    external_data=False, # required for Marabou verification
)

print("Saved to onnx_models/vanilla_classifier.onnx")
