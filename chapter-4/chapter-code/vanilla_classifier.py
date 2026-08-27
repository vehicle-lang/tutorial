import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.onnx

MEAN, STD = 0.2860, 0.3530 # mean and std dev of Fashion MNIST
BATCH_SIZE = 64
SUBSET_SIZE = 1024 # ensure SUBSET_SIZE mod BATCH_SIZE = 0

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,))
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
alpha = 0.5

for epoch in range(num_epochs):
    for step, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(images)
        loss = cross_entropy(logits, labels)

        print(f"Step: {step},   Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()

    print(
        f"Epoch: {epoch + 1}, "
        f"Total loss: {loss.item():.4f}"
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
