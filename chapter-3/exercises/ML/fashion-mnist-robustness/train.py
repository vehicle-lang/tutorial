
#########################
## Construct the model ##
#########################

import torch
torch.cuda.is_available()

model = torch.nn.Sequential(
    torch.nn.Linear(1, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

def network(x: torch.Tensor) -> torch.Tensor:
    return model(x.reshape(1, 1)).reshape(1)

###############################
## Generate the Vehicle loss ##
###############################

import vehicle_lang as vcl
from vehicle_lang.loss import pytorch as loss_pt


declarations = loss_pt.load_specification(
    "spec.vcl",
    logic=vcl.DifferentiableLogic.Vehicle,
    #logic=vcl.DifferentiableLogic.Custom(cappuci)
)

constraint_loss_fn = declarations["output_bounded"]

#######################
## Train the network ##
#######################

alpha = 0.5  # Blend task and constraint losses

for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()

        preds = model(x_batch)
        task_loss = torch.nn.functional.mse_loss(preds, y_batch)

        constraint_loss = constraint_loss_fn(network)
        total_loss = alpha * task_loss + (1.0 - alpha) * constraint_loss

        total_loss.backward()
        optimizer.step()
