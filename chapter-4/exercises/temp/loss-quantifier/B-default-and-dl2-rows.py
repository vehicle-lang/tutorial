# Produced the "Vehicle default" and "DL2" rows. Run from chapter-4/chapter-code/.
import torch, torch.nn as nn, vehicle_lang as vcl
from vehicle_lang.loss import pytorch as loss_pt
torch.manual_seed(0)
model = nn.Sequential(nn.Flatten(), nn.Linear(784,64), nn.ReLU(),
                      nn.Linear(64,32), nn.ReLU(), nn.Linear(32,10))
model.eval()
def net(x): return model(x.reshape(1,1,28,28)).reshape(10)
imgs = torch.rand(4,28,28); labs = torch.tensor([0,1,2,3])
for name, logic, path in [
    ("Vehicle default", vcl.VehicleDifferentiableLogic(), "fmnist-robustness.vcl"),
    ("DL2",             vcl.DL2DifferentiableLogic(),     "fmnist-robustness.vcl"),
]:
    fn = loss_pt.load_specification(path, logic=logic)["robust"]
    for eps in (0.0, 0.02, 0.5, 2.0):
        with torch.no_grad():                  # <-- the flaw, see below
            v = torch.stack(fn(n=4, classifier=net, epsilon=torch.tensor(eps),
                               trainingImages=imgs, trainingLabels=labs))
        print(f"{name} eps {eps} mean {v.mean().item():+.6f}")
