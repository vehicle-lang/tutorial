# Produced the "Capucci qllAdditive" row. Run from chapter-4/chapter-code/.
import torch, torch.nn as nn, vehicle_lang as vcl
from vehicle_lang.loss import pytorch as loss_pt
torch.manual_seed(0)
spec = loss_pt.load_specification("fmnist-robustness-capucci.vcl",
                                  logic=vcl.CustomDifferentiableLogic("qllAdditive"))
fn = spec["robust"]
model = nn.Sequential(nn.Flatten(), nn.Linear(784,64), nn.ReLU(),
                      nn.Linear(64,32), nn.ReLU(), nn.Linear(32,10))
model.eval()
def net(x): return model(x.reshape(1,1,28,28)).reshape(10)
imgs = torch.rand(4,28,28); labs = torch.tensor([0,1,2,3])
for eps in (0.0, 0.005, 0.02, 0.1, 0.5, 2.0):
    with torch.no_grad():                      # <-- the flaw, see below
        v = torch.stack(fn(n=4, classifier=net, epsilon=torch.tensor(eps),
                           trainingImages=imgs, trainingLabels=labs))
    print(f"eps {eps} mean {v.mean().item():+.5f}")
