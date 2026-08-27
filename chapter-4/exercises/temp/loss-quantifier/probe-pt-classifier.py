"""Does pt_classifier.py's training loop actually invoke the FGSM/PGD search?

Counts, without altering the script: how many times the sampler is asked for a loss,
and how many times torch.autograd.grad is called inside it. DefaultPyTorchSampler only
takes FGSM steps when the loss requires grad, so a zero autograd count would mean the
adversarial search is inert and the "search" is plain random sampling.
"""
import torch
from vehicle_lang.loss._pytorch.samplers import DefaultPyTorchSampler

stats = {"get_loss": 0, "autograd_grad": 0, "requires_grad_true": 0,
         "requires_grad_false": 0, "num_samples": None, "num_steps": None}

_orig_get_loss = DefaultPyTorchSampler.get_loss
def counted_get_loss(self, dims, lower_bound, upper_bound, search_lambda):
    stats["get_loss"] += 1
    stats["num_samples"], stats["num_steps"] = self.num_samples, self.num_steps
    try:
        probe = search_lambda(lower_bound.detach().clone().requires_grad_(True))
        if getattr(probe, "requires_grad", False): stats["requires_grad_true"] += 1
        else: stats["requires_grad_false"] += 1
    except Exception:
        pass
    return _orig_get_loss(self, dims, lower_bound, upper_bound, search_lambda)
DefaultPyTorchSampler.get_loss = counted_get_loss

_orig_grad = torch.autograd.grad
def counted_grad(*a, **k):
    stats["autograd_grad"] += 1
    return _orig_grad(*a, **k)
torch.autograd.grad = counted_grad

src = open("pt_classifier.py").read()
src = src.replace("SUBSET_SIZE = 1024", "SUBSET_SIZE = 64")   # one batch
src = src.replace("num_epochs = 5", "num_epochs = 1")
try:
    exec(compile(src, "pt_classifier.py", "exec"), {"__name__": "__main__"})
except Exception as e:
    import traceback; traceback.print_exc()

print("\n=== sampler instrumentation ===")
print(f"  sampler configured with num_samples={stats['num_samples']}, num_steps={stats['num_steps']}")
print(f"  sampler get_loss calls        : {stats['get_loss']}")
print(f"  torch.autograd.grad calls     : {stats['autograd_grad']}")
print(f"  probe: loss requires_grad True: {stats['requires_grad_true']}")
print(f"  probe: loss requires_grad False: {stats['requires_grad_false']}")
if stats["get_loss"]:
    exp = stats["get_loss"] * (stats["num_samples"] or 0) * (stats["num_steps"] or 0)
    print(f"  FGSM steps expected if active : {exp}")
    print(f"  VERDICT: {'FGSM/PGD IS running' if stats['autograd_grad'] > 0 else 'FGSM/PGD is NOT running (zero-gradient path)'}")
