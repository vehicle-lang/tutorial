"""Property-driven training of the ACAS Xu networks against Chapter 2, ACAS Xu Spec on all 10 properties.

It is chapter-code/capucci-pdt/pdt-Capucci-v028.py adapted to ACAS Xu: the same
structure (reload an ONNX checkpoint into PyTorch, compile the specification to a loss
with the Capucci `qllAdditive` logic, Adam at 1e-3, export a snapshot after every epoch,
write per-epoch statistics), with these differences forced by the problem:

  * There is no ACAS Xu data set in this repository, so there is no task term:

        total_loss = ALPHA * task_loss + (1 - ALPHA) * constraint_loss   with ALPHA = 0

    i.e. the objective is the compiled property loss and nothing else.

  * An "epoch" cannot be a pass over data. Here it is STEPS_PER_EPOCH gradient steps,
    each on one evaluation of the compiled loss, which itself runs Vehicle's adversarial
    search (10 random starting points, 5 PGD steps each) over the region Property 3
    quantifies over. 16 steps per epoch mirrors the chapter's 1024 images at batch 64.

  * The training specification is acasXu-training.vcl, not acasXu.vcl: `advises` is
    stated non-strictly without the `i != j` guard (the guarded form compiles to the
    constant 0 under Vehicle 0.28.0's loss backend; see the comment in that file and the
    README), and the `qllAdditive` logic is appended. Verification uses acasXu.vcl.

  * The network is rebuilt in PyTorch from the ONNX initialisers (seven Linear layers,
    six ReLUs; the leading `Sub` of the ONNX file subtracts a zero vector and is dropped),
    and the reload is checked against onnxruntime before training starts.

Because nothing holds the network to its original task, two proxies are tracked:

  * violation rate: the fraction of 20,000 inputs drawn uniformly from Property 3's
    region on which clear-of-conflict has the minimal score, i.e. the property fails.
    The compiled loss is an adversarial estimate of the same thing; this is the
    average-case one.
  * agreement: the fraction of 20,000 inputs drawn uniformly from the *whole* valid
    input space on which the trained network gives the same advisory as the original.
    This stands in for task accuracy, which cannot be measured without data.

Usage (run from this folder):

    python3 pdt-acas.py            # all three networks
    python3 pdt-acas.py 1_7        # one network
    python3 pdt-acas.py --spec acasXu-training-p2.vcl --tag p2   # the p = 2 run
    python3 pdt-acas.py 1_7 --steps 2 --epochs 1 --tag smoke   # a quick check

Do not run two copies at once, and do not run it while a `vehicle verify` is running:
concurrent Vehicle invocations have been seen to make the in-process compiler return
empty output.

Outputs (per network NET):

    models/acasXu_NET_pdt_e00.onnx     the starting network re-exported (sanity check)
    models/acasXu_NET_pdt_eNN.onnx     the network after epoch NN
    traces/NET_per_epoch.csv           one row per epoch
    traces/NET_steps.csv               one row per gradient step
    (with --tag TAG the names become acasXu_NET_pdt_TAG_eNN.onnx and NET_TAG_*.csv)
    traces/train_log.txt               the log below, if stdout is redirected there
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper

import vehicle_lang as vcl
from vehicle_lang.loss import pytorch as loss_pt

torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPEC = "acasXu-training.vcl"   # p = 5; acasXu-training-p2.vcl has p = 2. Verification uses acasXu.vcl
LOGIC = vcl.CustomDifferentiableLogic("qllAdditive")   # the Capucci logic, as in the chapter
PROPERTY = "property3"
NETWORKS = ["1_7", "1_8", "1_9"]

NUM_EPOCHS = 3
STEPS_PER_EPOCH = 16
LEARNING_RATE = 1e-3   # as in pdt-Capucci-v028.py
ALPHA = 0.0            # weight on the task loss; there is no data, so no task loss
SEED = 0
N_MONITOR = 20_000

# The constants of acasXu.vcl, repeated here only for the two monitoring proxies.
PI = 3.141592
MIN_IN = torch.tensor([0.0, -PI, -PI, 0.0, 0.0])
MAX_IN = torch.tensor([60261.0, PI, PI, 1200.0, 1200.0])
MEAN_IN = torch.tensor([19791.091, 0.0, 0.0, 650.0, 600.0])
# validInput and directlyAhead and movingTowards, as a box
P3_LO = torch.tensor([1500.0, -0.06, 3.10, 980.0, 960.0])
P3_HI = torch.tensor([1800.0, 0.06, PI, 1200.0, 1200.0])
CLEAR_OF_CONFLICT = 0


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def normalise(x):
    return (x - MEAN_IN) / (MAX_IN - MIN_IN)


# --- the network, rebuilt from the ONNX initialisers ---------------------------------
def load_network(path):
    weights = {i.name: numpy_helper.to_array(i) for i in onnx.load(path).graph.initializer}
    assert not weights["input_AvgImg"].any(), "expected the ONNX Sub to subtract zeros"
    names = [f"Operation_{k}" for k in range(1, 7)] + ["linear_7"]
    layers = []
    for k, name in enumerate(names):
        W = torch.tensor(weights[f"{name}_MatMul_W"].T.copy())  # ONNX is x @ W, torch is W x
        b = torch.tensor(weights[f"{name}_Add_B"])
        linear = nn.Linear(W.shape[1], W.shape[0])
        linear.weight.data = W
        linear.bias.data = b
        layers.append(linear)
        if k < len(names) - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def check_against_onnxruntime(model, path):
    try:
        import onnxruntime as ort
    except ImportError:
        log("onnxruntime not installed; skipping the reload check")
        return
    session = ort.InferenceSession(path)
    probe = (torch.rand(256, 5) * 2 - 1).numpy().astype(np.float32)
    ours = model(torch.tensor(probe)).detach().numpy()
    theirs = np.concatenate(
        [session.run(None, {"input": p.reshape(1, 1, 1, 5)})[0].reshape(1, 5) for p in probe]
    )
    deviation = float(np.abs(ours - theirs).max())
    assert deviation < 1e-4, f"reloaded network deviates from {path} by {deviation}"
    log(f"reloaded weights agree with onnxruntime on 256 probes (max deviation {deviation:.1e})")


def export(model, dst):
    """Snapshot as ONNX with only Gemm and Relu, so Marabou can read it."""
    was_training = model.training
    model.eval()
    torch.onnx.export(
        model,
        torch.randn(1, 5),
        dst + ".part",
        input_names=["input"],
        output_names=["output"],
        external_data=False,
    )
    os.replace(dst + ".part", dst)
    if was_training:
        model.train()


# --- the loss compiled from the specification -----------------------------------------
def load_constraint_loss(spec_path):
    """The in-process compiler has been seen to return empty output when another
    Vehicle process runs at the same time; retry a few times before giving up."""
    last = None
    for attempt in range(1, 6):
        try:
            spec = loss_pt.load_specification(spec_path, logic=LOGIC)
            return spec[PROPERTY]
        except Exception as error:  # noqa: BLE001
            last = error
            log(f"load_specification attempt {attempt} failed: {type(error).__name__}: {error}")
            time.sleep(1)
    raise RuntimeError("could not compile the specification to a loss") from last


# --- monitoring proxies ------------------------------------------------------------
def make_monitor_sets(generator):
    u = torch.rand(N_MONITOR, 5, generator=generator)
    region = P3_LO + u * (P3_HI - P3_LO)
    u = torch.rand(N_MONITOR, 5, generator=generator)
    whole = MIN_IN + u * (MAX_IN - MIN_IN)
    return normalise(region), normalise(whole)


@torch.no_grad()
def violation_rate(model, region_normalised):
    advisory = model(region_normalised).argmin(dim=1)
    return float((advisory == CLEAR_OF_CONFLICT).float().mean())


@torch.no_grad()
def agreement(model, original_advisories, whole_normalised):
    return float((model(whole_normalised).argmin(dim=1) == original_advisories).float().mean())


# --- one network -----------------------------------------------------------------------
def train(net, constraint_loss_fn, args):
    tag = f"_{args.tag}" if args.tag else ""
    src = os.path.join(HERE, f"acasXu_{net}.onnx")
    models_dir = os.path.join(HERE, "models")
    traces_dir = os.path.join(HERE, "traces")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)

    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    region_norm, whole_norm = make_monitor_sets(generator)

    log(f"=== network N_{{{net.replace('_', ',')}}} from {os.path.basename(src)}")
    model = load_network(src)
    check_against_onnxruntime(model, src)
    with torch.no_grad():
        original_advisories = model(whole_norm).argmin(dim=1)

    def network(x):
        return model(x.reshape(1, 5)).reshape(5)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def snapshot_name(epoch):
        return os.path.join(models_dir, f"acasXu_{net}_pdt{tag}_e{epoch:02d}.onnx")

    per_epoch = open(os.path.join(traces_dir, f"{net}{tag}_per_epoch.csv"), "w", newline="")
    steps_file = open(os.path.join(traces_dir, f"{net}{tag}_steps.csv"), "w", newline="")
    epoch_writer = csv.writer(per_epoch)
    step_writer = csv.writer(steps_file)
    epoch_writer.writerow(
        ["epoch", "constraint_loss_mean", "constraint_loss_last",
         "violation_rate_in_region", "agreement_with_original", "seconds", "snapshot"]
    )
    step_writer.writerow(["epoch", "step", "constraint_loss", "grad_norm", "seconds"])

    # Epoch 0: the starting network, re-exported so that the export path itself can be
    # verified against Chapter 2's result before any training is trusted.
    model.eval()
    with torch.no_grad():
        start_loss = torch.as_tensor(constraint_loss_fn(acasXu=network)).sum().item()
    vr, ag = violation_rate(model, region_norm), agreement(model, original_advisories, whole_norm)
    export(model, snapshot_name(0))
    epoch_writer.writerow([0, f"{start_loss:.6f}", f"{start_loss:.6f}", f"{vr:.4f}", f"{ag:.4f}", "",
                           os.path.basename(snapshot_name(0))])
    per_epoch.flush()
    log(f"epoch   0: constraint {start_loss:+.4f} | violation rate in P3 region {100 * vr:.2f}% "
        f"| agreement with original 100.00% | exported {os.path.basename(snapshot_name(0))}")

    log(f"{args.epochs} epochs x {args.steps} steps | qllAdditive logic | alpha {ALPHA} "
        f"(property loss only: no data, no task loss) | Adam lr {LEARNING_RATE} | seed {SEED}")

    model.train()
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        losses = []
        for step in range(1, args.steps + 1):
            step_started = time.time()
            optimizer.zero_grad()
            constraint_loss = torch.as_tensor(constraint_loss_fn(acasXu=network)).sum()
            total_loss = (1 - ALPHA) * constraint_loss   # ALPHA * task_loss has no data to act on
            total_loss.backward()
            grad_norm = float(torch.sqrt(sum(
                (p.grad ** 2).sum() for p in model.parameters() if p.grad is not None
            )))
            optimizer.step()
            losses.append(constraint_loss.item())
            step_writer.writerow([epoch, step, f"{constraint_loss.item():.6f}",
                                  f"{grad_norm:.6f}", f"{time.time() - step_started:.2f}"])
            steps_file.flush()
            log(f"  epoch {epoch} step {step:2d}/{args.steps}: constraint {constraint_loss.item():+.4f} "
                f"| grad norm {grad_norm:.3f} | {time.time() - step_started:.1f}s")

        seconds = time.time() - started
        model.eval()
        vr = violation_rate(model, region_norm)
        ag = agreement(model, original_advisories, whole_norm)
        export(model, snapshot_name(epoch))
        model.train()
        epoch_writer.writerow([epoch, f"{np.mean(losses):.6f}", f"{losses[-1]:.6f}",
                               f"{vr:.4f}", f"{ag:.4f}", f"{seconds:.1f}",
                               os.path.basename(snapshot_name(epoch))])
        per_epoch.flush()
        log(f"epoch {epoch:3d}: constraint mean {np.mean(losses):+.4f} last {losses[-1]:+.4f} "
            f"| violation rate in P3 region {100 * vr:.2f}% | agreement with original {100 * ag:.2f}% "
            f"| {seconds:.0f}s | exported {os.path.basename(snapshot_name(epoch))}")

    per_epoch.close()
    steps_file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("networks", nargs="*", metavar="NET",
                        help="which of 1_7 1_8 1_9 to train (default: all)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--steps", type=int, default=STEPS_PER_EPOCH, help="gradient steps per epoch")
    parser.add_argument("--tag", default="", help="suffix for output files (e.g. p2, smoke)")
    parser.add_argument("--spec", default=DEFAULT_SPEC,
                        help=f"training specification (default: {DEFAULT_SPEC}; use acasXu-training-p2.vcl for p = 2)")
    args = parser.parse_args()
    args.networks = args.networks or NETWORKS
    unknown = [n for n in args.networks if n not in NETWORKS]
    if unknown:
        parser.error(f"unknown network(s) {unknown}; choose from {NETWORKS}")

    log(f"vehicle_lang {vcl.VERSION}; torch {torch.__version__}; {torch.get_num_threads()} threads")
    spec_path = os.path.join(HERE, args.spec)
    constraint_loss_fn = load_constraint_loss(spec_path)
    log(f"compiled {PROPERTY} of {os.path.basename(spec_path)} to a PyTorch loss under the qllAdditive logic")
    for net in args.networks:
        train(net, constraint_loss_fn, args)
    log("done")


if __name__ == "__main__":
    sys.exit(main())
