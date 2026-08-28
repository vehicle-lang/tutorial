"""Verify the negated-sign snapshots against the SAME specification used for training.

Uses fashionRobustness-capucci.vcl, not Exercise #7's fashionRobustness-solution.vcl, so
the property optimised and the property checked are the same text. Note this makes the
counts NOT directly comparable with the vanilla baseline of 22/50, which was measured
under Exercise #7's strict formulation.

Guards: a per-image Marabou timeout, a wall-clock limit per model, and an address-space
cap, so a hard query cannot take the machine down.
"""
import csv, os, re, resource, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUTS = os.path.join(HERE, "marabou-outputs-neg")
SPEC = os.path.join(HERE, "fashionRobustness-capucci.vcl")
IMAGES = os.path.join(HERE, "0-49Images.idx")
LABELS = os.path.join(HERE, "0-49Labels.idx")
EPSILON = "0.02"
QUERY_TIMEOUT, WALL_TIMEOUT, MEM_CAP_GB = 120, 7200, 24

MODELS = [("capucci_neg_e01", 29), ("capucci_neg_e02", 29)]   # name, ceiling from training

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

os.makedirs(OUTS, exist_ok=True)
csv_path = os.path.join(HERE, "traces-neg", "verify.csv")
new = not os.path.exists(csv_path)
f = open(csv_path, "a", newline=""); cw = csv.writer(f)
if new:
    cw.writerow(["model", "spec", "epsilon", "ceiling", "verified", "falsified",
                 "timed_out", "errored", "seconds"]); f.flush()

for name, ceiling in MODELS:
    model = os.path.join(HERE, "capucci-models-neg", name + ".onnx")
    log(f"{name}: 50 images at epsilon {EPSILON} (ceiling {ceiling}/50)")
    t0 = time.time()
    try:
        r = subprocess.run(
            ["vehicle", "verify", "--specification", SPEC,
             "--network", f"classifier:{model}",
             "--parameter", f"epsilon:{EPSILON}",
             "--dataset", f"trainingImages:{IMAGES}",
             "--dataset", f"trainingLabels:{LABELS}",
             "--solver", "Marabou",
             "--solver-args", f"--timeout={QUERY_TIMEOUT}"],
            capture_output=True, text=True, timeout=WALL_TIMEOUT,
            preexec_fn=lambda: resource.setrlimit(
                resource.RLIMIT_AS, (MEM_CAP_GB * 1024**3, MEM_CAP_GB * 1024**3)))
    except subprocess.TimeoutExpired:
        secs = time.time() - t0
        log(f"{name}: hit the {WALL_TIMEOUT}s wall-clock limit")
        cw.writerow([name, "training", EPSILON, ceiling, "wall-timeout", "-", "-", "-",
                     f"{secs:.0f}"]); f.flush()
        continue
    secs = time.time() - t0
    txt = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", r.stdout + r.stderr)
    txt = "\n".join(l.split("\r")[-1].rstrip() for l in txt.replace("\r\n", "\n").split("\n"))
    def c(k):
        m = re.search(rf"{k}:\s+(\d+)/(\d+)", txt); return int(m.group(1)) if m else -1
    with open(os.path.join(OUTS, name + ".txt"), "w") as vf:
        vf.write(f"{name}.onnx verified against fashionRobustness-capucci.vcl (the TRAINING\n"
                 f"specification), epsilon {EPSILON}, 50 FashionMNIST test images.\n"
                 f"Elapsed {secs:.0f}s. Progress bars stripped; otherwise verbatim.\n"
                 + "=" * 78 + "\n\n" + txt)
    cw.writerow([name, "training", EPSILON, ceiling, c("verified"), c("falsified"),
                 c("timed-out"), c("errored"), f"{secs:.0f}"]); f.flush()
    log(f"{name}: verified {c('verified')}/50, falsified {c('falsified')}/50, "
        f"timed-out {c('timed-out')}, errored {c('errored')}  [{secs:.0f}s]")
f.close()
log("done")
