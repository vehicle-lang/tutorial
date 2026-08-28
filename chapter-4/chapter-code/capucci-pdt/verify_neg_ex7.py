"""Verify the negated-sign snapshots against EXERCISE #7's specification.

The companion script verify_neg.py checks the same two models against the specification
they were trained on. This one uses fashionRobustness-solution.vcl, Exercise #7's strict
formulation, so the counts are directly comparable with the vanilla baseline of 22/50.

Running both gives the pair of numbers: what the trained property says, and what the
published property says. The two differ only on ties -- Exercise #7 requires the advised
label to score strictly higher, the training spec allows equality -- so the strict count
can only be lower or equal.

Guards: a per-image Marabou timeout, a wall-clock limit per model, and an address-space
cap, so a hard query cannot take the machine down.
"""
import csv, os, re, resource, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUTS = os.path.join(HERE, "marabou-outputs-neg-ex7")
SPEC = os.path.join(HERE, "fashionRobustness-solution.vcl")
IMAGES = os.path.join(HERE, "0-49Images.idx")
LABELS = os.path.join(HERE, "0-49Labels.idx")
EPSILON = "0.02"
QUERY_TIMEOUT, WALL_TIMEOUT, MEM_CAP_GB = 120, 7200, 24

MODELS = [("capucci_neg_e01", 29), ("capucci_neg_e02", 29)]   # name, ceiling from training

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# one Marabou at a time: a 50-image run needs about 14 GB, two would risk the machine
while True:
    done = 0
    path = os.path.join(HERE, "traces-neg", "verify.csv")
    if os.path.exists(path):
        done = sum(1 for row in csv.reader(open(path)) if len(row) > 1 and row[1] == "training")
    if done >= 2:
        break
    log("waiting for the training-spec verifications to finish")
    time.sleep(60)

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
        cw.writerow([name, "exercise7", EPSILON, ceiling, "wall-timeout", "-", "-", "-",
                     f"{secs:.0f}"]); f.flush()
        continue
    secs = time.time() - t0
    txt = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", r.stdout + r.stderr)
    txt = "\n".join(l.split("\r")[-1].rstrip() for l in txt.replace("\r\n", "\n").split("\n"))
    def c(k):
        m = re.search(rf"{k}:\s+(\d+)/(\d+)", txt); return int(m.group(1)) if m else -1
    with open(os.path.join(OUTS, name + ".txt"), "w") as vf:
        vf.write(f"{name}.onnx verified against fashionRobustness-solution.vcl (EXERCISE #7's\n"
                 f"specification), epsilon {EPSILON}, 50 FashionMNIST test images.\n"
                 f"Elapsed {secs:.0f}s. Progress bars stripped; otherwise verbatim.\n"
                 + "=" * 78 + "\n\n" + txt)
    cw.writerow([name, "exercise7", EPSILON, ceiling, c("verified"), c("falsified"),
                 c("timed-out"), c("errored"), f"{secs:.0f}"]); f.flush()
    log(f"{name}: verified {c('verified')}/50, falsified {c('falsified')}/50, "
        f"timed-out {c('timed-out')}, errored {c('errored')}  [{secs:.0f}s]")
f.close()
log("done")
