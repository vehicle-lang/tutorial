"""Verify every run-3 snapshot against Exercise #7's specification, as each one appears.

Runs alongside pdt-Capucci-v028.py. For epochs 1..10 it waits for
capucci-models-v028/capucci_v028_eNN.onnx to exist (the trainer writes a .part file and
renames it, so existence means complete), then runs Chapter 3 Exercise #7's command on it
at epsilon 0.02, unchanged except for the network, so every count is directly comparable
with the vanilla baseline of 22/50.

Only the strict Exercise #7 specification is used. Run 2 showed the strict and non-strict
forms agree on every image of this problem, so the second pass adds nothing.

Guards, as in the run-2 scripts: a per-image Marabou timeout, a wall-clock limit per
model, and an address-space cap. One Marabou at a time; a 50-image run needs about 14 GB.

Results append to traces-v028/verify.csv; transcripts go to marabou-outputs-v028/.
"""
import csv, os, re, resource, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "capucci-models-v028")
TRACES = os.path.join(HERE, "traces-v028")
OUTS = os.path.join(HERE, "marabou-outputs-v028")
SPEC = os.path.join(HERE, "fashionRobustness-solution.vcl")
IMAGES = os.path.join(HERE, "0-49Images.idx")
LABELS = os.path.join(HERE, "0-49Labels.idx")
EPSILON = "0.02"
NUM_EPOCHS = 10
QUERY_TIMEOUT, WALL_TIMEOUT, MEM_CAP_GB = 120, 7200, 24
SNAPSHOT_WAIT = 4 * 3600   # give up on a snapshot if the trainer has produced nothing for 4 h

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def ceiling_for(epoch):
    """correct_on_50_test for this epoch, from the trainer's per_epoch.csv."""
    path = os.path.join(TRACES, "per_epoch.csv")
    if not os.path.exists(path):
        return ""
    for r in csv.DictReader(open(path)):
        if r["epoch"] == str(epoch):
            return r["correct_on_50_test"]
    return ""

os.makedirs(OUTS, exist_ok=True)
os.makedirs(TRACES, exist_ok=True)
csv_path = os.path.join(TRACES, "verify.csv")
new = not os.path.exists(csv_path)
f = open(csv_path, "a", newline=""); cw = csv.writer(f)
if new:
    cw.writerow(["model", "spec", "epsilon", "ceiling", "verified", "falsified",
                 "timed_out", "errored", "seconds"]); f.flush()

for epoch in range(1, NUM_EPOCHS + 1):
    name = f"capucci_v028_e{epoch:02d}"
    model = os.path.join(MODELS, name + ".onnx")
    waited = 0
    while not os.path.exists(model):
        if waited == 0:
            log(f"waiting for {name}.onnx")
        time.sleep(60); waited += 60
        if waited >= SNAPSHOT_WAIT:
            log(f"no {name}.onnx after {SNAPSHOT_WAIT // 3600} h; assuming the trainer stopped")
            f.close(); raise SystemExit(1)
    time.sleep(5)   # let per_epoch.csv's flush land
    ceiling = ceiling_for(epoch)
    log(f"{name}: 50 images at epsilon {EPSILON} (ceiling {ceiling or '?'}/50)")
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
                 f"specification), epsilon {EPSILON}, 50 FashionMNIST test images, Vehicle 0.28.0.\n"
                 f"Elapsed {secs:.0f}s. Progress bars stripped; otherwise verbatim.\n"
                 + "=" * 78 + "\n\n" + txt)
    cw.writerow([name, "exercise7", EPSILON, ceiling, c("verified"), c("falsified"),
                 c("timed-out"), c("errored"), f"{secs:.0f}"]); f.flush()
    log(f"{name}: verified {c('verified')}/50, falsified {c('falsified')}/50, "
        f"timed-out {c('timed-out')}, errored {c('errored')}  [{secs:.0f}s]")
f.close()
log("done")
