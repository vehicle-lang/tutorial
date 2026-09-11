"""Regenerate the run-3 results table in README.md from traces-v028/.

Joins the trainer's per_epoch.csv with the watcher's verify.csv, one row per epoch, and
replaces the table between the V028 markers. Safe to run repeatedly.

    python3 record_results_v028.py
"""
import csv, os, pathlib, re

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
EPOCHS = HERE / "traces-v028" / "per_epoch.csv"
VERIFY = HERE / "traces-v028" / "verify.csv"
README = HERE / "README.md"
START = "<!-- V028 RESULTS TABLE START -->"
END = "<!-- V028 RESULTS TABLE END -->"
NUM_EPOCHS = 10

epochs = {}
if EPOCHS.exists():
    for r in csv.DictReader(open(EPOCHS)):
        epochs[int(r["epoch"])] = r
verified = {}
if VERIFY.exists():
    for r in csv.DictReader(open(VERIFY)):
        m = re.search(r"_e(\d+)$", r["model"])
        if m:
            verified[int(m.group(1))] = r

out = ["| epoch | constraint | cross-entropy | train acc | correct on 50 test | verified | genuinely non-robust | robust share of eligible | solver |",
       "| ----: | ---------: | ------------: | --------: | -----------------: | -------: | -------------------: | -----------------------: | -----: |",
       "| 0 (start) | -- | 0.0413 | 99.5% | 38/50 | **22/50** | 16 | 57.9% | 924 s |"]
for e in range(1, NUM_EPOCHS + 1):
    t = epochs.get(e); v = verified.get(e)
    if t is None:
        out.append(f"| {e} | _not yet trained_ | | | | | | | |")
        continue
    row = (f"| {e} | {float(t['constraint_loss']):+.4f} | {float(t['cross_entropy']):.4f} | "
           f"{100 * float(t['train_accuracy']):.1f}% | {t['correct_on_50_test']}/50 | ")
    if v is None or not str(v["verified"]).isdigit():
        row += "_not yet verified_ | | | |"
    else:
        ceiling = int(t["correct_on_50_test"]); ver = int(v["verified"]); fal = int(v["falsified"])
        genuine = fal - (50 - ceiling)
        share = f"{100 * ver / ceiling:.1f}%" if ceiling else "--"
        row += f"**{ver}/50** | {genuine} | {share} | {int(float(v['seconds']))} s |"
    out.append(row)

text = README.read_text()
if START in text and END in text:
    text = re.sub(re.escape(START) + r".*?" + re.escape(END),
                  START + "\n\n" + "\n".join(out) + "\n\n" + END, text, flags=re.S)
    README.write_text(text)
    print(f"README.md updated: {len(epochs) - (1 if 0 in epochs else 0)}/{NUM_EPOCHS} epochs trained, "
          f"{sum(1 for v in verified.values() if str(v['verified']).isdigit())}/{NUM_EPOCHS} verified")
else:
    print("markers not found in README.md; table below, paste it in manually:\n")
    print("\n".join(out))
