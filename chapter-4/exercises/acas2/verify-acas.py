"""Verify Property 3 on every snapshot pdt-acas.py produced, with Chapter 2's command.

For each network and epoch this runs, unchanged apart from the network file,

    vehicle verify --specification acasXu.vcl --solver Marabou \
        --network acasXu:models/acasXu_NET_pdt_eNN.onnx --property property3

i.e. the verification uses the ORIGINAL Chapter 2 specification (strict `advises` with
the `i != j` guard), not the training specification, so the results are directly
comparable with the Chapter 2 table in the README. Epoch 00 is the starting network
re-exported through PyTorch, and is expected to reproduce Chapter 2's counterexample.

Verifications run one at a time. Full Vehicle/Marabou transcripts go to
marabou-outputs/, and one line per verification is appended to traces/verify.csv.

Usage (from this folder):

    python3 verify-acas.py                 # every snapshot in models/
    python3 verify-acas.py 1_7             # one network's snapshots
    python3 verify-acas.py 1_7 --epochs 0 3
    python3 verify-acas.py --tag p2        # only the p = 2 run's snapshots
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "acasXu.vcl")
MODELS = os.path.join(HERE, "models")
OUTPUTS = os.path.join(HERE, "marabou-outputs")
VERIFY_CSV = os.path.join(HERE, "traces", "verify.csv")
NETWORKS = ["1_7", "1_8", "1_9"]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def verify(model_path):
    command = [
        "vehicle", "verify",
        "--specification", SPEC,
        "--solver", "Marabou",
        "--network", f"acasXu:{model_path}",
        "--property", "property3",
    ]
    started = time.time()
    proc = subprocess.run(command, capture_output=True, text=True, cwd=HERE)
    seconds = time.time() - started
    text = ANSI.sub("", proc.stdout + proc.stderr)
    if "✓" in text or "proved no counterexample" in text:
        verdict = "verified"
    elif "✗" in text or "found a counterexample" in text:
        verdict = "falsified"
    else:
        verdict = "error"
    match = re.search(r"x:\s*\[(.*?)\]", text)
    counterexample = match.group(1).strip() if match else ""
    return verdict, counterexample, seconds, text, " ".join(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("networks", nargs="*", metavar="NET", help="1_7 1_8 1_9 (default: all)")
    parser.add_argument("--epochs", nargs="*", type=int, help="epochs to verify (default: all found)")
    parser.add_argument("--tag", default=None, help="only snapshots with this training tag ('' for untagged; default: all)")
    args = parser.parse_args()
    networks = args.networks or NETWORKS

    os.makedirs(OUTPUTS, exist_ok=True)
    os.makedirs(os.path.dirname(VERIFY_CSV), exist_ok=True)
    new_file = not os.path.exists(VERIFY_CSV)
    with open(VERIFY_CSV, "a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["network", "tag", "epoch", "model", "verdict", "counterexample", "solver_seconds", "transcript"])
        for net in networks:
            for model_path in sorted(glob.glob(os.path.join(MODELS, f"acasXu_{net}_pdt*_e*.onnx"))):
                match = re.search(rf"acasXu_{net}_pdt_?(.*?)_e(\d+)\.onnx$", os.path.basename(model_path))
                tag, epoch = match.group(1), int(match.group(2))
                if args.tag is not None and tag != args.tag:
                    continue
                if args.epochs is not None and epoch not in args.epochs:
                    continue
                name = os.path.basename(model_path)
                log(f"verifying {name} ...")
                verdict, counterexample, seconds, text, command = verify(model_path)
                transcript = os.path.join(OUTPUTS, name.replace(".onnx", ".txt"))
                with open(transcript, "w") as out:
                    out.write(f"$ {command}\n\n{text}")
                writer.writerow([net, tag, epoch, name, verdict, counterexample, f"{seconds:.1f}",
                                 os.path.relpath(transcript, HERE)])
                handle.flush()
                log(f"  {verdict}" + (f"  x = [{counterexample}]" if counterexample else "") + f"  ({seconds:.1f}s)")
    log(f"done; results in {os.path.relpath(VERIFY_CSV, HERE)}")


if __name__ == "__main__":
    sys.exit(main())
