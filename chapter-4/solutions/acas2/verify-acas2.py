"""Verify all ten properties of the upstream ACAS Xu specification on a set of ONNX files.

One `vehicle verify` call per (network file, property), so that each property gets its own
verdict, timing and transcript, and a slow property cannot hide the others:

    vehicle verify --specification acasXu-upstream.vcl --solver Marabou \
        --network acasXu:<file> --property propertyK

Verdicts: verified / falsified / timeout / error. Each call is limited to --timeout seconds
(default 900). Transcripts go to marabou-outputs/<file>_propertyK.txt, and one row per call
is appended to traces/verify.csv (or --csv). Several copies may run at once; each call runs in
its own process group so a timeout kills only its own solver.

Usage (from this folder):

    python3 verify-acas2.py acasXu_1_7.onnx                        # the starting network
    python3 verify-acas2.py models/acasXu_1_7_seq_e*.onnx           # every snapshot
    python3 verify-acas2.py models/acasXu_1_7_seq_e10.onnx --properties 3 4
    python3 verify-acas2.py acasXu_1_8.onnx --properties 6 --spec specs-verify/property06.vcl --timeout 9000
"""

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "acasXu-upstream.vcl")
OUTPUTS = os.path.join(HERE, "marabou-outputs")
VERIFY_CSV = os.path.join(HERE, "traces", "verify.csv")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def verify(model_path, k, timeout, spec=SPEC, solver_args=None):
    # GNU timeout is a second, independent layer under the watchdog: it kills its whole
    # process group (vehicle and its Marabou children) when the deadline passes.
    command = ["timeout", "-s", "KILL", str(timeout + 5),
               "vehicle", "verify", "--specification", spec, "--solver", "Marabou",
               "--network", f"acasXu:{model_path}", "--property", f"property{k}"]
    if solver_args:
        command += ["--solver-args", solver_args]
    started = time.time()
    # Run in its own process group, and kill that group from a watchdog thread when the
    # deadline passes. (Popen.communicate(timeout=...) was seen to overrun by many minutes
    # on long Marabou runs; the timer does not depend on how the output is read.) The group
    # kill hits this call's Marabou only, not another verify-acas2.py's.
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=HERE, start_new_session=True)
    timed_out = threading.Event()

    def kill():
        timed_out.set()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    watchdog = threading.Timer(timeout, kill)
    watchdog.start()
    text, _ = proc.communicate()
    watchdog.cancel()
    text = ANSI.sub("", text or "")
    if timed_out.is_set():
        verdict = "timeout"
        text += f"\n\n[killed after {timeout}s]"
    elif "🗸" in text or "✓" in text or "proved no counterexample" in text:
        verdict = "verified"
    elif "✗" in text or "found a counterexample" in text:
        verdict = "falsified"
    else:
        verdict = "error"
    seconds = time.time() - started
    match = re.search(r"x:\s*\[(.*?)\]", text)
    return verdict, (match.group(1).strip() if match else ""), seconds, text, " ".join(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="+", help="ONNX files to verify")
    parser.add_argument("--properties", nargs="*", type=int, default=list(range(1, 11)))
    parser.add_argument("--timeout", type=int, default=900, help="seconds per (file, property)")
    parser.add_argument("--csv", default=VERIFY_CSV, help="results file to append to (default traces/verify.csv)")
    parser.add_argument("--solver-args", default=None,
                        help="passed through to Marabou, e.g. '--snc --num-workers=8' for split-and-conquer")
    parser.add_argument("--spec", default=SPEC,
                        help="specification to verify against (default acasXu-upstream.vcl; e.g. specs-verify/property06.vcl)")
    args = parser.parse_args()
    verify_csv = os.path.join(HERE, args.csv)

    os.makedirs(OUTPUTS, exist_ok=True)
    os.makedirs(os.path.dirname(verify_csv), exist_ok=True)
    new_file = not os.path.exists(verify_csv)
    with open(verify_csv, "a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["model", "property", "verdict", "counterexample", "solver_seconds", "transcript"])
        for model_path in args.models:
            name = os.path.basename(model_path)
            log(f"=== {name}")
            for k in args.properties:
                verdict, counterexample, seconds, text, command = verify(os.path.abspath(model_path), k, args.timeout,
                                                                         os.path.join(HERE, args.spec), args.solver_args)
                suffix = "" if os.path.abspath(os.path.join(HERE, args.spec)) == SPEC else "_" + os.path.splitext(os.path.basename(args.spec))[0]
                if args.solver_args:
                    suffix += "_snc" if "--snc" in args.solver_args else "_solverargs"
                transcript = os.path.join(OUTPUTS, name.replace(".onnx", f"_property{k}{suffix}.txt"))
                with open(transcript, "w") as out:
                    out.write(f"$ {command}\n\n{text}")
                writer.writerow([name, k, verdict, counterexample, f"{seconds:.1f}", os.path.relpath(transcript, HERE)])
                handle.flush()
                log(f"  property{k:<2d}: {verdict:9s}" + (f"  x = [{counterexample}]" if counterexample else "") + f"  ({seconds:.0f}s)")
    log(f"done; results in {os.path.relpath(verify_csv, HERE)}")


if __name__ == "__main__":
    sys.exit(main())
