"""Rebuild the combined Marabou results table in README.md from the traces/verify_*.csv files.

Safe to run repeatedly: it replaces whatever lies between the two markers

    <!-- COMBINED VERIFY TABLE START -->  ...  <!-- COMBINED VERIFY TABLE END -->

Cells read: ✓ (verified, with solver time), ✗ (falsified, with the counterexample), timeout,
error, or "running" when that (snapshot, property) has no row yet.

    python3 record-results.py               # refresh the tables in README.md
    python3 record-results.py --baselines   # print the three-network baseline table (for Exercise #6)
"""

import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(HERE, "README.md")

# column title -> csv file (all relative to traces/)
COLUMNS = [
    ("N_{1,8} baseline", "verify_baseline.csv"),
    ("tour e10", "verify_e10.csv"),
    ("fresh Adam e10", "verify_cyc3reset_e10.csv"),
    ("fresh Adam e20", "verify_cyc3reset_e20.csv"),
    ("fresh Adam e30", "verify_cyc3reset_e30.csv"),
]
# long re-runs with one-property specs and a 9000 s limit; they override the cell of the
# same (network, property) in the compact table (marked with a dagger)
LONG_RUNS = [
    ("N_{1,8} baseline", "verify_long_baseline_p6.csv"),
    ("N_{1,8} baseline", "verify_long_baseline_p7.csv"),
    ("N_{1,8} baseline", "verify_long_baseline_p8.csv"),
    ("fresh Adam e30", "verify_long_e30_p6.csv"),
]
START, END = "<!-- COMBINED VERIFY TABLE START -->", "<!-- COMBINED VERIFY TABLE END -->"
COMPACT_START, COMPACT_END = "<!-- COMPACT VERIFY TABLE START -->", "<!-- COMPACT VERIFY TABLE END -->"


def load(name):
    path = os.path.join(HERE, "traces", name)
    if not os.path.exists(path):
        return {}
    return {int(r["property"]): r for r in csv.DictReader(open(path))}


def cell(r):
    if r is None:
        return "*running*"
    v, t = r["verdict"], f"{float(r['solver_seconds']):.0f} s"
    if v == "verified":
        return f"✓ ({t})"
    if v == "falsified":
        x = ", ".join(f"{float(a):.4g}" for a in r["counterexample"].split(",")) if r["counterexample"] else ""
        return f"✗ ({t})<br><small>[{x}]</small>" if x else f"✗ ({t})"
    return f"{v} ({t})"


def compact_cell(r, long=False):
    if r is None:
        return "running"
    v, t = r["verdict"], f"{float(r['solver_seconds']):.0f} s"
    mark = "†" if long else ""
    if v == "verified":
        return f"✓{mark} ({t})"
    if v == "falsified":
        return f"✗{mark} ({t})"
    return f"{v}{mark} ({t})"


def build_compact():
    data = {title: load(name) for title, name in COLUMNS}
    long = {}
    for title, name in LONG_RUNS:
        for k, r in load(name).items():
            long[(title, k)] = r
    lines = ["| Network | " + " | ".join(str(k) for k in range(1, 11)) + " | verified |", "|---|" + "---|" * 11]
    for title, _ in COLUMNS:
        d = data[title]
        cells, verified = [], []
        for k in range(1, 11):
            if (title, k) in long:
                r = long[(title, k)]
                cells.append(compact_cell(r, long=True))
            else:
                r = d.get(k)
                cells.append(compact_cell(r))
            if r is not None and r["verdict"] == "verified":
                verified.append(str(k))
        lines.append(f"| {title} | " + " | ".join(cells) + f" | {{{', '.join(verified)}}} |")
    lines.append("")
    lines.append("† from the re-run with a one-property specification and a 9000 s limit (see below); "
                 "the original 900 s verdict is in the detailed table.")
    return "\n".join(lines)


BASELINES = [("N_{1,7}", "verify_baseline_1_7.csv"), ("N_{1,8}", "verify_baseline.csv"), ("N_{1,9}", "verify_baseline_1_9.csv")]


def build_baselines():
    """The three shipped networks on the ten properties, verdict symbols only: the table
    Exercise #6 shows as what Chapter 2's exercises were expected to produce."""
    lines = ["| Network | " + " | ".join(str(k) for k in range(1, 11)) + " | verified |", "|---|" + "---|" * 11]
    for title, name in BASELINES:
        d = load(name)
        cells, verified = [], []
        for k in range(1, 11):
            r = d.get(k)
            if r is None:
                cells.append("running")
            elif r["verdict"] == "verified":
                cells.append("✓"); verified.append(str(k))
            elif r["verdict"] == "falsified":
                cells.append("✗")
            else:
                cells.append("timeout")
        lines.append(f"| {title} | " + " | ".join(cells) + f" | {{{', '.join(verified)}}} |")
    return "\n".join(lines)


def build():
    data = [(title, load(name)) for title, name in COLUMNS]
    lines = ["| Property | " + " | ".join(t for t, _ in data) + " |", "|---:|" + "---|" * len(data)]
    for k in range(1, 11):
        lines.append(f"| {k} | " + " | ".join(cell(d.get(k)) for _, d in data) + " |")
    verified = []
    for title, d in data:
        vs = [str(k) for k in range(1, 11) if d.get(k, {}).get("verdict") == "verified"]
        pending = any(d.get(k) is None for k in range(1, 11))
        verified.append(f"| {title} | {{{', '.join(vs)}}}{' (so far)' if pending else ''} |")
    summary = ["| Network | properties verified |", "|---|---|"] + verified
    return "\n".join(lines) + "\n\nProperties verified per network:\n\n" + "\n".join(summary)


def main():
    import sys
    if "--baselines" in sys.argv:
        print(build_baselines())
        return
    text = open(README).read()
    compact = build_compact()
    cblock = f"{COMPACT_START}\n{compact}\n{COMPACT_END}"
    if COMPACT_START in text:
        text = re.sub(re.escape(COMPACT_START) + r".*?" + re.escape(COMPACT_END), lambda _: cblock, text, flags=re.S)
    else:
        text = text.replace(START, cblock + "\n\nThe same verdicts in full, with counterexamples:\n\n" + START, 1)
    table = build()
    block = f"{START}\n{table}\n{END}"
    text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    open(README, "w").write(text)
    print(compact)


if __name__ == "__main__":
    main()
