"""Regenerate the results table in README.md from traces-neg/verify.csv.

The verifications outlive any one working session, so this exists to fold whatever has
completed into the README without needing to remember the arithmetic. Safe to run
repeatedly: it replaces the table between the two markers each time.

    python3 record_results.py
"""
import csv, os, pathlib, re

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
CSV = HERE / "traces-neg" / "verify.csv"
README = HERE / "README.md"
START = "<!-- RESULTS TABLE START -->"
END = "<!-- RESULTS TABLE END -->"

EXPECTED = [("capucci_neg_e01", "training"), ("capucci_neg_e02", "training"),
            ("capucci_neg_e01", "exercise7"), ("capucci_neg_e02", "exercise7")]
SPEC_LABEL = {"training": "training", "exercise7": "Exercise #7"}

rows = {}
if CSV.exists():
    for r in csv.DictReader(open(CSV)):
        rows[(r["model"], r["spec"])] = r

out = ["| Model | spec | correct | verified | falsified | of which misclassified | genuinely non-robust | robust share of eligible | solver |",
       "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for key in EXPECTED:
    model, spec = key
    r = rows.get(key)
    if r is None or not str(r["verified"]).isdigit():
        out.append(f"| `{model}` | {SPEC_LABEL[spec]} | 29/50 | _not yet run_ | | | | | |")
        continue
    ceiling = int(r["ceiling"]); verified = int(r["verified"]); falsified = int(r["falsified"])
    mis = 50 - ceiling
    genuine = falsified - mis
    share = f"{100 * verified / ceiling:.1f}%" if ceiling else "--"
    out.append(f"| `{model}` | {SPEC_LABEL[spec]} | {ceiling}/50 | **{verified}/50** | "
               f"{falsified}/50 | {mis} | {genuine} | {share} | {int(float(r['seconds']))} s |")

text = README.read_text()
if START in text and END in text:
    text = re.sub(re.escape(START) + r".*?" + re.escape(END),
                  START + "\n\n" + "\n".join(out) + "\n\n" + END, text, flags=re.S)
    README.write_text(text)
    print(f"README.md updated: {sum(1 for k in EXPECTED if k in rows and str(rows[k]['verified']).isdigit())}"
          f"/{len(EXPECTED)} verifications recorded")
else:
    print("markers not found in README.md; table below, paste it in manually:\n")
    print("\n".join(out))
