"""Evaluate on dataset/genuine and dataset/fake, print metrics + per-check means.
Use the per-check table to tune ranges in detector/config.py.
Usage: python evaluate.py [--out results.csv]
"""
import csv, sys
from pathlib import Path
import cv2
import numpy as np
from detector.pipeline import analyze, get_references

ROOT = Path(__file__).parent / "dataset"
refs = get_references()
rows, per_check = [], {}
for label, folder in ((1, "genuine"), (0, "fake")):
    for p in sorted((ROOT / folder).glob("*")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        r = analyze(img, refs)
        pred = 1 if r["verdict"] == "GENUINE" else 0   # suspicious counts as not-genuine
        rows.append((p.name, label, pred, r["score"], r["verdict"]))
        for c in r["results"]:
            if c.score is not None:
                per_check.setdefault(c.name, {1: [], 0: []})[label].append(c.score)

if not rows:
    sys.exit("Put images in dataset/genuine and dataset/fake first.")
tp = sum(1 for _, l, p, *_ in rows if l == 0 and p == 0)   # fake caught
fn = sum(1 for _, l, p, *_ in rows if l == 0 and p == 1)   # fake missed
tn = sum(1 for _, l, p, *_ in rows if l == 1 and p == 1)
fp = sum(1 for _, l, p, *_ in rows if l == 1 and p == 0)   # genuine flagged
prec = tp / (tp + fp) if tp + fp else 0
rec = tp / (tp + fn) if tp + fn else 0
acc = (tp + tn) / len(rows)
print(f"\nImages: {len(rows)}   Accuracy: {acc:.2%}   Precision(fake): {prec:.2%}   Recall(fake): {rec:.2%}")
print("\nConfusion matrix (rows=actual, cols=predicted)")
print(f"              pred FAKE   pred GENUINE\n actual FAKE   {tp:>8}   {fn:>10}\n actual GEN    {fp:>8}   {tn:>10}")
print("\nMean score per check (genuine vs fake) - big gap = useful check")
for name, d in per_check.items():
    g = np.mean(d[1]) if d[1] else float('nan'); f = np.mean(d[0]) if d[0] else float('nan')
    print(f" {name:<18} genuine {g:.2f}   fake {f:.2f}   gap {g-f:+.2f}")
out = Path(sys.argv[sys.argv.index('--out') + 1]) if '--out' in sys.argv else None
if out:
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "actual_genuine", "pred_genuine", "score", "verdict"]); w.writerows(rows)
