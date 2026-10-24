"""Print watermark metrics for backlit photos so you can calibrate WM_BAND_RANGE in detector/config.py.
Usage: python watermark_probe.py dataset/backlit_genuine dataset/backlit_plain_paper
Pick a range whose low end sits just above plain paper and whose high end sits near genuine notes."""
import sys
from pathlib import Path
import cv2
from detector.preprocess import rectify
from detector.checks import _wm_metrics

for folder in sys.argv[1:]:
    vals = []
    for p in sorted(Path(folder).glob("*")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        _, rel = _wm_metrics(rectify(img)["note"])
        vals.append(rel)
        print(f"{folder}/{p.name}: band contrast {rel:.2f}%")
    if vals:
        print(f"== {folder}: mean {sum(vals)/len(vals):.2f}%  (n={len(vals)})\n")
