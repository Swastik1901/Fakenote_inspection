"""Pre-check: is the image a banknote at all?

Rejects only on positive evidence of a non-note (wrong colours, blank, wrong shape). Lack of colour is NOT
a reason, so a black-and-white photocopy still reaches the checks and is flagged as a fake instead.
"""
import cv2
import numpy as np

from . import config as C


def check_banknote(img_bgr, raw_aspect=None):
    n = cv2.resize(img_bgr, (C.NOTE_W, C.NOTE_H), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(n, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lo, hi = C.GATE_HUE
    strong = (S > 60) & (V > 50)
    off_pct = float((strong & ((H < lo) | (H > hi))).mean() * 100)
    sat_pct = float(((S > 150) & (V > 50)).mean() * 100)
    gray = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rows, cols = C.NOTE_H // 16, C.NOTE_W // 16
    blocks = gray[:rows * 16, :cols * 16].reshape(rows, 16, cols, 16).transpose(0, 2, 1, 3).reshape(-1, 256)
    flat_pct = float((blocks.std(axis=1) < 2.5).mean() * 100)

    reasons = []
    if off_pct > C.GATE_OFF_PALETTE_MAX:
        reasons.append(f"{off_pct:.0f}% of the image is strongly coloured in hues this note does not use")
    if sat_pct > C.GATE_SATURATED_MAX:
        reasons.append(f"{sat_pct:.0f}% of pixels are intensely saturated, but banknote inks are muted")
    if flat_pct > C.GATE_FLAT_MAX:
        reasons.append(f"{flat_pct:.0f}% of the image is flat or blank, but a note is printed with fine detail everywhere")
    if raw_aspect is not None and not (C.GATE_ASPECT[0] <= raw_aspect <= C.GATE_ASPECT[1]):
        reasons.append(f"the shape is {raw_aspect:.2f}:1, but a banknote is a long rectangle (about 2:1)")
    return dict(ok=not reasons, reasons=reasons,
                metrics=dict(off_palette=off_pct, saturated=sat_pct, flat=flat_pct, aspect=raw_aspect))