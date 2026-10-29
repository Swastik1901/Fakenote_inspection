"""Pre-check: is the image a banknote at all?

Rejects only on positive evidence of a non-note (scattered colours, blank, wrong shape). Lack of colour is NOT
a reason, so a black-and-white photocopy still reaches the checks and is flagged as a fake instead.

Colour rule: Indian notes each have ONE dominant tint (brown, yellow-green, blue, lavender, yellow, stone-grey,
magenta ...), so the old "must stay inside the Rs 500 hue band" test wrongly threw out every other denomination
(a genuine lavender Rs 100 was rejected). A note now passes when EITHER it sits inside the Rs 500 palette (as before)
OR most of its strongly coloured pixels sit in one hue family. Only multi-coloured images (photos, screenshots,
posters) fail both.
"""
import cv2
import numpy as np

from . import config as C


def hue_dominance(h_strong, width=None):
    """Share (0..100) of the strongly coloured pixels that fall inside the best circular hue window, and that window's centre."""
    width = width or C.GATE_HUE_FAMILY
    if h_strong.size == 0:
        return 100.0, None
    hist = np.bincount(h_strong.ravel().astype(np.int64) % 180, minlength=180).astype(np.float64)
    ext = np.concatenate([hist, hist])                              # circular: red sits at both ends of the hue axis
    cs = np.concatenate([[0.0], np.cumsum(ext)])
    win = cs[width:width + 180] - cs[:180]                          # pixels in [s, s+width) for every start s
    s = int(np.argmax(win))
    return float(win[s] / hist.sum() * 100.0), (s + width / 2.0) % 180


def check_banknote(img_bgr, raw_aspect=None):
    n = cv2.resize(img_bgr, (C.NOTE_W, C.NOTE_H), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(n, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lo, hi = C.GATE_HUE
    strong = (S > 60) & (V > 50)
    off_pct = float((strong & ((H < lo) | (H > hi))).mean() * 100)                       # outside the Rs 500 palette (legacy test)
    strong_pct = float(strong.mean() * 100)
    dom_pct, dom_hue = hue_dominance(H[strong])
    scattered = (100.0 - dom_pct) * strong_pct / 100.0                                   # % of the whole image coloured outside the main tint
    sat_pct = float(((S > 150) & (V > 50)).mean() * 100)
    gray = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rows, cols = C.NOTE_H // 16, C.NOTE_W // 16
    blocks = gray[:rows * 16, :cols * 16].reshape(rows, 16, cols, 16).transpose(0, 2, 1, 3).reshape(-1, 256)
    flat_pct = float((blocks.std(axis=1) < 2.5).mean() * 100)

    reasons = []
    if off_pct > C.GATE_OFF_PALETTE_MAX and scattered > C.GATE_SCATTERED_MAX:
        reasons.append(f"{scattered:.0f}% of the image is strongly coloured in several unrelated hues, "
                       "but a banknote has one dominant tint")
    if sat_pct > C.GATE_SATURATED_MAX:
        reasons.append(f"{sat_pct:.0f}% of pixels are intensely saturated, but banknote inks are muted")
    if flat_pct > C.GATE_FLAT_MAX:
        reasons.append(f"{flat_pct:.0f}% of the image is flat or blank, but a note is printed with fine detail everywhere")
    if raw_aspect is not None and not (C.GATE_ASPECT[0] <= raw_aspect <= C.GATE_ASPECT[1]):
        reasons.append(f"the shape is {raw_aspect:.2f}:1, but a banknote is a long rectangle (about 2:1)")
    return dict(ok=not reasons, reasons=reasons,
                metrics=dict(off_palette=off_pct, saturated=sat_pct, flat=flat_pct, aspect=raw_aspect,
                             dominant_hue=dom_hue, dominant_share=dom_pct, scattered=scattered))