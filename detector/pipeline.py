"""Stage 3: run every check, fuse scores, produce a verdict + annotated image."""
from pathlib import Path

import cv2

from . import config as C
from .checks import run_all, load_references, load_wm_references
from .preprocess import rectify, draw_quad

REF_DIR = Path(__file__).resolve().parent.parent / "references"


def get_references():
    paths = [p for p in REF_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return load_references(paths)


def get_wm_references():
    d = REF_DIR / "watermark"
    return load_wm_references([q for q in d.glob("*") if q.suffix.lower() in {".jpg", ".jpeg", ".png"}]) if d.exists() else []


def fuse(results, weights=None):
    """Weakest-link fusion: 60% weighted mean + 40% mean of the two lowest scores.
    A security screen should not let strong checks hide a clearly failed one."""
    weights = weights or C.WEIGHTS
    active = [r for r in results if r.score is not None]
    if not active:
        return 0.0
    wsum = sum(weights.get(r.name, 1.0) for r in active)
    wmean = sum(weights.get(r.name, 1.0) * r.score for r in active) / wsum
    lowest = sorted(r.score for r in active)[:2]
    return 0.6 * wmean + 0.4 * (sum(lowest) / len(lowest))


def verdict_for(score, genuine_at=C.GENUINE_AT, suspicious_at=C.SUSPICIOUS_AT):
    if score >= genuine_at:
        return "GENUINE"
    if score >= suspicious_at:
        return "SUSPICIOUS"
    return "LIKELY FAKE"


def annotate(note, results):
    vis = note.copy()
    h, w = vis.shape[:2]
    for r in results:
        if r.roi is None or r.score is None:
            continue
        x0, y0, x1, y1 = C.ROIS[r.roi]
        colour = (80, 220, 100) if r.passed else (60, 60, 240)
        p1, p2 = (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h))
        cv2.rectangle(vis, p1, p2, colour, 3)
        if r.roi == "serial" and "serial2" in C.ROIS:
            a0, b0, a1, b1 = C.ROIS["serial2"]
            cv2.rectangle(vis, (int(a0 * w), int(b0 * h)), (int(a1 * w), int(b1 * h)), colour, 3)
        label = f"{r.name} {r.score*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (p1[0], max(p1[1] - th - 10, 0)), (p1[0] + tw + 8, max(p1[1], th + 10)), colour, -1)
        cv2.putText(vis, label, (p1[0] + 4, max(p1[1] - 5, th + 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return vis


def _already_cropped(img):
    """Treat the whole image as the note (used when the image was cropped/straightened already)."""
    note = cv2.resize(img, (C.NOTE_W, C.NOTE_H), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(cv2.cvtColor(note, cv2.COLOR_BGR2GRAY), 40, 120)
    return dict(note=note, quad=None, edges=edges, found=True)


def analyze(img_bgr, refs=None, weights=None, genuine_at=C.GENUINE_AT, suspicious_at=C.SUSPICIOUS_AT,
            backlit_bgr=None, wm_refs=None, assume_cropped=False):
    refs = get_references() if refs is None else refs
    rect = _already_cropped(img_bgr) if assume_cropped else rectify(img_bgr)
    note = rect["note"]
    backlit_note = rectify(backlit_bgr)["note"] if backlit_bgr is not None else None
    wm_refs = get_wm_references() if wm_refs is None else wm_refs
    results = run_all(note, refs, backlit_note, wm_refs)
    score = fuse(results, weights)
    return dict(
        original=img_bgr, outline=draw_quad(img_bgr, rect["quad"]), edges=rect["edges"],
        note=note, note_found=rect["found"], results=results, score=score,
        verdict=verdict_for(score, genuine_at, suspicious_at), annotated=annotate(note, results),
    )