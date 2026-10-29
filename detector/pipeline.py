"""Stage 3: run every check, fuse scores, produce a verdict + annotated image."""
from pathlib import Path

import cv2

from . import config as C
from .checks import run_all, load_references, load_wm_references, best_inliers
from .preprocess import rectify, draw_quad

REF_DIR = Path(__file__).resolve().parent.parent / "references"


def get_references(value=500):
    """Genuine reference images for one denomination: references/<value>/ (and, for the Rs 500, the files in references/)."""
    ok = {".jpg", ".jpeg", ".png"}
    paths = [p for p in REF_DIR.glob("*") if p.suffix.lower() in ok] if value == 500 else []
    d = REF_DIR / str(value)
    if d.exists():
        paths += [p for p in d.glob("*") if p.suffix.lower() in ok]
    return load_references(paths)


def get_reference_sets():
    """{value: references} for every denomination that has genuine reference images (used to recognise a note by its design)."""
    values = {500} | ({int(d.name) for d in REF_DIR.iterdir() if d.is_dir() and d.name.isdigit()} if REF_DIR.exists() else set())
    return {v: get_references(v) for v in sorted(values)}


def get_all_references():
    """Obverse references of every denomination (used to tell front from back before the denomination is known)."""
    ok = {".jpg", ".jpeg", ".png"}
    paths = [q for q in REF_DIR.glob("*") if q.suffix.lower() in ok]
    for d in (REF_DIR.iterdir() if REF_DIR.exists() else []):
        if d.is_dir() and d.name.isdigit():
            paths += [q for q in d.glob("*") if q.suffix.lower() in ok]
    return load_references(paths)


def get_side_references():
    d = REF_DIR / "sides"
    return load_references([q for q in d.glob("*") if q.suffix.lower() in {".jpg", ".jpeg", ".png"}]) if d.exists() else []


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


def verdict_for(score, genuine_at=C.GENUINE_AT, suspicious_at=C.SUSPICIOUS_AT, failed=None):
    if score >= genuine_at:
        return "GENUINE"
    if score >= suspicious_at:
        return "SUSPICIOUS"
    # consensus rule: one or two odd checks (bad lighting, glare) must not condemn a note as a fake
    if failed is not None and failed < C.MIN_FAILED_FOR_FAKE:
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
    return dict(note=note, note_hr=cv2.resize(img, (2 * C.NOTE_W, 2 * C.NOTE_H), interpolation=cv2.INTER_CUBIC),
                quad=None, edges=edges, found=True)


def analyze(img_bgr, refs=None, weights=None, genuine_at=C.GENUINE_AT, suspicious_at=C.SUSPICIOUS_AT,
            backlit_bgr=None, wm_refs=None, assume_cropped=False, limited=False, profile=500):
    prof = C.set_profile(profile)
    refs = get_references(profile) if refs is None else refs
    rect = _already_cropped(img_bgr) if assume_cropped else rectify(img_bgr)
    note = rect["note"]
    backlit_note = rectify(backlit_bgr)["note"] if backlit_bgr is not None else None
    wm_refs = get_wm_references() if wm_refs is None else wm_refs
    results = run_all(note, refs, backlit_note, wm_refs, note_hr=rect.get("note_hr"), generic_only=limited)
    score = fuse(results, weights)
    failed = sum(1 for r in results if r.score is not None and r.score < 0.5)
    if limited or not prof["validated"]:
        # Generic checks, or a layout measured from an official specimen only: they can show signs of copying but can neither
        # confirm a note nor condemn it, so the verdict never says "Genuine" or "Likely Fake".
        verdict = "NO RED FLAGS" if score >= genuine_at else "SUSPICIOUS"
    else:
        verdict = verdict_for(score, genuine_at, suspicious_at, failed=failed)
    proof = [r for r in results if r.decisive and r.score is not None and r.score < 0.5]
    if proof:      # positive proof of a non-genuine note (e.g. "CHILDREN BANK") beats any amount of design conformity
        verdict, score = "LIKELY FAKE", min(score, 0.15)
    return dict(
        original=img_bgr, outline=draw_quad(img_bgr, rect["quad"]), edges=rect["edges"], decisive=[(r.name, r.detail) for r in proof],
        note=note, note_found=rect["found"], results=results, score=score, verdict=verdict,
        annotated=annotate(note, results),
    )