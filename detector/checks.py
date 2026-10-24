"""Stage 2: one classical image-processing check per security/print feature.

Every check returns a CheckResult with a 0..1 score (1 = looks genuine),
a human-readable explanation, and optionally a debug image for the dashboard.
A score of None means the check was skipped (e.g. no reference images).
"""
import re
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from . import config as C
from .preprocess import rectify


@dataclass
class CheckResult:
    name: str
    score: Optional[float]
    detail: str
    metric: str = ""
    roi: Optional[str] = None
    debug: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def passed(self):
        return self.score is not None and self.score >= 0.5


def _scale(v, lo, hi):
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def _crop(img, roi):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


# ---------------------------------------------------------------- 1. Thread
def check_security_thread(note):
    gray = cv2.cvtColor(note, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    x0, _, x1, _ = C.ROIS["thread"]
    x0, x1 = int(x0 * w), int(x1 * w)
    pad = 14
    x0, x1 = max(x0, pad), min(x1, w - pad)
    band = gray[:, x0:x1]
    side = np.stack([gray[:, x0 - 12:x1 - 12], gray[:, x0 - 8:x1 - 8],
                     gray[:, x0 + 8:x1 + 8], gray[:, x0 + 12:x1 + 12]])
    diff = band - np.median(side, axis=0)          # median ignores stray line-art on one side
    dark = (diff < -14).mean(axis=0)
    bright = (diff > 14).mean(axis=0)
    cont = np.maximum(dark, bright)
    cont = np.convolve(cont, np.ones(3) / 3, mode="same")
    best = int(np.argmax(cont))
    best_val = float(cont[best])

    dbg = note.copy()
    cv2.line(dbg, (x0 + best, 0), (x0 + best, h), (0, 255, 255), 2)
    return CheckResult(
        "Security thread", _scale(best_val, *C.THREAD_CONT_RANGE),
        f"Strongest vertical line is continuous over {best_val*100:.0f}% of the note height "
        f"(genuine notes carry an unbroken embedded thread).",
        f"continuity = {best_val:.2f}", "thread", dbg)


# ------------------------------------------------------------ 2. Sharpness
def check_sharpness(note):
    gray = cv2.cvtColor(note, cv2.COLOR_BGR2GRAY)
    # measure at a fixed width so the score does not depend on how many pixels the photo had
    scale = C.SHARPNESS_WIDTH / gray.shape[1]
    g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    var = float(lap.var())
    vis = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    return CheckResult(
        "Print sharpness", _scale(var, *C.SHARPNESS_RANGE),
        "Intaglio / offset printing gives crisp fine lines. Photocopies and re-prints "
        "usually lose edge detail, lowering Laplacian variance (measured at a fixed 500 px width).",
        f"Laplacian var = {var:.0f}", None, vis)


# ------------------------------------------------------- 3. Halftone / moire
def fft_spectrum(gray_patch):
    win = np.outer(np.hanning(gray_patch.shape[0]), np.hanning(gray_patch.shape[1]))
    f = np.fft.fftshift(np.fft.fft2((gray_patch.astype(np.float32) - gray_patch.mean()) * win))
    return np.abs(f)


def check_halftone(note):
    gray = cv2.cvtColor(note, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    n = 256
    cy, cx = h // 2, int(w * 0.8)
    cx = min(max(cx, n // 2), w - n // 2)
    patch = gray[cy - n // 2:cy + n // 2, cx - n // 2:cx + n // 2]
    mag = fft_spectrum(patch)

    yy, xx = np.indices(mag.shape)
    r = np.hypot(yy - n / 2, xx - n / 2)
    ring = (r > 0.25 * n / 2) & (r < 0.95 * n / 2)
    ring &= (np.abs(yy - n / 2) > 3) & (np.abs(xx - n / 2) > 3)     # drop axis streaks
    vals = mag[ring]
    ratio = float(vals.max() / (np.median(vals) + 1e-6))

    vis = np.log1p(mag)
    vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.applyColorMap(cv2.resize(vis, (384, 384), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_MAGMA)

    score = 1.0 - _scale(ratio, *C.HALFTONE_RANGE)
    return CheckResult(
        "Halftone / moire", score,
        "Inkjet/laser reproductions are built from regular dot grids, which show up as sharp "
        "isolated peaks in the frequency spectrum. Genuine print gives a smooth spectrum.",
        f"peak/median = {ratio:.1f}", None, vis)


# -------------------------------------------------------------- 4. Saturation
def check_saturation(note):
    hsv = cv2.cvtColor(note, cv2.COLOR_BGR2HSV)
    mean_s = float(hsv[..., 1].mean())
    hue = cv2.calcHist([hsv], [0], None, [180], [0, 180]).ravel()
    hue /= hue.sum() + 1e-9
    vis = np.zeros((120, 360, 3), np.uint8)
    for i in range(180):
        col = cv2.cvtColor(np.uint8([[[i, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        hgt = int(min(hue[i] * 120 * 12, 118))
        cv2.rectangle(vis, (i * 2, 119 - hgt), (i * 2 + 1, 119), col, -1)
    return CheckResult(
        "Ink saturation", _scale(mean_s, *C.SATURATION_RANGE),
        "Genuine notes use vivid, specific ink colours. Black & white photocopies and "
        "washed-out prints have very low saturation.",
        f"mean S = {mean_s:.0f}/255", None, vis)


# ------------------------------------------------------------ 5. LBP texture
def _lbp(gray):
    c = gray[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(shifts):
        nb = gray[1 + dy:gray.shape[0] - 1 + dy, 1 + dx:gray.shape[1] - 1 + dx]
        code |= ((nb >= c).astype(np.uint8) << i)
    return code


def _lbp_hist(note):
    roi = _crop(cv2.cvtColor(note, cv2.COLOR_BGR2GRAY), C.ROIS["portrait"])
    code = _lbp(roi)
    hist = np.bincount(code.ravel(), minlength=256).astype(np.float64)
    return code, hist / hist.sum()


def check_texture(note, refs=None):
    """With references: histogram intersection of LBP micro-texture vs a genuine note
    (robust - noise from photocopies does NOT fool it). Without: weak entropy fallback."""
    code, p = _lbp_hist(note)
    vis = cv2.applyColorMap(cv2.normalize(code, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_TWILIGHT)
    if refs:
        sim = max(float(np.minimum(p, _lbp_hist(r[1])[1]).sum()) for r in refs)
        return CheckResult(
            "Intaglio texture", _scale(sim, *C.LBP_SIM_RANGE),
            "Local Binary Pattern micro-texture of the portrait region compared with a genuine "
            "note (histogram intersection). Raised-ink printing has a distinctive texture signature.",
            f"LBP similarity = {sim:.2f}", "portrait", vis)
    q = p[p > 0]
    entropy = float(-(q * np.log2(q)).sum())
    return CheckResult(
        "Intaglio texture", _scale(entropy, *C.LBP_ENTROPY_RANGE) * 0.5 + 0.25,
        "No reference available: using a weak LBP-entropy fallback (low confidence).",
        f"LBP entropy = {entropy:.2f} bits", "portrait", vis)


# --------------------------------------------------- 6. Reference (ORB) match
_orb = cv2.ORB_create(nfeatures=2500)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _describe(img):
    gray = _clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return _orb.detectAndCompute(gray, None)


def load_references(paths):
    refs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (C.NOTE_W, C.NOTE_H))
        kp, des = _describe(img)
        if des is not None:
            refs.append((str(p), img, kp, des))
    return refs


def check_reference_match(note, refs):
    if not refs:
        return CheckResult("Reference match", None,
                           "Skipped - add genuine note images to the references/ folder to enable this check.")
    kp, des = _describe(note)
    if des is None or len(kp) < 20:
        return CheckResult("Reference match", 0.0, "Too few keypoints found in the note.", "keypoints < 20")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    best = (-1.0, None, None, 0, 0)
    for name, rimg, rkp, rdes in refs:
        pairs = bf.knnMatch(des, rdes, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 10:
            ratio, mask = 0.0, None
        else:
            src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([rkp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            ratio = float(mask.sum()) / len(good) if mask is not None else 0.0
        inl = int(mask.sum()) if mask is not None else 0
        if inl > best[3] or best[1] is None:
            best = (ratio, (name, rimg, rkp, good, mask), None, inl, len(good))
    ratio, pack, _, inl, ngood = best
    name, rimg, rkp, good, mask = pack
    matches_mask = mask.ravel().tolist() if mask is not None else None
    vis = cv2.drawMatches(note, kp, rimg, rkp, good, None, matchesMask=matches_mask,
                          matchColor=(0, 255, 120), singlePointColor=None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    score = _scale(ratio, *C.ORB_INLIER_RANGE) if inl >= 10 else 0.0
    return CheckResult(
        "Reference match", score,
        f"ORB features matched against a genuine reference and verified with RANSAC homography "
        f"({inl} geometric inliers out of {ngood} good matches).",
        f"inlier ratio = {ratio:.2f}", None, vis)


# ------------------------------------------------------------ 7. Serial (OCR)
SERIAL_STRICT = re.compile(r"[0-9][A-Z]{2}\s?[0-9]{6}")
_TO_DIGIT = str.maketrans("SOIBZGQDT", "501826009")      # common OCR look-alikes
_TO_LETTER = str.maketrans("501826", "SOIBZG")


def correct_serial(text):
    """Slide a 9-char window over the OCR text and fix look-alike characters by position
    (digit, letter, letter, 6 digits). Returns the corrected serial or None."""
    t = re.sub(r"[^0-9A-Z]", "", text.upper())
    for i in range(len(t) - 8):
        w = t[i:i + 9]
        fixed = w[0].translate(_TO_DIGIT) + w[1:3].translate(_TO_LETTER) + w[3:].translate(_TO_DIGIT)
        if SERIAL_STRICT.fullmatch(fixed):
            return fixed[:3] + " " + fixed[3:]
    return None


def _canon(serial):
    """Compare serials tolerant to OCR look-alikes in the two letter positions (Q/O/D/G/C/0)."""
    d = serial.replace(" ", "")
    return d[0] + d[1:3].translate(str.maketrans("QDGC0", "OOOOO")) + d[3:]


def _read_panel(note, key):
    """OCR one number panel with several darkness thresholds; return every valid serial found."""
    import pytesseract
    roi = _crop(note, C.ROIS[key])
    gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), None, fx=3.5, fy=3.5, interpolation=cv2.INTER_CUBIC)
    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw, valid = [], []
    for pct, invert in ((8, False), (14, False), (20, False), (8, True), (14, True)):
        t = np.percentile(gray, 100 - pct if invert else pct)
        mask = (gray >= t) if invert else (gray <= t)
        th = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)   # strips thin line-art
        im = cv2.copyMakeBorder(255 - th, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        try:
            txt = pytesseract.image_to_string(im, config=cfg, timeout=8).strip().replace("\n", " ")
        except Exception:          # Tesseract can crash/time out on degenerate crops - never take the app down
            continue
        if txt:
            raw.append(txt)
        fixed = correct_serial(txt)
        if fixed:
            valid.append(fixed)
    return roi, raw, valid


def check_serial(note):
    """The same serial number is printed twice on the note. Reading both panels and checking
    that they agree is far stronger evidence than a single OCR read matching a format."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        return CheckResult("Serial number", None,
                           "Skipped - install Tesseract + pytesseract to enable OCR of the serial number.")
    roi_a, raw_a, val_a = _read_panel(note, "serial")
    roi_b, raw_b, val_b = _read_panel(note, "serial2") if "serial2" in C.ROIS else (None, [], [])

    pair = next(((a, b) for a in val_a for b in val_b if _canon(a) == _canon(b)), None)
    if pair:
        score, msg = 1.0, f"both number panels read as {pair[0]} / {pair[1]} and agree"
    elif val_a and val_b:
        score, msg = 0.3, f"panels disagree: {val_a[0]} vs {val_b[0]} (genuine notes print the same number twice)"
    elif val_a or val_b:
        score, msg = 0.5, f"only one panel gave a valid serial ({(val_a or val_b)[0]}); could not cross-check"
    elif any(sum(ch.isdigit() for ch in t) >= 6 for t in raw_a + raw_b):
        score, msg = 0.25, "digits found but no valid serial format could be read"
    else:
        score, msg = 0.0, "no serial number could be read"

    dbg = roi_a if roi_b is None else np.vstack([cv2.resize(roi_a, (400, 90)), cv2.resize(roi_b, (400, 90))])
    return CheckResult("Serial number", score, f"Format: digit-letter-letter-6 digits. Result: {msg}.",
                       "bottom-right vs top-left panel", "serial", dbg)


# ---------------------------------------- 8. Watermark window (normal light)
def check_watermark_window(note):
    roi = _crop(note, C.ROIS["watermark"])
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
    edges = cv2.Canny(gray, 25, 75)
    density = float((edges > 0).mean())
    return CheckResult(
        "Watermark window", 1.0 - _scale(density, *C.WM_EDGE_RANGE),
        "In normal light the watermark window of a genuine note is blank - the portrait only shows "
        "against light. A visibly printed portrait or pattern here is a classic counterfeit sign.",
        f"edge density = {density*100:.1f}%", "watermark", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))


# ------------------------------------------- 9. Watermark (transmitted light)
def _wm_metrics(note):
    gray = cv2.cvtColor(_crop(note, C.ROIS["watermark"]), cv2.COLOR_BGR2GRAY).astype(np.float32)
    w = gray.shape[1]
    # band-pass at the scale of the watermark portrait; removes backlight gradient and paper fibres
    band = cv2.GaussianBlur(gray, (0, 0), max(w / 30.0, 2.0)) - cv2.GaussianBlur(gray, (0, 0), w / 5.0)
    band_rel = float(band.std()) / (float(gray.mean()) + 1.0) * 100.0
    return band, band_rel


def load_wm_references(paths):
    out = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            out.append((str(p), _wm_metrics(rectify(img)["note"])[0]))
    return out


def check_watermark_backlit(trans, wm_refs=None):
    if trans is None:
        return CheckResult("Watermark (backlit)", None,
                           "Skipped - add a second photo of the note held against a light to check the watermark.")
    band, band_rel = _wm_metrics(trans)
    score = _scale(band_rel, *C.WM_BAND_RANGE)     # presence of smooth structure inside the paper
    extra = ""
    if wm_refs:
        padded = cv2.copyMakeBorder(band, 10, 10, 10, 10, cv2.BORDER_REFLECT)
        best = -1.0
        for _, rb in wm_refs:
            t = cv2.resize(rb, (band.shape[1], band.shape[0]))
            res = np.nan_to_num(cv2.matchTemplate(padded, t, cv2.TM_CCOEFF_NORMED))
            best = max(best, float(res.max()))
        score = 0.4 * score + 0.6 * _scale(best, *C.WM_NCC_RANGE)
        extra = f"; correlation with genuine watermark = {best:.2f}"
    m = float(np.abs(band).max()) + 1e-6
    vis = cv2.applyColorMap(((band / m * 0.5 + 0.5) * 255).astype(np.uint8), cv2.COLORMAP_BONE)
    vis = cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return CheckResult(
        "Watermark (backlit)", score,
        "Against light, a genuine watermark shows as a smooth tonal portrait inside the paper. Plain paper "
        "shows almost nothing. This proves something is in the paper; the normal-light window check "
        "separately proves it is not just printed on the surface" + extra + ".",
        f"band contrast = {band_rel:.1f}%", None, vis)


def run_all(note, refs, backlit=None, wm_refs=None):
    return [check_security_thread(note), check_sharpness(note), check_halftone(note),
            check_saturation(note), check_texture(note, refs),
            check_reference_match(note, refs), check_serial(note),
            check_watermark_window(note), check_watermark_backlit(backlit, wm_refs)]