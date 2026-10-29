"""Stage 2: one classical image-processing check per security/print feature.

Every check returns a CheckResult with a 0..1 score (1 = looks genuine),
a human-readable explanation, and optionally a debug image for the dashboard.
A score of None means the check was skipped (e.g. no reference images).
"""
import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
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
    decisive: bool = False        # a failing decisive check is positive proof of a non-genuine note and overrides the fused score

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
def _dash_count(mask, h, w):
    """Find thread dashes in a 2-D mask (rows x band columns), tolerant to a tilted note: take narrow vertical blobs,
    fit one line x = a*y + b through their centres and count the blobs lying on it. Returns (count, coverage)."""
    k = max(int(0.012 * h), 2)
    m = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                         np.ones((max(int(0.02 * h), 5), max(int(0.012 * w), 5)), np.uint8))   # text strokes are too thin to survive
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((2 * k + 1, 3), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
    blobs = [(cen[i][0], cen[i][1], st[i][3], st[i][2]) for i in range(1, n)
             if st[i][3] >= 0.03 * h and st[i][2] <= 0.06 * w and st[i][3] >= 0.8 * st[i][2]]
    best = []
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            (x1, y1, _, _), (x2, y2, _, _) = blobs[i], blobs[j]
            if abs(y2 - y1) < 0.05 * h:
                continue
            a = (x2 - x1) / (y2 - y1)
            if abs(a) > 0.35:                                   # a thread is near-vertical
                continue
            on = [bl for bl in blobs if abs(bl[0] - (x1 + a * (bl[1] - y1))) <= 0.012 * w + 4]
            if len(on) >= 2:                       # dashes of one thread have similar widths; random texture does not
                mw = float(np.median([bl[3] for bl in on]))
                on = [bl for bl in on if 0.6 * mw <= bl[3] <= 1.6 * mw]
            if len(on) > len(best):
                best = on
    cover = sum(bl[2] for bl in best) / h
    return len(best), float(cover), best


def _recount_on_line(gray, x_off, blobs, h, w):
    """Blob detection misses dashes sitting on dark/coloured artwork (low contrast). Once two or more strong dashes fix the
    thread's line, walk down that line and count every stretch that is darker than the paper on both sides of it.
    Returns (count, covered_rows_fraction, runs[(y0, y1)])."""
    if len(blobs) < 2:
        return len(blobs), 0.0, []
    ys = np.array([b[1] for b in blobs]); xs = np.array([b[0] + x_off for b in blobs])
    a, b0 = np.polyfit(ys, xs, 1)
    mw = max(int(np.median([b[3] for b in blobs])), 4)
    prof = np.zeros(h, np.float32)
    for y in range(h):
        cx = int(round(a * y + b0))
        c0, c1 = cx - mw // 2, cx + mw // 2 + 1
        if c0 - mw - 6 < 0 or c1 + mw + 6 > w:
            continue
        centre = gray[y, c0:c1].mean()
        side = 0.5 * (gray[y, c0 - mw - 6:c0 - 6].mean() + gray[y, c1 + 6:c1 + mw + 6].mean())
        prof[y] = side - centre                        # > 0: darker than its surroundings
    prof = np.convolve(prof, np.ones(5) / 5, mode="same")
    on = prof > 14
    runs, y = [], 0
    while y < h:
        if on[y]:
            y0 = y
            while y < h and on[y]:
                y += 1
            if y - y0 >= 0.022 * h and y0 > 0.03 * h and y < 0.97 * h:    # at least ~2% of the height; the very edge rows are border/margin, not thread
                runs.append((y0, y))
        else:
            y += 1
    return len(runs), float(sum(y1 - y0 for y0, y1 in runs)) / h, runs


def check_security_thread(note):
    """Genuine Rs 500 notes carry a windowed thread: continuous in transmitted light, but in reflected light it shows
    as a column of dark dashes. Score the better of (a) one continuous line, (b) several aligned dashes."""
    gray = cv2.cvtColor(note, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    x0, _, x1, _ = C.ROIS["thread"]
    x0, x1 = int(x0 * w), int(x1 * w)
    offs = C.THREAD_SIDE_OFFSETS
    pad = max(offs)
    x0, x1 = max(x0, pad), min(x1, w - pad)
    band = gray[:, x0:x1]
    side = np.stack([gray[:, x0 + sgn * o:x1 + sgn * o] for o in offs for sgn in (-1, 1)])
    diff = band - np.median(side, axis=0)          # median ignores stray line-art on one side
    dark_m, bright_m = diff < -14, diff > 14
    cont = np.maximum(dark_m.mean(axis=0), bright_m.mean(axis=0))
    cont = np.convolve(cont, np.ones(3) / 3, mode="same")
    best = int(np.argmax(cont))
    best_val = float(cont[best])

    # dashed thread: repeated narrow dark (or bright) blobs that line up, even if the note is tilted
    g8 = np.clip(gray, 0, 255).astype(np.uint8)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(0.06 * w) | 1, 41), 1))
    bh = cv2.morphologyEx(g8, cv2.MORPH_BLACKHAT, kern)[:, x0:x1]       # dark features narrower than the kernel
    th = cv2.morphologyEx(g8, cv2.MORPH_TOPHAT, kern)[:, x0:x1]         # bright features narrower than the kernel
    dashes, cover, blobs = max((_dash_count(m, h, w) for m in (bh > 22, th > 22)), key=lambda t: t[0])
    runs = []
    if dashes >= 2:                                   # re-count along the fitted line: finds dashes the blob filter lost on dark art
        n2, cover2, runs = _recount_on_line(gray, x0, blobs, h, w)
        if n2 > dashes:
            dashes, cover = n2, max(cover, cover2)
    s_cont = _scale(best_val, *C.THREAD_CONT_RANGE)
    s_dash = _scale(dashes, 2, 5) * _scale(cover, 0.05, 0.15)
    score = max(s_cont, s_dash)

    dbg = note.copy()
    cv2.line(dbg, (x0 + best, 0), (x0 + best, h), (0, 255, 255), 1)
    for cx, cy, bh, bw in blobs:                      # show exactly which dashes were counted
        cv2.rectangle(dbg, (int(x0 + cx - bw / 2) - 3, int(cy - bh / 2)), (int(x0 + cx + bw / 2) + 3, int(cy + bh / 2)), (255, 200, 0), 3)
    for y0, y1 in runs:                               # recounted dashes in orange
        bx = int(x0 + np.mean([b[0] for b in blobs]))
        cv2.rectangle(dbg, (bx - 12, y0), (bx + 12, y1), (0, 160, 255), 2)
    return CheckResult(
        "Security thread", score,
        f"Strongest vertical line is continuous over {best_val*100:.0f}% of the note height, or shows {dashes} aligned "
        f"dashes (a windowed thread looks dashed in reflected light). The better of the two is scored.",
        f"continuity = {best_val:.2f}, dashes = {dashes}", "thread", dbg)


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
    decisive = ratio >= C.HALFTONE_DECISIVE
    detail = ("Inkjet/laser reproductions are built from regular dot grids, which show up as sharp "
              "isolated peaks in the frequency spectrum. Genuine print gives a smooth spectrum.")
    if decisive:
        detail = (f"A printer's dot grid is unmistakable here (peak/median {ratio:.0f}, genuine notes stay far below {C.HALFTONE_DECISIVE:.0f}). "
                  "This is a reprint. If you photographed a screen instead of the paper note, retake the photo of the note itself.")
    return CheckResult("Halftone / moire", 0.0 if decisive else score, detail, f"peak/median = {ratio:.1f}", None, vis, decisive=decisive)


# -------------------------------------------------------------- 4. Saturation
def check_saturation(note, generic=False):
    hsv = cv2.cvtColor(note, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mean_s = float(S.mean())
    lo, hi = C.INK_HUE
    ink_pct = float(((S >= 25) & (V > 40) & (V < 235) & (H >= lo) & (H <= hi)).mean() * 100)
    # Mean saturation alone depends on lighting/white balance (a real note photographed indoors can look grey).
    # The note's own green/olive ink is a better signature: a black-and-white copy has none, even with a paper tint.
    score = _scale(mean_s, *C.SATURATION_RANGE) if generic else max(_scale(ink_pct, *C.INK_RANGE), _scale(mean_s, *C.SATURATION_RANGE))
    hue = cv2.calcHist([hsv], [0], None, [180], [0, 180]).ravel()
    hue /= hue.sum() + 1e-9
    vis = np.zeros((120, 360, 3), np.uint8)
    for i in range(180):
        col = cv2.cvtColor(np.uint8([[[i, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        hgt = int(min(hue[i] * 120 * 12, 118))
        cv2.rectangle(vis, (i * 2, 119 - hgt), (i * 2 + 1, 119), col, -1)
    no_colour = mean_s < C.NO_COLOUR_S
    detail = ("Genuine notes carry the vivid ink of their denomination (olive-green on the Rs 500, lavender on the Rs 100, magenta on the Rs 2000, ...). "
              "Black & white photocopies and washed-out prints have none, even when the paper looks tinted. Judged on the ink itself, "
              "so a grey-looking phone photo is not penalised.")
    if no_colour:
        detail = (f"The image has no colour at all (mean saturation {mean_s:.0f}/255) and every Indian note is printed in colour, so this is a "
                  "black-and-white photocopy or a grayscale conversion. If it is a photo of a real note, upload the colour original.")
    return CheckResult("Ink saturation", 0.0 if no_colour else score, detail,
                       f"note ink = {ink_pct:.1f}% of pixels, mean S = {mean_s:.0f}/255", None, vis, decisive=no_colour)


# ------------------------------------------------------------ 5. LBP texture
def _lbp(gray):
    c = gray[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(shifts):
        nb = gray[1 + dy:gray.shape[0] - 1 + dy, 1 + dx:gray.shape[1] - 1 + dx]
        code |= ((nb >= c).astype(np.uint8) << i)
    return code


LBP_WIDTH = 128


def _lbp_hist(note):
    roi = _crop(cv2.cvtColor(note, cv2.COLOR_BGR2GRAY), C.ROIS["portrait"])
    # fixed analysis width: LBP is resolution-sensitive, so a 1600 px scan and a 776 px reference must be compared at the same scale
    roi = cv2.resize(roi, (LBP_WIDTH, max(int(LBP_WIDTH * roi.shape[0] / max(roi.shape[1], 1)), 8)), interpolation=cv2.INTER_AREA)
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


_REF_CACHE = {}


def load_references(paths):
    """Decode + describe reference images. Cached by (path, mtime, size): Streamlit reruns the script on every widget change."""
    refs = []
    for p in paths:
        try:
            stt = Path(p).stat()
            key = (str(p), stt.st_mtime_ns, stt.st_size)
        except OSError:
            continue
        if key not in _REF_CACHE:
            img = cv2.imread(str(p))
            if img is None:
                _REF_CACHE[key] = None
            else:
                img = cv2.resize(img, (C.NOTE_W, C.NOTE_H))
                kp, des = _describe(img)
                _REF_CACHE[key] = (str(p), img, kp, des) if des is not None else None
        if _REF_CACHE[key] is not None:
            refs.append(_REF_CACHE[key])
    return refs


def check_reference_match(note, refs):
    if not refs:
        return CheckResult("Reference match", None,
                           "Skipped - add genuine note images to the references/ folder to enable this check.")
    kp, des = _describe(note)
    if des is None or len(kp) < 20:
        return CheckResult("Reference match", None, "Inconclusive - too few keypoints found in the note (not counted).")
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
    if inl < 10:
        # A weak/blurry/small reference or a different framing fails to align even for a genuine note, and a
        # missing match is no proof of forgery - so this check gives positive evidence only, never a penalty.
        return CheckResult("Reference match", None,
                           f"Inconclusive - could not align the note with the reference ({inl} inliers). Add a sharp, "
                           "full-note reference image for a stronger check. Not counted.", f"{inl} inliers", None, vis)
    score = _scale(ratio, *C.ORB_INLIER_RANGE)
    return CheckResult(
        "Reference match", score,
        f"ORB features matched against a genuine reference and verified with RANSAC homography "
        f"({inl} geometric inliers out of {ngood} good matches).",
        f"inlier ratio = {ratio:.2f}", None, vis)


def best_inliers(note, refs):
    """Largest RANSAC-verified ORB match count between the note and any reference (0 if none / too few features)."""
    if not refs:
        return 0
    kp, des = _describe(note)
    if des is None or len(kp) < 20:
        return 0
    bf, best = cv2.BFMatcher(cv2.NORM_HAMMING), 0
    for _, _, rkp, rdes in refs:
        pairs = bf.knnMatch(des, rdes, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 10:
            continue
        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([rkp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        best = max(best, int(mask.sum()) if mask is not None else 0)
    return best


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
    fx = float(np.clip(220.0 / max(roi.shape[0], 1), 1.0, 3.5))      # aim for ~220 px tall text crops
    gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
    white = "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw, valid = [], []
    t_start = time.time()
    for pct, invert in ((8, False), (14, False), (20, False), (8, True), (14, True)):
        if time.time() - t_start > 3.0:      # time budget per panel so a hopeless crop cannot stall the app
            break
        t = np.percentile(gray, 100 - pct if invert else pct)
        mask = (gray >= t) if invert else (gray <= t)
        th = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)   # strips thin line-art
        im = cv2.copyMakeBorder(255 - th, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        for psm in (7, 6):          # 7 = one line; 6 = block (Tesseract finds the line itself when the crop is generous)
            try:
                txt = pytesseract.image_to_string(im, config=f"--psm {psm} {white}", timeout=3).strip().replace("\n", " ")
            except Exception:       # Tesseract can crash/time out on degenerate crops - never take the app down
                continue
            if txt:
                raw.append(txt)
            fixed = correct_serial(txt)
            if fixed:
                valid.append(fixed)
        if len(valid) >= 2:         # enough agreeing reads - stop early to keep the app fast
            break
    return roi, raw, valid


_SERIAL_CACHE = {}


def check_serial(note):
    """The same serial number is printed twice on the note. Reading both panels and checking
    that they agree is far stronger evidence than a single OCR read matching a format."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        return CheckResult("Serial number", None,
                           "Skipped - install Tesseract + pytesseract to enable OCR of the serial number.")
    cache_key = hashlib.md5(note.tobytes()).hexdigest()
    if cache_key in _SERIAL_CACHE:                       # OCR is slow; sliders/toggles rerun the page, so remember it
        return _SERIAL_CACHE[cache_key]
    roi_a, raw_a, val_a = _read_panel(note, "serial")
    roi_b, raw_b, val_b = _read_panel(note, "serial2") if "serial2" in C.ROIS else (None, [], [])

    pair = next(((a, b) for a in val_a for b in val_b if _canon(a) == _canon(b)), None)
    if pair:
        score, msg = 1.0, f"both number panels read as {pair[0]} / {pair[1]} and agree"
    else:
        # OCR on a phone photo is unreliable, so a mismatch is not evidence of forgery. Two independent reads
        # that agree are strong evidence (chance agreement is negligible); anything else is inconclusive.
        dbg = roi_a if roi_b is None else np.vstack([cv2.resize(roi_a, (400, 90)), cv2.resize(roi_b, (400, 90))])
        res = CheckResult("Serial number", None,
                          "Inconclusive - the two serial panels could not be read to the same number (blur, glare or "
                          "font). Not counted: an unreadable photo is not evidence of a fake.", "no agreement", "serial", dbg)
        _SERIAL_CACHE[cache_key] = res
        return res

    dbg = roi_a if roi_b is None else np.vstack([cv2.resize(roi_a, (400, 90)), cv2.resize(roi_b, (400, 90))])
    res = CheckResult("Serial number", score, f"Format: digit-letter-letter-6 digits. Result: {msg}.",
                      "bottom-right vs top-left panel", "serial", dbg)
    _SERIAL_CACHE[cache_key] = res
    return res


# ---------------------------------------- 8. Watermark window (normal light)
def check_watermark_window(note):
    roi = _crop(note, C.ROIS["watermark"])
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
    edges = cv2.Canny(gray, 25, 75)
    density = float((edges > 0).mean())
    decisive = density >= C.WM_EDGE_DECISIVE and C.PROFILE_VALIDATED       # the window position is only trustworthy on a validated layout
    detail = ("In normal light the watermark window of a genuine note is blank - the portrait only shows "
              "against light. A visibly printed portrait or pattern here is a classic counterfeit sign.")
    if decisive:
        detail = ("A portrait or pattern is printed in the watermark window, visible in normal light. A real watermark is only visible against "
                  "light, so this is a classic counterfeit trick. (Check there is no crease, finger or glare on the window.)")
    return CheckResult("Watermark window", 0.0 if decisive else 1.0 - _scale(density, *C.WM_EDGE_RANGE), detail,
                       f"edge density = {density*100:.1f}%", "watermark", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), decisive=decisive)


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


# ---------------------------------------------- 10. Issuer text (OCR of the whole note)
# A copied DESIGN passes every image check (texture, reference match, ink), but a toy/prop note still prints its own words.
# Genuine notes always read "RESERVE BANK OF INDIA". Only positive evidence counts: unreadable text is inconclusive.
_PROP_WORDS = ("CHILDREN", "MANORANJAN", "CHURAN", "COUPON", "ENTERTAINMENT", "REPLICA", "DUMMY", "MOVIE", "TOY", "TOYS", "PROP", "FAKE")
_ISSUER_CACHE = {}


def _fuzzy(a, b):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def _is(tok, word, strict_short=True):
    """Typo-tolerant word match; very short words must match exactly so OCR noise cannot create them."""
    if len(word) <= 4 and strict_short:
        return tok == word
    return len(tok) >= 3 and _fuzzy(tok, word) >= 0.78


def analyse_issuer_tokens(toks):
    """-> (rbi_found, prop_hits, other_bank). Pure function of the OCR tokens (testable without Tesseract)."""
    rbi, other_bank, props = False, False, []
    for i, t in enumerate(toks):
        if _is(t, "BANK", strict_short=False) and len(t) <= 5:
            before, after = toks[max(0, i - 2):i], toks[i + 1:i + 3]
            if any(_is(x, "RESERVE", False) for x in before) or any(_is(x, "INDIA", False) for x in after):
                rbi = True
            else:
                other_bank = True
                if before and _is(before[-1], "CHILDREN", False):
                    props.append("'CHILDREN BANK'")
        for p in _PROP_WORDS:
            if _is(t, p) and f"'{p}'" not in props:
                props.append(f"'{p}'")
        if t == "FULL" and toks[i + 1:i + 3] == ["OF", "FUN"]:
            props.append("'FULL OF FUN'")
    return rbi, list(dict.fromkeys(props)), other_bank


def _issuer_tokens(note_hr):
    import pytesseract
    h, w = note_hr.shape[:2]
    toks_all = []
    for img in (note_hr, cv2.rotate(note_hr, cv2.ROTATE_180)):            # the note may be upside-down
        whole = cv2.resize(img, (1800, int(1800 * h / w)), interpolation=cv2.INTER_CUBIC)
        best = max(cv2.split(whole), key=lambda c: c.std())                 # highest-contrast channel helps on tinted notes
        for g in (cv2.cvtColor(whole, cv2.COLOR_BGR2GRAY), best):
            try:
                txt = pytesseract.image_to_string(g, config="--psm 11", timeout=10)
            except Exception:
                continue
            toks = [t for t in re.sub(r"[^A-Z]", " ", txt.upper()).split() if len(t) >= 2]
            toks_all.append(toks)
            rbi, props, _ = analyse_issuer_tokens(toks)
            if props or rbi:
                return toks_all
    return toks_all


def check_issuer_text(note_hr):
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        return CheckResult("Issuer text", None, "Skipped - install Tesseract + pytesseract to read the issuer text of the note.")
    key = hashlib.md5(note_hr.tobytes()).hexdigest()
    if key in _ISSUER_CACHE:
        return _ISSUER_CACHE[key]
    runs = _issuer_tokens(note_hr)
    rbi = any(analyse_issuer_tokens(t)[0] for t in runs)
    props = list(dict.fromkeys(p for t in runs for p in analyse_issuer_tokens(t)[1]))
    other_bank = any(analyse_issuer_tokens(t)[2] for t in runs)
    if props:
        res = CheckResult("Issuer text", 0.0,
                          "The note prints words that no Reserve Bank of India note carries: " + ", ".join(props) + ". "
                          "This is a toy, prop or novelty note, not currency, even if its design copies a real note.",
                          "toy/prop text: " + ", ".join(props), None, None, decisive=True)
    elif rbi:
        res = CheckResult("Issuer text", 1.0, "OCR found the issuer line 'RESERVE BANK OF INDIA' printed on the note.",
                          "RESERVE BANK OF INDIA found")
    elif other_bank:
        res = CheckResult("Issuer text", 0.25, "A \"BANK\" line was read, but not \"RESERVE BANK OF INDIA\". Could be a different issuer, "
                          "or a blurry photo - check by eye.", "issuer line differs")
    else:
        res = CheckResult("Issuer text", None, "Inconclusive - the issuer line could not be read (blur, glare or angle). "
                          "Not counted: an unreadable photo is not evidence of a fake.", "not readable")
    _ISSUER_CACHE[key] = res
    return res


def run_all(note, refs, backlit=None, wm_refs=None, note_hr=None, generic_only=False):
    hr = note_hr if note_hr is not None else note
    if generic_only:      # denomination without a calibrated profile: only checks that do not depend on the note's layout
        return [check_sharpness(note), check_halftone(note), check_saturation(note, generic=True), check_issuer_text(hr)]
    return [check_security_thread(note), check_sharpness(note), check_halftone(note),
            check_saturation(note), check_texture(note, refs),
            check_reference_match(note, refs), check_serial(hr), check_issuer_text(hr),
            check_watermark_window(note), check_watermark_backlit(backlit, wm_refs)]