"""Stage 1: locate the note in the photo and perspective-correct it."""
import cv2
import numpy as np
from .config import NOTE_W, NOTE_H


def _order_points(pts):
    pts = np.asarray(pts, dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")  # tl, tr, br, bl


def _looks_like_banknote(quad, hull):
    """Reject blobs that are not rectangular (e.g. a round retina photo) or have the wrong proportions."""
    q = _order_points(quad)
    tl, tr, br, bl = q
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if min(w, h) < 1:
        return False
    aspect = max(w, h) / min(w, h)
    solidity = cv2.contourArea(hull) / max(cv2.contourArea(q.astype("float32")), 1.0)
    return 1.4 <= aspect <= 3.2 and solidity >= 0.88


def find_note_quad(img):
    """Return (quad Nx2 in original coords or None, edge map for visualisation)."""
    scale = 800.0 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.20 * small.shape[0] * small.shape[1]
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:6]:
        if cv2.contourArea(c) < min_area:
            break
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        quad = None
        # finest tolerance that still collapses the hull to exactly 4 corners -> most accurate quad
        for eps in np.linspace(0.004, 0.06, 40):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2)
                break
        if quad is None:
            quad = cv2.boxPoints(cv2.minAreaRect(hull))      # fallback for crumpled notes
        if _looks_like_banknote(quad, hull):
            return quad / scale, edges
    return None, edges


def rectify(img):
    """Returns dict(note=BGR NOTE_WxNOTE_H, quad=..., edges=..., found=bool)."""
    quad, edges = find_note_quad(img)
    if quad is None:
        note = cv2.resize(img, (NOTE_W, NOTE_H), interpolation=cv2.INTER_AREA)
        return dict(note=note, note_hr=cv2.resize(img, (2 * NOTE_W, 2 * NOTE_H), interpolation=cv2.INTER_CUBIC),
                    quad=None, edges=edges, found=False)

    rect = _order_points(quad)
    tl, tr, br, bl = rect
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    portrait = h > w
    out_w, out_h = (NOTE_H, NOTE_W) if portrait else (NOTE_W, NOTE_H)
    def warp(scale):
        ow, oh = (NOTE_H, NOTE_W) if portrait else (NOTE_W, NOTE_H)
        ow, oh = ow * scale, oh * scale
        dst = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], dtype="float32")
        out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(rect, dst), (ow, oh), flags=cv2.INTER_CUBIC)
        return cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE) if portrait else out
    note, note_hr = warp(1), warp(2)       # note_hr keeps native detail for OCR of the serial numbers
    return dict(note=note, note_hr=note_hr, quad=rect, edges=edges, found=True)


def draw_quad(img, quad):
    vis = img.copy()
    if quad is not None:
        cv2.polylines(vis, [quad.astype(int)], True, (0, 255, 200), max(2, img.shape[1] // 300))
    return vis