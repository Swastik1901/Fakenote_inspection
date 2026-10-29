"""Build a 'fake' dataset by degrading genuine note photos.

Real counterfeits are hard to obtain legally, so many academic projects simulate
them by photocopying / re-printing genuine notes. This script mimics that:
  - photocopy : grayscale + contrast crush + blur + noise
  - inkjet    : colour kept but halftone dot screen + slight blur
  - lowres    : heavy downscale/upscale + JPEG artefacts
Usage: python simulate_fake.py dataset/genuine dataset/fake
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from detector import config as C


def photocopy(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.6)
    g = np.clip((g.astype(np.float32) - 110) * 1.6 + 128, 0, 255).astype(np.uint8)
    g = np.clip(g + np.random.normal(0, 8, g.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def inkjet(img, period=5):
    h, w = img.shape[:2]
    yy, xx = np.indices((h, w))
    screen = (np.sin(2 * np.pi * xx / period) * np.sin(2 * np.pi * yy / period) + 1) / 2
    out = img.astype(np.float32) * (0.72 + 0.45 * screen[..., None])
    return cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (0, 0), 0.9)


def lowres(img):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    _, enc = cv2.imencode(".jpg", up, [cv2.IMWRITE_JPEG_QUALITY, 25])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def print_watermark(img):
    """Counterfeit trick: print a faint portrait into the watermark window so it shows in normal light."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = C.ROIS["watermark"]; px0, py0, px1, py1 = C.ROIS["portrait"]
    X0, Y0, X1, Y1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    p = cv2.resize(img[int(py0 * h):int(py1 * h), int(px0 * w):int(px1 * w)], (X1 - X0, Y1 - Y0))
    out = img.copy()
    out[Y0:Y1, X0:X1] = cv2.addWeighted(out[Y0:Y1, X0:X1], 0.6, p, 0.4, 0)
    return out


def backlit(note_bgr, mode="genuine", seed=0):
    """DEMO ONLY: simulate a note held against a light.
    mode: 'genuine' (soft watermark) | 'none' (plain paper) | 'printed' (sharp printed portrait)."""
    rng = np.random.RandomState(seed)
    h, w = note_bgr.shape[:2]
    g = cv2.cvtColor(note_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    yy, xx = np.indices((h, w))
    light = 215 - 100 * ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2)
    img = light * (0.55 + 0.45 * g / 255.0) + rng.normal(0, 2.0, (h, w))
    x0, y0, x1, y1 = C.ROIS["watermark"]
    X0, Y0, X1, Y1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    if mode != "none":
        px0, py0, px1, py1 = C.ROIS["portrait"]
        p = g[int(py0 * h):int(py1 * h), int(px0 * w):int(px1 * w)]
        p = cv2.resize(p, (X1 - X0, Y1 - Y0))
        if mode == "genuine":
            p = cv2.GaussianBlur(p, (0, 0), (X1 - X0) / 9.0)
            amp = 0.07 * float(img.mean())
        else:
            amp = 0.12 * float(img.mean())
        p = (p - p.mean()) / (p.std() + 1e-6)
        img[Y0:Y1, X0:X1] += amp * p
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob("*"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        for tag, fn in (("photocopy", photocopy), ("inkjet", inkjet), ("lowres", lowres)):
            cv2.imwrite(str(dst / f"{p.stem}_{tag}.jpg"), fn(img))
    print("Done ->", dst)