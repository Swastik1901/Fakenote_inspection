"""Robust image loading: never hands the app a None.

cv2.imdecode returns None for anything OpenCV was not built to read (a WebP/AVIF/HEIC saved with a .png
extension, a PNG with an odd colour profile, ...). Pillow reads many more of those, so it is the fallback.
"""
import io

import cv2
import numpy as np

try:                                    # optional: iPhone HEIC/HEIF photos
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass


def _to_bgr8(im):
    """Any decoded array (gray / BGR / BGRA, 8 or 16 bit) -> 3-channel uint8 BGR. Transparency is flattened onto white."""
    if im is None:
        return None
    if im.dtype != np.uint8:
        im = np.clip(im.astype(np.float32) / (257.0 if im.dtype == np.uint16 else 1.0), 0, 255).astype(np.uint8)
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 1:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:
        a = im[..., 3:4].astype(np.float32) / 255.0
        return (im[..., :3].astype(np.float32) * a + 255.0 * (1.0 - a)).astype(np.uint8)
    return im[..., :3]


def load_image_bgr(data):
    """bytes -> BGR uint8 array, or None when the bytes are not a readable image."""
    if not data:
        return None
    try:
        im = _to_bgr8(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED))
        if im is not None and im.size:
            return im
    except Exception:
        pass
    try:                                                            # Pillow fallback
        from PIL import Image, ImageOps
        pil = Image.open(io.BytesIO(data))
        pil = ImageOps.exif_transpose(pil)
        if pil.mode in ("RGBA", "LA", "P"):
            pil = pil.convert("RGBA")
            bg = Image.new("RGBA", pil.size, (255, 255, 255, 255))
            pil = Image.alpha_composite(bg, pil)
        rgb = np.asarray(pil.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.size else None
    except Exception:
        return None