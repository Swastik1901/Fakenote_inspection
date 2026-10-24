"""Central tunable settings. Calibrate these on YOUR dataset with evaluate.py."""

# Canonical size every note is warped to (width x height, landscape)
NOTE_W, NOTE_H = 1000, 440

# Layout below is for the Rs 500 Mahatma Gandhi New Series note; change ROIs for other notes.
# Regions of interest as fractions of the rectified note: (x0, y0, x1, y1).
# Layout differs per denomination/series, so adjust these for the note you use.
ROIS = {
    "thread":   (0.50, 0.00, 0.72, 1.00),   # band that should contain the security thread
    "portrait": (0.27, 0.15, 0.53, 0.95),   # Gandhi portrait
    "watermark":(0.77, 0.14, 0.88, 0.56),   # blank window on the right where the watermark portrait appears in transmitted light
    "serial":   (0.62, 0.78, 0.93, 0.94),   # bottom-right number panel
    "serial2":  (0.05, 0.26, 0.31, 0.37),   # top-left number panel (same number is printed twice)
}

# Weights used when fusing check scores (renormalised if a check is skipped)
WEIGHTS = {
    "Security thread":    1.5,
    "Print sharpness":    1.0,
    "Halftone / moire":   1.5,
    "Ink saturation":     1.0,
    "Intaglio texture":   1.0,
    "Reference match":    1.0,   # proves design conformity only - a photocopy matches too
    "Serial number":      1.0,
    "Watermark window":   1.0,   # normal light: window must be blank
    "Watermark (backlit)": 1.5,  # transmitted light: soft portrait must be visible
}

# Verdict thresholds on the fused score (0..1)
GENUINE_AT = 0.70
SUSPICIOUS_AT = 0.45

# Per-check calibration points: score goes 0 -> 1 between (low, high)
SHARPNESS_WIDTH = 500                # sharpness is measured at this fixed width -> independent of photo resolution
SHARPNESS_RANGE = (150.0, 700.0)     # Laplacian variance at SHARPNESS_WIDTH
HALFTONE_RANGE = (25.0, 100.0)       # FFT peak/median ratio; HIGH ratio = halftone dots -> score drops
SATURATION_RANGE = (15.0, 70.0)      # mean S channel (0..255)
LBP_ENTROPY_RANGE = (4.5, 6.8)       # bits (fallback only)
LBP_SIM_RANGE = (0.25, 0.55)         # histogram intersection vs genuine reference
THREAD_CONT_RANGE = (0.35, 0.85)     # continuity ratio
ORB_INLIER_RANGE = (0.08, 0.40)      # inlier / good-match ratio

# Watermark checks (assumed starting values - calibrate with watermark_probe.py on real backlit photos)
WM_EDGE_RANGE = (0.030, 0.080)       # edge density in the window under normal light (lower = blank = good)
WM_BAND_RANGE = (1.5, 4.0)           # % brightness variation at watermark scale (backlit)
WM_NCC_RANGE = (0.15, 0.55)          # correlation with a genuine backlit reference

# "Is this a banknote at all?" gate (profile of the Rs 500 Mahatma Gandhi series).
# It rejects only on POSITIVE evidence of a non-note, so grayscale photocopies still reach the checks.
GATE_HUE = (8, 75)            # OpenCV hue band used by the note (olive / yellow-green / orange)
GATE_OFF_PALETTE_MAX = 6.0    # % of pixels strongly coloured OUTSIDE that band
GATE_SATURATED_MAX = 5.0      # % of intensely saturated pixels (banknote inks are muted)
GATE_FLAT_MAX = 85.0          # % of 16x16 blocks that are flat/blank
GATE_ASPECT = (1.45, 3.6)     # allowed long/short side ratio when the image has no detectable outline