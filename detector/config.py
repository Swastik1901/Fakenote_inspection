"""Central tunable settings. Calibrate these on YOUR dataset with evaluate.py."""

# Canonical size every note is warped to (width x height, landscape)
NOTE_W, NOTE_H = 1000, 440

# Layout below is for the Rs 500 Mahatma Gandhi New Series note; change ROIs for other notes.
# Regions of interest as fractions of the rectified note: (x0, y0, x1, y1).
# Layout differs per denomination/series, so adjust these for the note you use.
ROIS = {
    "thread":   (0.51, 0.00, 0.60, 1.00),   # narrow band where the thread sits (a wider band lets text count as dashes)
    "portrait": (0.27, 0.15, 0.53, 0.95),   # Gandhi portrait
    "watermark":(0.77, 0.14, 0.88, 0.56),   # blank window on the right where the watermark portrait appears in transmitted light
    "serial":   (0.60, 0.76, 0.99, 0.98),   # bottom-right number panel (generous: layout shifts a few % between photos)
    "serial2":  (0.00, 0.15, 0.36, 0.38),   # top-left number panel (same number is printed twice)
}

# Weights used when fusing check scores (renormalised if a check is skipped)
WEIGHTS = {
    "Security thread":    1.0,   # weak specificity: any strong vertical line also scores well
    "Print sharpness":    1.0,
    "Halftone / moire":   1.5,
    "Ink saturation":     1.0,
    "Intaglio texture":   1.0,
    "Reference match":    1.0,   # proves design conformity only - a photocopy matches too
    "Serial number":      1.0,
    "Issuer text":        1.5,   # OCR of "RESERVE BANK OF INDIA"; toy/prop words are decisive on their own
    "Watermark window":   1.0,   # normal light: window must be blank
    "Watermark (backlit)": 1.5,  # transmitted light: soft portrait must be visible
}

# Verdict thresholds on the fused score (0..1)
GENUINE_AT = 0.80
SUSPICIOUS_AT = 0.45

# Per-check calibration points: score goes 0 -> 1 between (low, high)
SHARPNESS_WIDTH = 500                # sharpness is measured at this fixed width -> independent of photo resolution
SHARPNESS_RANGE = (150.0, 700.0)     # Laplacian variance at SHARPNESS_WIDTH
HALFTONE_RANGE = (25.0, 100.0)       # FFT peak/median ratio; HIGH ratio = halftone dots -> score drops
SATURATION_RANGE = (15.0, 70.0)      # mean S channel (0..255); only a fallback for other note colours
INK_HUE = (25, 100)                  # OpenCV hue band of the note's green/olive ink
INK_RANGE = (1.0, 8.0)               # % of pixels carrying that ink (a B/W copy has ~0, even with a paper tint)
# Copy signals so strong that one of them is enough for "Likely Fake" (a design copy passes the design checks, so a copy usually fails
# only ONE process check). Calibrated on the simulated attacks in simulate_fake.py: genuine photos stay far below these values.
NO_COLOUR_S = 6.0                    # mean saturation (0..255) below this = no colour at all; every Indian note is printed in colour
HALFTONE_DECISIVE = 250.0            # FFT peak/median; genuine ~5-40, a printer's dot screen gives hundreds to thousands
WM_EDGE_DECISIVE = 0.10              # edge density in the blank watermark window (validated layouts only); genuine ~2-5%, a printed portrait ~12%
PROFILE_VALIDATED = True             # set by set_profile(): layout-based decisive checks only fire on a validated layout
MIN_FAILED_FOR_FAKE = 3              # 'Likely Fake' needs at least this many failed checks, otherwise 'Suspicious'
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
GATE_OFF_PALETTE_MAX = 6.0    # % of pixels strongly coloured OUTSIDE that band (Rs 500 palette)...
GATE_HUE_FAMILY = 40          # ...but a note is still accepted when its colours sit in ONE hue family this wide (OpenCV units, 40 = 80 deg):
GATE_SCATTERED_MAX = 12.0     # reject only if more than this % of the whole image is coloured outside the main tint (other denominations, e.g. lavender Rs 100)
GATE_SATURATED_MAX = 5.0      # % of intensely saturated pixels (banknote inks are muted)
GATE_FLAT_MAX = 85.0          # % of 16x16 blocks that are flat/blank
GATE_ASPECT = (1.45, 3.6)     # allowed long/short side ratio when the image has no detectable outline

# Thread: background is sampled this many columns to each side of a column (must exceed half a dash's width)
THREAD_SIDE_OFFSETS = (22, 30, 38)


# ---- Per-denomination layout profiles (fractions of the straightened note). Only the Rs 500 has been validated on a real phone
# photo; the others were measured from the official RBI specimen images, so their verdict is capped at "No red flags".
PROFILES = {
    500:  dict(validated=True,  ink_hue=(25, 100),
               rois=dict(thread=(0.51, 0.0, 0.60, 1.0), portrait=(0.27, 0.15, 0.53, 0.95), watermark=(0.77, 0.14, 0.88, 0.56),
                         serial=(0.60, 0.76, 0.99, 0.98), serial2=(0.0, 0.15, 0.36, 0.38))),
    2000: dict(validated=False, ink_hue=(135, 179),
               rois=dict(thread=(0.55, 0.0, 0.65, 1.0), portrait=(0.32, 0.17, 0.55, 0.95), watermark=(0.76, 0.20, 0.88, 0.62),
                         serial=(0.60, 0.78, 0.99, 0.98), serial2=(0.0, 0.16, 0.36, 0.38))),
    1000: dict(validated=False, ink_hue=(25, 100),     # old (pre-2016) Rs 1000: portrait right, blank window LEFT, thread at ~56%
               rois=dict(thread=(0.535, 0.0, 0.595, 1.0), portrait=(0.68, 0.20, 0.84, 0.95), watermark=(0.12, 0.22, 0.27, 0.62),
                         serial=(0.72, 0.03, 0.99, 0.20), serial2=(0.03, 0.82, 0.34, 0.99))),
    100:  dict(validated=False, ink_hue=(110, 140),
               rois=dict(thread=(0.51, 0.0, 0.60, 1.0), portrait=(0.28, 0.20, 0.52, 0.95), watermark=(0.74, 0.12, 0.88, 0.58),
                         serial=(0.60, 0.78, 0.97, 0.98), serial2=(0.0, 0.16, 0.36, 0.38))),
}


def set_profile(value):
    """Activate the layout profile of a denomination (module-level state: fine for one user at a time)."""
    global INK_HUE, PROFILE_VALIDATED
    p = PROFILES.get(value, PROFILES[500])
    PROFILE_VALIDATED = bool(p["validated"])
    ROIS.clear()
    ROIS.update(p["rois"])
    INK_HUE = p["ink_hue"]
    return p