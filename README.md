# 💵 FakeNote Inspector

**Fake Currency Detection using Classical Image Processing**

An explainable counterfeit-note screening tool built with OpenCV. Instead of a black-box
classifier, every security / print feature of the note is analysed by its own image-processing
check, scored, and shown visually, so you can see *why* a note was flagged.

> ⚠️ Screening aid only. Phone cameras cannot capture UV, microprinting, tactile or
> transmitted-light features. Suspicious notes should be verified by a bank.

---

## Folder structure

```
fakenote/
├── app.py                 # Streamlit dashboard (the demo)
├── evaluate.py            # Accuracy / precision / recall / confusion matrix + per-check calibration table
├── watermark_probe.py     # Prints watermark metrics for backlit photos (calibration)
├── simulate_fake.py       # Builds a "fake" dataset by photocopy / inkjet / low-res degradation
├── requirements.txt
├── README.md
│
├── detector/              # The detection engine (importable package)
│   ├── __init__.py
│   ├── config.py          # ROIs, thresholds, weights  <- the file you tune
│   ├── preprocess.py      # Note detection + perspective correction
│   ├── checks.py          # The 7 feature checks
│   └── pipeline.py        # Score fusion, verdict, annotated overlay
│
├── references/            # Put 1-3 clean images of GENUINE notes here (same denomination)
└── dataset/
    ├── genuine/           # genuine note photos (for evaluate.py)
    └── fake/              # fake / simulated-fake photos (for evaluate.py)
```

## Pipeline

```
Photo
  └─ Stage 1  Preprocess   Canny edges → contour → convex hull → 4-corner fit → perspective warp (1000×440)
       └─ Stage 2  Checks  9 independent feature checks, each → score 0..1 + evidence image
            └─ Stage 3  Fusion   60% weighted mean + 40% mean of the two weakest checks
                 └─ Verdict      GENUINE / SUSPICIOUS / LIKELY FAKE  (+ JSON report)
```

| # | Check | Technique | Why it works |
|---|-------|-----------|--------------|
| 1 | Security thread | Median-background vertical-line continuity | Genuine notes have an unbroken embedded thread |
| 2 | Print sharpness | Laplacian variance at a fixed 500 px width (independent of photo resolution) | Copies lose fine-line detail |
| 3 | Halftone / moire | 2-D FFT peak-to-median ratio | Inkjet/laser dot grids create sharp spectral peaks |
| 4 | Ink saturation | HSV saturation statistics | B&W copies / washed-out prints are desaturated |
| 5 | Intaglio texture | LBP histogram intersection vs reference | Raised-ink printing has a distinct micro-texture |
| 6 | Reference match | ORB features + RANSAC homography | Verifies design/layout conformity |
| 7 | Serial number | OCR of **both** number panels (the serial is printed twice), look-alike-tolerant comparison | Panels must read as the same valid `digit-letter-letter-6 digits` serial |

| 8 | Watermark window (normal light) | Edge density in the blank watermark window | Genuine window is blank in normal light; a printed portrait shows up |
| 9 | Watermark (backlit) | Band-pass tonal structure in the window on a photo held against light (+ optional correlation with a genuine watermark) | Plain paper or a surface print shows no structure inside the paper |

The watermark portrait only appears in **transmitted light**, so check 9 needs an optional second photo
(hold the note against a lamp/window). Checks 8 and 9 complement each other: 9 shows something is *in* the paper,
8 shows it is *not just printed on the surface*. Without a backlit photo, check 9 is skipped.
The watermark thresholds are assumed starting values validated on simulated backlit images only; calibrate them
with `python watermark_probe.py <folder of genuine backlit photos> <folder of plain-paper photos>`.
Optionally put a genuine backlit watermark photo in `references/watermark/` for a correlation check.

**Weakest-link fusion:** a plain average lets strong checks hide a failed one (e.g. a photocopy
matches the *design* perfectly). Blending in the two lowest scores means one clearly failed
check pulls the verdict down, which is the right bias for a security screen.

## Setup & run

```bash
pip install -r requirements.txt
# optional, for serial-number OCR:
#   Ubuntu: sudo apt install tesseract-ocr     macOS: brew install tesseract
#   Windows: install from https://github.com/UB-Mannheim/tesseract/wiki

# 1. put clean genuine-note images in references/
streamlit run app.py
```

In the sidebar, use **Live attack simulation** to degrade an uploaded genuine note into a
photocopy / inkjet reprint / low-res scan and watch the verdict flip in real time.

## Calibrate on your own data (important)

The thresholds in `detector/config.py` are starting points. Real notes differ by denomination,
camera and lighting.

```bash
# 1. add genuine photos to dataset/genuine/
# 2. generate simulated fakes from them
python simulate_fake.py dataset/genuine dataset/fake
# 3. evaluate
python evaluate.py --out results.csv
```

`evaluate.py` prints accuracy, precision, recall, the confusion matrix and a **per-check table
(genuine vs fake mean score)**. A large gap means the check is useful. Adjust the `*_RANGE`
values and `ROIS` in `config.py` until the gaps are large, then re-run.

ROIs (thread band, portrait, the two serial panels) are fractions of the rectified note. The defaults
are set for the **Rs 500 Mahatma Gandhi New Series**; adjust them in `config.py` for other notes.

For best results add a *different* clean genuine image to `references/` (never the image you test on).

## Classical vs Deep Learning

| | **This project (classical)** | **CNN approach** |
|---|---|---|
| Core idea | Hand-designed checks for each security feature | Network learns features from labelled images |
| Training data | None required (only a few references to calibrate) | Hundreds-thousands of genuine **and** fake images |
| Explainability | Per-feature score + visual evidence | Black box (needs Grad-CAM to approximate) |
| Compute | Runs in milliseconds on a CPU | Needs GPU for training |
| Generalisation | Limited by your hand-written rules; needs ROI tuning per note | Can generalise if data is diverse |
| Robustness to new fakes | Only catches what the rules cover | Can pick up subtle patterns, but fails on unseen fake types |
| Failure mode | Lighting/angle shifts break thresholds | Overfits to dataset artefacts (e.g. camera, background) |
| Best for | Teaching, transparency, low-resource devices | Maximum accuracy with a large dataset |

## Limitations

- Validated on synthetic notes and simulated fakes; real counterfeit data is scarce and
  illegal to produce. Accuracy on real notes depends on calibration.
- Defaults assume a generic layout; ROIs need tuning per denomination.
- Photos taken under poor lighting, heavy shadows or on cluttered backgrounds reduce accuracy.
- A reference match proves design conformity only; high-quality copies share the design.

## Future work

Per-denomination ROI profiles, watermark check on transmitted-light images, colour-shift ink
detection across two angles, and a hybrid mode adding a small CNN score into the fusion.
