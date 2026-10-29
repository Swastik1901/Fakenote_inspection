# FakeNote Inspector

**An explainable fake-currency screening tool for Indian banknotes, built with classical computer vision. No black box, no trained model.**

Upload (or photograph) a banknote and the app checks each security and print feature on its own, shows the evidence for every check, and gives a final verdict with a 0 to 100 authenticity score.

> **Screening aid only.** A phone camera cannot see UV features, microprinting or the feel of raised ink. Never use this tool as the only basis to accept or reject real money. For a final answer, check the note against the RBI's published security features or take it to a bank.

---

## 1. Quick start

```bash
pip install -r requirements.txt
brew install tesseract          # macOS. Strongly recommended (see section 4)
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload a photo, or use the webcam.

If you change files while the server is running, stop it (Ctrl+C) and start it again. After replacing anything in
`detector/`, delete `detector/__pycache__/` first.

---

## 2. Features

### Analysis
- **10 independent checks** (above), each with a score, the measured value, a plain-language explanation and a debug image as proof.
- **Automatic note detection and straightening.** Finds the note in the photo (edges, contour, 4 corners) and flattens the perspective. If the image is already a cropped note, this is detected automatically.
- **Denomination detection** (₹100, ₹500, ₹1000, ₹2000, and others for the status banner) using a *voting* system:
  - printed amount words read by OCR (strongest),
  - design match against reference images (ORB),
  - dominant colour (only for notes with a unique tint: lavender ₹100, magenta ₹2000),
  - big numerals read by OCR (weakest, because OCR often drops digits).

  Votes add up. A design match is ignored if the note's colour clearly contradicts it. The app shows the votes, so a wrong call can be explained.
- **Status banner.** Tells you if a note is demonetised or withdrawn (₹1000 demonetised on 8 Nov 2016; ₹2000 withdrawn from circulation on 19 May 2023 but still legal tender; old pre-2016 ₹500 not valid).
- **"Is this even a banknote?" gate.** Rejects screenshots, photos, blank images and wrong shapes, but never rejects a black and white photocopy (that should be flagged as fake instead).
- **Front/back detection** using reverse-side reference images in `references/sides/` (optional).

### Smart decision making
- **Weakest-link fusion.** Final score = 60% weighted average + 40% average of the two lowest checks. One badly failed check cannot hide behind many good ones.
- **Decisive evidence.** Some findings are proof by themselves and override the average:
  - a printer's dot grid (halftone peak ratio of 250 or more),
  - no colour at all (a black and white copy),
  - a portrait printed in the watermark window (validated layouts only),
  - toy/prop words found by OCR ("CHILDREN BANK", "REPLICA", "FAKE" ...).
- **Consensus rule.** "Likely Fake" needs at least 3 failed checks, unless there is decisive evidence. One or two odd checks (glare, bad light) only give "Suspicious".
- **Honest limits.** Checks that cannot read something (blurry serial number, unreadable text) are marked *inconclusive* and are **not counted** against the note.
- **Capped verdicts for unvalidated denominations.** Layouts that were only measured from an official specimen can say "No red flags" or "Suspicious", never "Genuine" or "Likely Fake".

### Demo and testing tools
- **Attack simulator** in the sidebar: photocopy, inkjet reprint, low-res scan, printed fake watermark. Watch the verdict react live.
- **Backlit watermark simulator** (demo only) and a real "upload backlit photo" option.
- **Adjustable thresholds and weights**, and a switch for each check.
- **Add your own genuine reference images** from the sidebar.
- **Evidence tab** (what was measured, with debug images), **Pipeline tab** (edges, outline, straightened note) and **Report tab** (JSON download).
- **Step-by-step analysis animation** showing each check ticking off.
- **Webcam input** as well as file upload. Reads many formats including WebP, BMP, TIFF and iPhone HEIC.
- **Dark, clean UI** with a score gauge, per-check bars, and an annotated note (green boxes passed, red boxes failed).

---

## 3. Technology

| Part | Tool |
|------|------|
| Language | Python 3 |
| Image processing | **OpenCV** (`opencv-python`) |
| Numbers and maths | **NumPy** |
| Web app / UI | **Streamlit** |
| Image loading fallback | **Pillow**, plus optional **pillow-heif** for HEIC/AVIF |
| Text reading (OCR) | **Tesseract** via **pytesseract** (optional, but needed for serial number, issuer text and amount words) |

### Computer vision techniques used (and where)
- **Canny edges, contours, `approxPolyDP`, perspective warp**: finding and straightening the note.
- **Morphology (black-hat / top-hat, opening, closing) and connected components**: finding thread dashes; **line fitting** to count dashes that sit on dark artwork.
- **Laplacian variance**: print sharpness at a fixed width (so photo resolution does not matter).
- **FFT with a Hann window**: detecting halftone dot grids.
- **HSV colour space**: ink colour and colour-based denomination guess.
- **Local Binary Patterns (LBP) + histogram intersection**: intaglio micro-texture (compared at a fixed size so scan resolution does not matter).
- **ORB features + Brute-Force matcher + ratio test + RANSAC homography**: design matching.
- **Tesseract OCR with fuzzy matching** (`difflib`): serial numbers, issuer text, amount words; look-alike character correction (0/O, 1/I, 5/S ...).
- **Gaussian band-pass and template matching (NCC)**: watermark in transmitted light.

No machine learning model is trained. Every number comes from a formula you can explain.

---

## 4. How it works (pipeline)

```
Photo
  -> load (any format)
  -> find note outline (edges -> contour -> 4 corners)
  -> straighten to 1000 x 440 px
  -> read denomination (OCR words + design match + colour + numerals, voting)
  -> "is it a banknote?" gate + front/back check
  -> pick the layout profile for that denomination
  -> run the 10 checks
  -> weakest-link fusion + decisive-evidence override + consensus rule
  -> verdict + score + annotated image + evidence
```

---

## 5. Project structure

```
app.py                  Streamlit user interface
detector/
  __init__.py           version stamp (app refuses to run if folders are mismatched)
  config.py             all tunable settings + per-denomination layout profiles
  preprocess.py         find the note and straighten it
  gate.py               "is this a banknote at all?" test
  checks.py             the 10 checks
  denomination.py       which note is it? voting + status banners
  pipeline.py           runs everything, fuses scores, draws annotations
  imageio.py            robust image loading
references/             genuine reference images (references/<value>/ per denomination)
simulate_fake.py        makes fake test images (photocopy, inkjet, low-res, printed watermark)
evaluate.py             accuracy / confusion matrix on dataset/genuine and dataset/fake
watermark_probe.py      helps calibrate the backlit watermark range
requirements.txt
```

---

## 4. Tesseract (OCR)

`pytesseract` is only the Python wrapper. The **Tesseract program** must be installed on the computer:

```bash
brew install tesseract          # macOS
sudo apt install tesseract-ocr  # Debian / Ubuntu
tesseract --version             # confirm it works
```

Without Tesseract the app still runs, but:

- the printed amount cannot be read (the denomination then comes from references or colour, see section 5),
- the **Serial number** and **Issuer text** checks are skipped. A toy or prop note that copies a real design
  is only caught by the Issuer text check, so it can go undetected.

---

## 5. Which note is it? (denomination)

Three layers, the first that answers wins. The source is shown under the banner whenever it is not the printed text.

1. **Text.** OCR of the printed amount ("THE SUM OF FIVE HUNDRED RUPEES", or the big numerals). Most reliable. Needs Tesseract.
2. **Design.** ORB match against genuine reference images in `references/<value>/`. Works without Tesseract.
   Add a sharp, flat, full-note photo of any denomination to its folder, for example `references/2000/front.jpg`.
3. **Colour.** Only for notes with a unique tint: lavender (₹100) and magenta (₹2000). Warm notes overlap too much,
   so they are never guessed from colour. This is the weakest layer and is labelled as an estimate.

### Status banners

| Note | Banner |
|---|---|
| ₹1000 | **NOT IN USE ANYMORE.** Demonetised on 8 Nov 2016 |
| ₹2000 | **NOT IN USE ANYMORE.** Withdrawn from circulation on 19 May 2023, but not demonetised and still legal tender |
| ₹500 not matching the 2016 design | Warning that the old ₹500 (before 2016) is demonetised. Not certain, so there is no hard badge |
| ₹10, 20, 50, 100, 200, 500 | Legal tender and in circulation |

The analysis **always runs**, also for notes that are not in use. It only judges how the note was printed and does not change
the note's status. Status text lives in `NOTES` in `detector/denomination.py` and is dated October 2026, so
verify it on rbi.org.in.

---

## 8. Reference images

- `references/500_note.jpg` and `references/<value>/` hold genuine photos used for texture and design matching.
- Use a **different** photo for the reference than the one you test, otherwise the match is trivially perfect.
- Adding a good reference for a denomination (for example `references/2000/`) makes its identification and design check stronger.

---

## 9. Current limitations (be honest in the demo)

- Only the **₹500 (Mahatma Gandhi New Series)** layout is validated on real phone photos. ₹100, ₹1000 (old) and ₹2000 layouts were measured from specimen images, so their verdict stays at "No red flags" or "Suspicious".
- A good **design copy** (or a toy note with a copied design) can pass the design checks. That is why process checks (halftone, colour, printed watermark) and OCR text checks can override the average.
- The main fake data is **simulated** (photocopy, inkjet, low-res). Real counterfeits can behave differently.
- Phone cameras cannot capture UV, microprinting, tactile or colour-shifting features.
- OCR depends on photo quality; unreadable text is treated as inconclusive, not as proof of a fake.
- Layout profiles use module-level state, which is fine for one user at a time (a local demo), not for a multi-user server.
- Note status information is as of October 2026; verify on rbi.org.in.

---

## 10. Possible future work

- Validate the ₹100, ₹200, ₹2000 and old ₹1000 profiles on real photos and unlock full verdicts.
- Add ₹10, ₹20, ₹50, ₹200 layout profiles.
- Collect a real backlit watermark dataset and calibrate the watermark check.
- Add a small learned classifier on top of the explainable features and compare it with the rule-based fusion.
- Mobile-friendly capture guide (edge overlay, blur and glare warnings).

---

## 10. Troubleshooting

| Message or symptom | Fix |
|---|---|
| "Your `detector/` folder does not match this `app.py`" | Replace **every** file in `detector/` with the matching set, delete `detector/__pycache__/`, restart |
| "This file could not be read as an image" | The file is corrupt or a different format renamed to .png/.jpg (AVIF, HEIC). Re-export it as PNG or JPG. `pip install pillow-heif` helps with HEIC/AVIF |
| "This does not look like a banknote" | The photo shows several unrelated colours, is blank, or has the wrong shape. Retake it, or turn off *Reject non-banknote images* |
| "No banknote detected" | The edges were not found. Use a plain background, or tick *Image is already a cropped note* |
| Serial number or Issuer text always *skipped* | Tesseract is not installed or not on the PATH (`tesseract --version`) |
| Denomination shown as "estimated from colour" | Tesseract could not read the amount. Install it, or add a reference to `references/<value>/` |

---

## 11. Limitations

- Classical image checks cannot match a physical inspection. A high-quality copy that reproduces the printed words and
  the design well can pass the image checks.
- The Issuer text check is positive evidence only. It can prove a note is a toy or prop, but cannot prove a note is genuine.
- Only the ₹500 layout is validated on real photos. All other denominations are limited to *No red flags* / *Suspicious*.
- Real counterfeits cannot be obtained legally, so fake-detection quality was checked with simulated attacks and
  toy/prop notes, not with real counterfeits.
- Currencies other than the Indian rupee are not supported in this version.