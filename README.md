# FakeNote Inspector

**An explainable fake-currency screening tool for Indian banknotes, built with classical computer vision. No black box, no trained model.**

Upload (or photograph) a banknote and the app checks each security and print feature on its own, shows the evidence for every check, and gives a final verdict with a 0 to 100 authenticity score.

> **Screening aid only.** A phone camera cannot see UV features, microprinting or the feel of raised ink. Never use this tool as the only basis to accept or reject real money. For a final answer, check the note against the RBI's published security features or take it to a bank.

---

## 1. What does it do? (in simple words)

You give it a photo of a note. It then asks ten simple questions, the same way a person would examine a note:

| # | Check | The question it asks | How (simple) |
|---|-------|----------------------|--------------|
| 1 | **Security thread** | Is there a thread running top to bottom? | Looks for a line of dark dashes (or one continuous line) near the thread position, and counts the dashes |
| 2 | **Print sharpness** | Are the fine lines crisp, like real printing? | Laplacian variance (how much sharp edge detail there is). Photocopies are blurrier |
| 3 | **Halftone / moire** | Is the picture made of printer dots? | FFT (frequency analysis). A home printer's dot grid shows up as sharp spikes |
| 4 | **Ink saturation** | Does it have the real coloured ink? | HSV colour analysis. A black and white photocopy has almost no colour |
| 5 | **Intaglio texture** | Does the portrait have the raised-ink texture? | LBP (Local Binary Pattern) texture compared with a genuine reference |
| 6 | **Reference match** | Does the whole design match a genuine note? | ORB feature matching plus RANSAC geometry check |
| 7 | **Serial number** | Is the serial number printed twice, and do both copies agree? | OCR on both number panels |
| 8 | **Issuer text** | Does it say "RESERVE BANK OF INDIA"? Any toy words like "CHILDREN BANK"? | OCR on the whole note |
| 9 | **Watermark window** | Is the watermark window blank in normal light? | Edge density. A printed portrait there is a classic counterfeit trick |
| 10 | **Watermark (backlit)** | Does a soft portrait show when the note is held to light? | Optional second photo, band-pass filtering |

Each check gives a score from 0 to 1. The scores are combined into the final verdict.

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

## 6. Setup and run

```bash
# 1. Python packages
pip install -r requirements.txt

# 2. (Recommended) Tesseract OCR engine, needed for serial number, issuer text and amount reading
#    macOS:   brew install tesseract
#    Ubuntu:  sudo apt install tesseract-ocr
#    Windows: install from the UB Mannheim build and add it to PATH

# 3. Start the app
streamlit run app.py
```

Then open <http://localhost:8501>. Without Tesseract the app still works; the OCR-based checks are simply skipped and the denomination falls back to design match or colour.

**Tips for good photos:** whole note in frame, flat, plain background, even light, no glare, sharp focus.

If you see a "detector/ folder does not match this app.py" message, replace *every* file in `detector/`, delete `detector/__pycache__`, and restart.

---

## 7. Testing and calibration

```bash
# make fake images from genuine ones
python simulate_fake.py dataset/genuine dataset/fake

# accuracy, precision, recall, confusion matrix and the mean score of each check
python evaluate.py --out results.csv

# calibrate the backlit watermark range from real backlit photos
python watermark_probe.py dataset/backlit_genuine dataset/backlit_plain_paper
```

`evaluate.py` prints a table showing how far apart genuine and fake scores are for each check: a big gap means a useful check. Tune ranges and weights in `detector/config.py`.

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

## 11. Quick demo script (2 minutes)

1. Upload a genuine ₹500 photo: show the score gauge, the green annotated boxes, and open the *Evidence* tab for the thread and texture checks.
2. In *Simulate an attack*, pick **Photocopy**: colour disappears, so the verdict drops to **Likely Fake** because of decisive evidence.
3. Pick **Inkjet reprint**: the halftone check catches the printer dot grid.
4. Pick **Printed fake watermark**: the watermark window check fires.
5. Upload a ₹2000 specimen: show the "not in use anymore" status banner and the "denomination votes" line.
6. Open the *Pipeline* tab to show edges, outline and the straightened note, and the *Report* tab for the JSON export.

---

*Built for learning and demonstration of explainable image processing. Not affiliated with the Reserve Bank of India.*