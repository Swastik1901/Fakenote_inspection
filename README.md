# FakeNote Inspector

A Streamlit app that inspects a photo of an Indian banknote and explains, check by check, whether it looks genuine.
It uses classical computer vision (OpenCV, FFT, feature matching, OCR). There is no black-box model: every score comes with
the measurement and a picture of the evidence.

> **Screening aid only.** A phone camera cannot capture UV, microprinting or the feel of the paper. A note that looks
> suspicious should be verified by a bank.

---

## 1. Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Install Tesseract too (strongly recommended, see section 4 for macOS, Linux and Windows).
On Windows, if `pip` is not found, use `py -m pip install -r requirements.txt` and `py -m streamlit run app.py`.

The app opens at `http://localhost:8501`. Upload a photo, or use the webcam.

If you change files while the server is running, stop it (Ctrl+C) and start it again. After replacing anything in
`detector/`, delete `detector/__pycache__/` first.

---

## 2. Project layout

```
fakenote 2/
├── app.py                  Streamlit UI (upload, verdict, evidence, report)
├── requirements.txt
├── simulate_fake.py        Builds simulated fakes (photocopy, inkjet, low-res, printed watermark)
├── watermark_probe.py      Prints watermark metrics so you can calibrate WM_BAND_RANGE
├── evaluate.py             Calibrate thresholds on your own dataset
├── detector/
│   ├── __init__.py         Version stamp (app.py refuses to run with a mismatched detector/ folder)
│   ├── config.py           All tunable numbers: regions, weights, thresholds, per-denomination profiles
│   ├── imageio.py          Robust image loading (WebP/AVIF/GIF/ICO, transparency, 16-bit, corrupt files)
│   ├── preprocess.py       Finds the note in the photo and straightens it (perspective correction)
│   ├── gate.py             "Is this a banknote at all?" pre-check
│   ├── denomination.py     Which note is it, and is it still in use? (text, design, colour)
│   ├── checks.py           The individual checks (one function per security feature)
│   └── pipeline.py         Runs the checks, fuses the scores, produces the verdict
└── references/
    ├── *.jpg / *.png       Genuine ₹500 reference photos
    ├── 1000/               Genuine references for other denominations (folder name = value)
    ├── 100/  2000/ ...
    ├── sides/              Reverse sides, so a back-of-note upload is recognised as such
    └── watermark/          Optional backlit photos of genuine watermarks
```

---

## 3. How a photo is processed

1. **Load.** The file is decoded by OpenCV, with Pillow as a fallback. An unreadable file gets a clear message
   instead of a crash.
2. **Find and straighten the note.** Edges, contour, four-corner fit, perspective warp to a fixed 1000 x 440 canvas.
   Tick *Image is already a cropped note* if the photo contains only the note.
3. **Front or back?** Compared with the reverse-side references in `references/sides/`.
4. **Which note?** See section 5.
5. **Banknote gate.** Rejects obvious non-notes (photos, screenshots, blank images). It only runs when the denomination
   was not already proven by the printed text or a matching reference.
6. **Checks.** See the table below.
7. **Fusion and verdict.** See section 6.

### The checks

| Check | What it measures |
|---|---|
| Security thread | A continuous line or a column of aligned dashes where the windowed thread sits |
| Print sharpness | Laplacian variance at a fixed width (intaglio print is crisp, copies are soft) |
| Halftone / moire | FFT peaks from regular dot grids (inkjet and laser reprints) |
| Ink saturation | The note's own ink colour, not just overall colour (a grey photocopy has none) |
| Intaglio texture | Local Binary Pattern texture of the portrait compared with a genuine reference |
| Reference match | ORB features plus RANSAC against genuine reference photos. A design copy also passes this |
| Serial number | OCR of both number panels; they must read the same number |
| Issuer text | OCR for "RESERVE BANK OF INDIA", and for words no RBI note carries (see below) |
| Watermark window | The window must be blank in normal light (a printed fake portrait shows up here) |
| Watermark (backlit) | Optional second photo with the note held against a light |

A check that cannot decide (blur, glare, no reference) returns *skipped* and is not counted. An unreadable photo is
never treated as evidence of a fake.

---

## 4. Tesseract (OCR)

Tesseract is **recommended, not strictly required**. The app runs without it, but with reduced detection (see below).

`pytesseract` (installed by `pip`) is only the Python wrapper. The **Tesseract program itself** must be installed on the
computer, and it must be reachable from the terminal.

**macOS**

```bash
brew install tesseract
```

**Ubuntu / Debian**

```bash
sudo apt install tesseract-ocr
```

**Windows**

1. Install the Windows build (maintained by UB Mannheim). Either use the terminal:

   ```powershell
   winget install UB-Mannheim.TesseractOCR
   ```

   or download the installer from <https://github.com/UB-Mannheim/tesseract/wiki> and run it (keep the default
   English language data).
2. Add the install folder to your PATH. The default folder is `C:\Program Files\Tesseract-OCR`.
   Start menu, search "Edit the system environment variables", *Environment Variables*, select `Path`, *Edit*, *New*,
   paste the folder, OK. (Some installer versions offer to do this for you.)
3. **Close and reopen** the terminal (PowerShell or Command Prompt), then stop and restart `streamlit run app.py`.

**Check that it works (all systems):**

```bash
tesseract --version
```

If this prints a version, the app will find it. If Windows says `'tesseract' is not recognized`, the folder from step 2
is not on your PATH yet.

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

## 6. Verdict logic

- Each check gives a score from 0 to 1. Scores are fused with weights from `config.WEIGHTS`:
  60% weighted mean plus 40% mean of the two lowest scores, so one clearly failed check cannot be hidden by strong ones.
- Verdicts: **Genuine** (score at or above the *Genuine if* threshold), **Suspicious**, **Likely Fake**.
  *Likely Fake* needs at least 3 failed checks, so a bad light or glare alone cannot condemn a note.
- **Decisive evidence overrides the score.** If the Issuer text check reads words that no RBI note carries
  (for example "CHILDREN BANK", "FULL OF FUN", "MANORANJAN", "COUPON", "MOVIE", "PROP", "FAKE"), the verdict is
  **Likely Fake** and the score is capped at 15%. A red *Counterfeit evidence* card lists the words found.
  Other checks may stay green because a copied design passes design checks, so green bars do not mean genuine.
- Notes without a validated layout profile (everything except ₹500, and the demonetised ₹1000) can only be
  **No red flags** or **Suspicious**, never *Genuine*. For ₹1000, which has no layout profile, only the generic print
  checks and the issuer check run.

The *Report* tab downloads the result as JSON (denomination, how it was identified, status, verdict, every check).

---

## 7. Sidebar options

| Option | Effect |
|---|---|
| Source: Upload / Webcam | Photo input |
| Image is already a cropped note | Use when the photo contains only the note |
| Reject non-banknote images | The gate. Turn it off to analyse any image anyway |
| Analysis animation | Step-by-step scan before the verdict |
| Simulate an attack | Degrades the note as a photocopy, inkjet reprint, low-res scan, or printed fake watermark, to watch the verdict react |
| Watermark (hold note to light) | Upload a backlit photo, or use a demo simulation (not a real measurement) |
| Decision thresholds | Move the Genuine / Suspicious cut-offs |
| Enabled checks | Switch individual checks off |
| Genuine reference images | Add or remove ₹500 reference photos. Use a **different** photo from the one you test |

---

## 8. Tuning

Everything tunable is in `detector/config.py`:

- `WEIGHTS` is the weight of each check in the fused score.
- `GENUINE_AT`, `SUSPICIOUS_AT`, `MIN_FAILED_FOR_FAKE` are the verdict thresholds.
- `*_RANGE` values map a raw measurement to a 0 to 1 score. Calibrate them on your own photos with `evaluate.py`.
- `GATE_*` values tune the banknote gate.
- `PROFILES` holds per-denomination layouts (regions as fractions of the straightened note). Only the ₹500 is validated
  on real phone photos. The others were measured from official RBI specimen images.

Helper scripts:

```bash
python simulate_fake.py dataset/genuine dataset/fake      # build simulated fakes from genuine photos
python watermark_probe.py dataset/backlit_genuine dataset/backlit_plain_paper   # calibrate the watermark range
```

---

## 9. Taking photos that work well

- Flat note, plain contrasting background, the whole note in frame, even light, no glare, camera square-on.
- Sharp focus. A blurred photo lowers sharpness and also stops the OCR checks from reading.
- Upload the **front** (the side with the portrait). A back-side upload is recognised and refused politely.
- Use a different photo for the reference than for the test, otherwise the match is trivially perfect.

---

## 10. Troubleshooting

| Message or symptom | Fix |
|---|---|
| "Your `detector/` folder does not match this `app.py`" | Replace **every** file in `detector/` with the matching set, delete `detector/__pycache__/`, restart |
| "This file could not be read as an image" | The file is corrupt or a different format renamed to .png/.jpg (AVIF, HEIC). Re-export it as PNG or JPG. `pip install pillow-heif` helps with HEIC/AVIF |
| "This does not look like a banknote" | The photo shows several unrelated colours, is blank, or has the wrong shape. Retake it, or turn off *Reject non-banknote images* |
| "No banknote detected" | The edges were not found. Use a plain background, or tick *Image is already a cropped note* |
| Serial number or Issuer text always *skipped* | Tesseract is not installed or not on the PATH. Run `tesseract --version`. On Windows, add `C:\Program Files\Tesseract-OCR` to PATH and reopen the terminal and the app |
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
