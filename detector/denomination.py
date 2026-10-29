"""Which banknote is this, and is it still valid?

Classical approach, three layers (first one that answers wins):
  1. text    : OCR the printed amount ("THE SUM OF FIVE HUNDRED RUPEES", big numerals). Needs Tesseract.
  2. design  : ORB feature match against the genuine reference images in references/<value>/ (no OCR needed).
  3. colour  : the note's dominant tint, only for denominations with a unique colour (Rs 100 lavender, Rs 2000 magenta).
Status facts (checked against RBI statements and press reports, Oct 2026) live in NOTES.
"""
import difflib
import hashlib
import re

import cv2

from . import config as C

_COMMON = "Legal tender and in circulation (Mahatma Gandhi New Series)."

NOTES = {
    10:   dict(label="₹10",   kind="info", title="₹10 note",   text=_COMMON),
    20:   dict(label="₹20",   kind="info", title="₹20 note",   text=_COMMON),
    50:   dict(label="₹50",   kind="info", title="₹50 note",   text=_COMMON),
    100:  dict(label="₹100",  kind="info", title="₹100 note",  text=_COMMON),
    200:  dict(label="₹200",  kind="info", title="₹200 note",  text=_COMMON),
    500:  dict(label="₹500",  kind="ok",   title="₹500 note",  text="Legal tender and in circulation (Mahatma Gandhi New Series, 2016 design)."),
    2000: dict(label="₹2000", kind="warn", badge="NOT IN USE ANYMORE", title="₹2000 note: not in use anymore (withdrawn from circulation)",
               text=("The RBI withdrew the ₹2000 note from circulation on 19 May 2023, so it is not in everyday use any more. It has "
                     "<b>not</b> been demonetised and is still legal tender (reconfirmed by the RBI in 2026). Banks no longer exchange it: "
                     "it can be deposited or exchanged only at the 19 RBI Issue Offices, or sent to them through India Post.")),
    1000: dict(label="₹1000", kind="bad", badge="NOT IN USE ANYMORE", title="₹1000 note: not in use anymore (demonetised)",
               text=("The ₹1000 note was demonetised on 8 November 2016 and is no longer legal tender. There is no ₹1000 note in "
                     "circulation today. Holding more than 10 pieces of the demonetised ₹500/₹1000 notes can attract a fine under the "
                     "Specified Bank Notes (Cessation of Liabilities) Act, 2017.")),
}
NOT_IN_USE = {1000, 2000}              # notes that must always carry the "not in use anymore" banner (their print analysis still runs)
OLD_500 = dict(kind="warn", title="₹500 note: could not confirm the current (2016) design",
               text=("The text reads ₹500 but the note did not match the current 2016 design. The <b>old ₹500 note (before 2016) is not in "
                     "use anymore: it was demonetised on 8 November 2016</b> and is not legal tender. If this is a current note, retake "
                     "the photo (flat, sharp, whole note in frame)."))
CALIBRATED = {500}                      # denominations with a calibrated authenticity profile in this build
STATUS_DATE = "Status as of October 2026; verify on rbi.org.in."

_WORDS = {("TEN",): 10, ("TWENTY",): 20, ("FIFTY",): 50, ("ONE", "HUNDRED"): 100, ("TWO", "HUNDRED"): 200,
          ("FIVE", "HUNDRED"): 500, ("ONE", "THOUSAND"): 1000, ("TWO", "THOUSAND"): 2000}
_CACHE = {}


def _sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def find_amount(text):
    """Return (value, confidence) for the best '<amount words> RUPEES' phrase in OCR text, tolerant to typos."""
    # OCR often swaps look-alike digits for letters (F1VE, TW0): undo that before matching
    toks = [t for t in re.sub(r"[^A-Z ]", " ", text.upper().translate(str.maketrans("0158", "OISB"))).split() if len(t) >= 3]
    best = (None, 0.0)
    for phrase, value in _WORDS.items():
        n = len(phrase)
        for i in range(len(toks) - n + 1):
            s = min(_sim(w, p) for w, p in zip(toks[i:i + n], phrase))
            if s < 0.8:
                continue
            if not any(_sim(t, "RUPEES") >= 0.7 for t in toks[i + n:i + n + 2]):     # amount words are always followed by RUPEES
                continue
            if s > best[1]:
                best = (value, s)
    return best


_NUM = re.compile(r"(?<!\d)(100|200|500|1000|2000)(?!\d)")


def find_numeral(text):
    """Big printed numerals (the ₹ amount) as a fallback when the small 'THE SUM OF ... RUPEES' text is unreadable.
    Accept only when exactly one denomination value appears, so serial-number digits cannot confuse it."""
    vals = {int(m) for m in _NUM.findall(text)}
    return (vals.pop(), 0.7) if len(vals) == 1 else (None, 0.0)


def read_amount(note_hr):
    """OCR the whole note (upright and upside-down) and read the printed amount. None if Tesseract is missing or unreadable."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        return None, 0.0
    key = hashlib.md5(note_hr.tobytes()).hexdigest()
    if key in _CACHE:
        return _CACHE[key]
    out = (None, 0.0)
    for note in (note_hr, cv2.rotate(note_hr, cv2.ROTATE_180)):          # the note may be uploaded upside-down
        h, w = note.shape[:2]
        region = cv2.resize(note[int(0.25 * h):int(0.70 * h), int(0.50 * w):int(0.85 * w)], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        whole = cv2.resize(note, (1800, int(1800 * h / w)), interpolation=cv2.INTER_CUBIC)
        best = max(cv2.split(whole), key=lambda c: c.std())             # highest-contrast channel: helps on magenta/lavender notes
        attempts = [(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), find_amount),      # promise-text area: fast
                    (cv2.cvtColor(whole, cv2.COLOR_BGR2GRAY), find_amount),       # whole note, words
                    (best, find_numeral)]                                          # whole note, big numerals
        for gray_img, finder in attempts:
            try:
                text = pytesseract.image_to_string(gray_img, config="--psm 11", timeout=8)
            except Exception:
                continue
            val, conf = finder(text)
            if val is not None and conf > out[1]:
                out = (val, conf)
            if out[1] >= 0.6:
                break
        if out[1] >= 0.6:
            break
    _CACHE[key] = out
    return out

# ------------------------------------------------------------------ layers 2 and 3: no OCR needed
# Dominant tint of the notes whose colour is unique among Indian notes (OpenCV hue 0..179). The warm notes (Rs 10, 20, 200,
# old 500/1000) overlap each other too much to be told apart by colour, so those are never guessed from colour alone
# (they are recognised by their printed text or by a reference image instead).
# Kept deliberately narrow: the validated Rs 500 (stone grey with green/teal ink) must never be mistaken for another note by colour.
_COLOUR_FAMILIES = ((100, 112, 143),     # lavender
                    (2000, 146, 178))    # magenta


def colour_guess(note_bgr):
    """(value, confidence) from the note's dominant tint, or (None, 0). Weak evidence: callers must say it is an estimate."""
    n = cv2.resize(note_bgr, (250, 110), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(n, cv2.COLOR_BGR2HSV)
    strong = (hsv[..., 1] >= 45) & (hsv[..., 2] >= 60)
    if strong.mean() < 0.12:                                     # grey / photocopy / blank: no tint to read
        return None, 0.0
    h = hsv[..., 0][strong]
    best = (None, 0.0)
    for value, lo, hi in _COLOUR_FAMILIES:
        share = float(((h >= lo) & (h <= hi)).mean())
        if share >= 0.60 and share > best[1]:
            best = (value, share)
    return (best[0], 0.5) if best[0] else (None, 0.0)


def design_guess(note, ref_sets):
    """(value, confidence) from ORB matches against the genuine references of every denomination, or (None, 0).
    ref_sets = {value: [reference, ...]}. Needs a clear winner so a shared motif (the Gandhi portrait) cannot decide it."""
    from .checks import best_inliers
    scores = {v: best_inliers(note, refs) for v, refs in (ref_sets or {}).items() if refs}
    if not scores:
        return None, 0.0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_v, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    if top >= 15 and top >= 2 * max(second, 1):
        return top_v, min(0.95, 0.4 + top / 200.0)
    return None, 0.0


def identify_note(note_hr, note, ref_sets=None):
    """Which denomination is this? Returns dict(value, conf, source) with source in {"text", "design", "colour", None}.
    Text (OCR) is the most reliable; design and colour keep the app useful when Tesseract is missing or cannot read the note."""
    val, conf = read_amount(note_hr)
    if val is not None:
        return dict(value=val, conf=conf, source="text")
    val, conf = design_guess(note, ref_sets)
    if val is not None:
        return dict(value=val, conf=conf, source="design")
    val, conf = colour_guess(note)
    if val is not None:
        return dict(value=val, conf=conf, source="colour")
    return dict(value=None, conf=0.0, source=None)


SOURCE_LABEL = {"text": "read from the printed amount", "design": "matched against a genuine reference of that denomination",
                "colour": "estimated from the note's colour only (install Tesseract to read the printed amount)"}