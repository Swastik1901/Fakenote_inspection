import hashlib
import html
import json
import time

import cv2
import numpy as np
import streamlit as st

try:
    from detector import config as C
    from detector.checks import best_inliers
    from detector.denomination import NOTES, NOT_IN_USE, OLD_500, STATUS_DATE, SOURCE_LABEL, identify_note
    from detector.pipeline import (analyze, get_all_references, get_reference_sets, get_references, get_side_references, REF_DIR)
    from detector.gate import check_banknote
    from detector.imageio import load_image_bgr
    from detector.preprocess import rectify
    from simulate_fake import photocopy, inkjet, lowres, print_watermark, backlit
    import detector as _detector
    if getattr(_detector, "__version__", None) != "2026-10-28d":
        raise ImportError(f"detector/__init__.py is version {getattr(_detector, '__version__', 'unknown')}, expected 2026-10-28d")
    if "note_hr" not in rectify(np.zeros((40, 80, 3), np.uint8)):       # catches a stale preprocess.py
        raise ImportError("detector/preprocess.py is out of date (no 'note_hr')")
except ImportError as _e:      # app.py and the detector/ folder come from different versions
    st.set_page_config(page_title="FakeNote Inspector", page_icon="💵")
    st.error("**Your `detector/` folder does not match this `app.py`** (files from different versions).\n\n"
             "Replace **every** file in `detector/` (`__init__`, `checks`, `config`, `denomination`, `gate`, `imageio`, `pipeline`, `preprocess`), "
             "plus `app.py`, `simulate_fake.py` and the `references/` folder, with the ones from the latest `detector.zip`, "
             "delete `detector/__pycache__`, then restart the server.\n\n"
             f"Details: `{_e}`")
    st.stop()

st.set_page_config(page_title="FakeNote Inspector", page_icon="💵", layout="wide")

GOLD, GREEN, AMBER, RED, MUTED = "#c9a96a", "#5fb88a", "#d9a441", "#d4605a", "#8b909c"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

/* chrome */
[data-testid="stHeader"] {{background: transparent;}}
footer, #MainMenu {{visibility: hidden;}}
.block-container {{padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px;}}

/* fonts: only our own blocks use Inter; never touch Streamlit's icon font */
.eyebrow, .tagline, .vcard, .clist, .step, .note {{font-family: 'Inter', system-ui, sans-serif;}}
[data-testid="stIconMaterial"], [data-testid="stSidebarCollapseButton"] span, [data-testid="stExpanderToggleIcon"] {{
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
    font-weight: normal; letter-spacing: normal; text-transform: none;
}}

/* wordmark */
.eyebrow {{letter-spacing: .28em; font-size: .72rem; color: {GOLD}; font-weight: 600; text-transform: uppercase;}}
.title {{font-family: 'Fraunces', Georgia, serif; font-size: 2.6rem; line-height: 1.1; margin: .25rem 0 .4rem; color: #f3f1ea; font-weight: 500;}}
.tagline {{color: {MUTED}; font-size: .98rem; max-width: 640px; margin-bottom: 1.6rem;}}
.rule {{height: 1px; background: linear-gradient(90deg, {GOLD}55, transparent); margin: 0 0 1.6rem;}}

/* verdict card */
.vcard {{display: flex; gap: 1.4rem; align-items: center; padding: 1.4rem 1.5rem; border-radius: 18px;
        background: #10131a; border: 1px solid rgba(255,255,255,.07);}}
.vcard .lbl {{font-size: .7rem; letter-spacing: .22em; color: {MUTED}; text-transform: uppercase;}}
.vcard .word {{font-family: 'Fraunces', Georgia, serif; font-size: 2rem; line-height: 1.1; margin: .2rem 0 .35rem;}}
.vcard .sub {{color: {MUTED}; font-size: .9rem;}}

/* check list */
.clist {{margin-top: 1rem; padding: .5rem 1.2rem; border-radius: 18px; background: #10131a; border: 1px solid rgba(255,255,255,.07);}}
.crow {{display: grid; grid-template-columns: 1fr 110px 44px; gap: .9rem; align-items: center; padding: .62rem 0;
       border-bottom: 1px solid rgba(255,255,255,.05); font-size: .9rem;}}
.crow:last-child {{border-bottom: none;}}
.crow .nm {{color: #e8e6e1;}} .crow .mt {{display: block; color: {MUTED}; font-size: .74rem; margin-top: 1px;}}
.bar {{height: 5px; border-radius: 99px; background: rgba(255,255,255,.08); overflow: hidden;}}
.bar > div {{height: 100%; border-radius: 99px;}}
.crow .pc {{text-align: right; color: {MUTED}; font-variant-numeric: tabular-nums; font-size: .85rem;}}

/* images + empty state */
[data-testid="stImage"] img {{border-radius: 14px; border: 1px solid rgba(255,255,255,.08);}}
.steps {{display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: .4rem;}}
.step {{padding: 1.2rem 1.3rem; border-radius: 16px; background: #10131a; border: 1px solid rgba(255,255,255,.07);}}
.step .n {{font-family: 'Fraunces', serif; color: {GOLD}; font-size: 1.5rem;}}
.step .t {{font-weight: 600; margin: .2rem 0 .2rem;}} .step .d {{color: {MUTED}; font-size: .86rem;}}
.note {{padding: 1rem 1.2rem; border-radius: 14px; background: #161a22; border: 1px solid {AMBER}55; color: #e9dcc0; font-size: .92rem;}}

/* analysis animation */
.scancard {{padding: 1.4rem 1.6rem; border-radius: 18px; background: #10131a; border: 1px solid rgba(255,255,255,.07); font-family: 'Inter', system-ui, sans-serif;}}
.slbl {{font-size: .7rem; letter-spacing: .24em; color: {GOLD}; font-weight: 600;}}
.sst {{font-family: 'Fraunces', Georgia, serif; font-size: 1.5rem; margin: .3rem 0 .9rem; color: #f3f1ea;}}
.pbar {{height: 4px; border-radius: 99px; background: rgba(255,255,255,.08); overflow: hidden;}}
.pbar > div {{height: 100%; background: linear-gradient(90deg, {GOLD}, #e8d29a); border-radius: 99px; transition: width .3s;}}
.slist {{margin-top: 1rem; display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: .35rem 1.4rem;}}
.srow {{display: flex; gap: .6rem; font-size: .88rem; color: #cfcdc6;}}
.srow .snm {{flex: 1;}} .srow .spc {{color: {MUTED}; font-variant-numeric: tabular-nums;}}

.note.ok {{border-color: {GREEN}66;}} .note.bad {{border-color: {RED}99; background: #1c1517;}} .note.info {{border-color: rgba(255,255,255,.18);}}
.note b {{color: #f3f1ea;}} .note ul {{color: #e9dcc0;}}
.badge {{display: inline-block; padding: .18rem .7rem; margin-bottom: .55rem; border-radius: 99px; font-size: .72rem; font-weight: 700;
        letter-spacing: .16em; color: #fff; background: {RED}; font-family: 'Inter', system-ui, sans-serif;}}
.badge.warn {{background: {AMBER}; color: #20170a;}}

/* sidebar */
[data-testid="stSidebar"] {{border-right: 1px solid rgba(255,255,255,.06);}}
[data-testid="stSidebar"] h3 {{font-size: .72rem; letter-spacing: .2em; color: {GOLD}; text-transform: uppercase; margin-top: 1rem;}}
.stTabs [data-baseweb="tab"] {{font-size: .92rem;}}
details {{border-radius: 12px !important; border-color: rgba(255,255,255,.07) !important;}}
</style>
""", unsafe_allow_html=True)


def show(a, **kw):
    """st.image that works across Streamlit versions."""
    try:
        st.image(a, width="stretch", **kw)
    except Exception:
        st.image(a, use_container_width=True, **kw)


rgb = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2RGB) if a.ndim == 3 else a
tone = lambda s: GREEN if s >= 0.65 else (AMBER if s >= 0.4 else RED)

def banner_html(b, extra=""):
    badge = f'<span class="badge {b["kind"]}">{b["badge"]}</span><br>' if b.get("badge") else ""
    return (f'<div class="note {b["kind"]}">{badge}<b>{b["title"]}</b><br>{b["text"]}{extra}'
            f'<br><span style="opacity:.6;font-size:.78rem">{STATUS_DATE}</span></div>')


def scan_html(pct, status, done):
    rows = ""
    for name, score in done:
        if score is None:
            mark, col, txt = "–", MUTED, "skipped"
        else:
            mark, col, txt = ("✓" if score >= 0.5 else "✗"), (GREEN if score >= 0.5 else RED), f"{score*100:.0f}%"
        rows += f'<div class="srow"><span style="color:{col}">{mark}</span><span class="snm">{name}</span><span class="spc">{txt}</span></div>'
    return (f'<div class="scancard"><div class="slbl">ANALYSING</div><div class="sst">{status}</div>'
            f'<div class="pbar"><div style="width:{pct}%"></div></div><div class="slist">{rows}</div></div>')


def play_scan(results):
    """Staged reveal: each check ticks off with its real score before the verdict appears."""
    ph = st.empty()
    if results is None:                                  # nothing to inspect yet - just look for a note
        ph.markdown(scan_html(40, "Looking for a banknote…", []), unsafe_allow_html=True)
        time.sleep(1.0)
        ph.empty()
        return
    steps = ["Locating the banknote", "Straightening perspective"]
    total, n, done = len(steps) + len(results), 0, []
    for s in steps:
        ph.markdown(scan_html(int(n / total * 100), s + "…", done), unsafe_allow_html=True)
        time.sleep(0.45)
        n += 1
    for r in results:
        ph.markdown(scan_html(int(n / total * 100), f"Inspecting {r.name.lower()}…", done), unsafe_allow_html=True)
        time.sleep(0.12 if r.score is None else 0.35)
        done.append((r.name, r.score))
        n += 1
    ph.markdown(scan_html(100, "Weighing the evidence…", done), unsafe_allow_html=True)
    time.sleep(0.7)
    ph.empty()


# ------------------------------------------------------------------ header
st.markdown(
    '<div class="eyebrow">Currency forensics</div>'
    '<div class="title">FakeNote Inspector</div>'
    '<div class="tagline">Every security feature of the note is checked on its own and shown with evidence. '
    'No black box, just explainable computer vision.</div><div class="rule"></div>',
    unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Input")
    mode = st.radio("Source", ["Upload", "Webcam"], horizontal=True, label_visibility="collapsed")
    file = st.file_uploader("Note photo", type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"], label_visibility="collapsed") \
        if mode == "Upload" else st.camera_input("Take a photo", label_visibility="collapsed")
    cropped = st.toggle("Image is already a cropped note", value=False,
                        help="Skip note detection when the photo contains only the note.")

    gate_on = st.toggle("Reject non-banknote images", value=True,
                        help="Stops obvious non-notes (photos, screenshots, blank images) before analysis.")
    animate = st.toggle("Analysis animation", value=True, help="Show the step-by-step scan before the verdict.")

    st.markdown("### Simulate an attack")
    attack = st.selectbox("Degrade as a", ["None (original)", "Photocopy", "Inkjet reprint", "Low-res scan", "Printed fake watermark"],
                          label_visibility="collapsed")

    with st.expander("Watermark (hold note to light)"):
        st.caption("Optional. Hold the note against a lamp or window so light passes through, then photograph it. "
                   "The watermark portrait only shows in transmitted light.")
        wm_file = st.file_uploader("Backlit photo", type=["jpg", "jpeg", "png"], key="wm", label_visibility="collapsed")
        wm_sim = st.selectbox("Or simulate (demo only, not a real measurement)",
                              ["Off", "Genuine watermark", "No watermark"])

    with st.expander("Decision thresholds"):
        g_at = st.slider("Genuine if score ≥", 0.30, 0.95, C.GENUINE_AT, 0.01)
        s_at = st.slider("Suspicious if score ≥", 0.10, g_at, min(C.SUSPICIOUS_AT, g_at), 0.01)
    with st.expander("Enabled checks"):
        weights = {k: (1.0 if st.checkbox(k, True) else 0.0) * v for k, v in C.WEIGHTS.items()}
    with st.expander("Genuine reference images"):
        st.caption("Add a sharp, flat, full-note photo of a genuine ₹500. Use a **different** photo from the one you test, "
                   "otherwise the match is trivially perfect.")
        new_refs = st.file_uploader("Add reference", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                                    key="refup", label_visibility="collapsed")
        for f in new_refs or []:
            REF_DIR.mkdir(exist_ok=True)
            (REF_DIR / f.name).write_bytes(f.getvalue())
            if file is not None and f.getvalue() == file.getvalue():
                st.warning(f"{f.name} is the same photo you are testing. Use a different one.")
        if st.button("Remove all references"):
            for p in REF_DIR.glob("*"):
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    p.unlink()
            st.rerun()
    refs = get_references()
    st.caption(f"{len(refs)} reference image(s) for the ₹500 in references/")

# ------------------------------------------------------------------ empty state
if file is None:
    st.markdown("""
    <div class="steps">
      <div class="step"><div class="n">01</div><div class="t">Capture</div>
        <div class="d">Photograph the note flat, on a plain background, in even light.</div></div>
      <div class="step"><div class="n">02</div><div class="t">Inspect</div>
        <div class="d">Nine checks run: thread, sharpness, halftone, ink, texture, design, serial, watermark.</div></div>
      <div class="step"><div class="n">03</div><div class="t">Stress-test</div>
        <div class="d">Simulate a photocopy or inkjet reprint and watch the verdict react live.</div></div>
    </div>""", unsafe_allow_html=True)
    st.stop()

key = hashlib.md5(file.getvalue()).hexdigest() + "|".join(
    str(v) for v in (attack, wm_sim, cropped, getattr(wm_file, "name", None), getattr(wm_file, "size", None)))
new_input = st.session_state.get("scan_key") != key      # thresholds/weights changes do not replay it
img = load_image_bgr(file.getvalue())
if img is None:                                          # unreadable / corrupt / unsupported file: say so instead of crashing
    st.session_state["scan_key"] = key
    st.markdown('<div class="note bad"><b>This file could not be read as an image.</b><br>'
                f'<code>{html.escape(getattr(file, "name", "upload"))}</code> is probably corrupt or in a format that was only renamed to .png/.jpg '
                '(AVIF, HEIC, a web page saved as an image, ...). Open it, export it again as a normal PNG or JPG and upload that.</div>',
                unsafe_allow_html=True)
    st.stop()
# Attacks are applied to the NOTE itself (straightened first), like photocopying a real note,
# so the checks stay aligned and only the genuine effect of the attack is measured.
fn = {"Photocopy": photocopy, "Inkjet reprint": inkjet, "Low-res scan": lowres,
      "Printed fake watermark": print_watermark}.get(attack)
img0 = img.copy()
r0 = rectify(img)
base = r0["note"] if (r0["found"] or cropped) else None      # None -> no note in the image
pre_rect = False

# ---- which note is it? Read the printed amount (needs Tesseract), then check it is a note at all
cand = r0["note"] if r0["found"] else (img0 if cropped else None)
amount, amount_conf, amount_src = None, 0.0, None
amount_votes, amount_veto = {}, None
if cand is not None:
    with st.spinner("Reading the note…"):
        hr = r0["note_hr"] if r0["found"] else cv2.resize(img0, (2 * C.NOTE_W, 2 * C.NOTE_H), interpolation=cv2.INTER_CUBIC)
        # printed amount (needs Tesseract) -> design match against references/<value>/ -> dominant colour
        ident = identify_note(hr, cv2.resize(cand, (C.NOTE_W, C.NOTE_H), interpolation=cv2.INTER_AREA), get_reference_sets())
        amount, amount_conf, amount_src = ident["value"], ident["conf"], ident["source"]
        amount_votes, amount_veto = ident.get("votes", {}), ident.get("vetoed")
# ---- front or back? (before the colour gate: the back of a lavender/magenta note would otherwise look "off-palette")
side_refs = get_side_references()
if cand is not None and side_refs:
    cand_n = cv2.resize(cand, (C.NOTE_W, C.NOTE_H), interpolation=cv2.INTER_AREA)
    rev_in, any_obv = best_inliers(cand_n, side_refs), best_inliers(cand_n, get_all_references())
    if rev_in >= 15 and rev_in > 2 * max(any_obv, 1):
        st.session_state["scan_key"] = key
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            st.markdown('<div class="note"><b>This is the back of the note.</b><br>It matches the reverse side of a note we have a '
                        'specimen for. Please upload a photo of the <b>front</b> (the side with the portrait).</div>', unsafe_allow_html=True)
        with right:
            show(rgb(img0), caption="Uploaded image")
        st.stop()
if gate_on and cand is not None and amount_src not in ("text", "design"):    # a read amount / matched design proves it is a note, whatever its colours
    gate = check_banknote(cand, None if r0["found"] else max(img0.shape[:2]) / min(img0.shape[:2]))
    if not gate["ok"]:
        if animate and new_input:
            play_scan(None)
        st.session_state["scan_key"] = key
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            why = "".join(f"<li>{html.escape(r)}</li>" for r in gate["reasons"])
            st.markdown('<div class="note"><b>This does not look like a banknote.</b>'
                        f'<ul style="margin:.6rem 0 .6rem 1.1rem;padding:0">{why}</ul>'
                        'Upload a photo of the note itself. The denomination is read from the printed amount '
                        '("THE SUM OF ... RUPEES", needs Tesseract), recognised from the references/ folder, or estimated from the colour. '
                        'Turn off <i>Reject non-banknote images</i> in the sidebar to analyse it anyway.</div>',
                        unsafe_allow_html=True)
        with right:
            show(rgb(img0), caption="Uploaded image")
        st.stop()

# ---- which side, which design, what status?
banner, limited = None, False
profile = amount if amount in C.PROFILES else 500                 # layout profile (the Rs 500 one when the amount is unknown)
C.set_profile(profile)
refs = get_references(profile)                                    # genuine references of THIS denomination
obv_in = 0
if cand is not None and refs:
    obv_in = best_inliers(cv2.resize(cand, (C.NOTE_W, C.NOTE_H), interpolation=cv2.INTER_AREA), refs)
if amount == 500 and refs and obv_in < 10:
    banner = OLD_500
elif amount in NOTES:
    banner = NOTES[amount]
    limited = amount not in C.PROFILES                           # no layout profile (e.g. the withdrawn Rs 1000) -> generic checks only

# ---- simulated attack on the straightened note (uses the active denomination profile)
if fn and base is not None:
    img, pre_rect = fn(base), True

backlit_img = None
if wm_file is not None:
    backlit_img = load_image_bgr(wm_file.getvalue())
elif wm_sim != "Off":
    backlit_img = backlit(base if base is not None else img, "genuine" if wm_sim == "Genuine watermark" else "none")

with st.spinner("Analysing security features..."):
    res = analyze(img, refs, weights, g_at, s_at, backlit_bgr=backlit_img, assume_cropped=pre_rect, limited=limited, profile=profile)

# ------------------------------------------------------------------ no note found
if not res["note_found"] and not cropped:
    if animate and new_input:
        play_scan(None)
    st.session_state["scan_key"] = key
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.markdown(
            '<div class="note"><b>No banknote detected.</b><br>The edges of a note could not be outlined in this image, '
            'so no verdict is given. Use a plain background with the whole note in frame, or switch on '
            '<i>Image is already a cropped note</i> in the sidebar if the photo contains only the note.</div>',
            unsafe_allow_html=True)
    with right:
        show(rgb(res["original"]), caption="Uploaded image")
    st.stop()

if animate and new_input:
    play_scan(res["results"])
st.session_state["scan_key"] = key

# ------------------------------------------------------------------ verdict
active = [r for r in res["results"] if r.score is not None]
passed = sum(r.passed for r in active)
col = {"GENUINE": GREEN, "NO RED FLAGS": GREEN, "SUSPICIOUS": AMBER}.get(res["verdict"], RED)
circ = 2 * np.pi * 52
pct = int(round(res["score"] * 100))
gauge = f"""
<svg viewBox="0 0 120 120" width="128" height="128" style="flex:none">
  <circle cx="60" cy="60" r="52" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="7"/>
  <circle cx="60" cy="60" r="52" fill="none" stroke="{col}" stroke-width="7" stroke-linecap="round"
          stroke-dasharray="{circ * res['score']:.1f} {circ:.1f}" transform="rotate(-90 60 60)"/>
  <text x="60" y="65" text-anchor="middle" font-family="Fraunces, Georgia, serif" font-size="30" fill="#f3f1ea">{pct}</text>
  <text x="60" y="82" text-anchor="middle" font-size="7" letter-spacing="1.6" fill="{MUTED}">AUTHENTICITY</text>
</svg>"""

rows = ""
for r in res["results"]:
    if r.score is None:
        rows += (f'<div class="crow"><div class="nm">{r.name}<span class="mt">skipped</span></div>'
                 f'<div class="bar"></div><div class="pc">-</div></div>')
    else:
        rows += (f'<div class="crow"><div class="nm">{r.name}<span class="mt">{html.escape(r.metric)}</span></div>'
                 f'<div class="bar"><div style="width:{r.score*100:.0f}%;background:{tone(r.score)}"></div></div>'
                 f'<div class="pc">{r.score*100:.0f}%</div></div>')

if banner:
    extra = ""
    if limited:
        extra = ("<br><br>There is no layout profile for this denomination, so just the three generic print checks "
                 "(sharpness, halftone, ink) ran. They can show signs of copying but cannot confirm the note.")
    elif amount in C.PROFILES and not C.PROFILES[amount]["validated"]:
        extra = ("<br><br>The layout profile for this denomination was measured from the official RBI specimen only and has not been "
                 "validated on real photos, so the verdict is limited to <i>No red flags</i> or <i>Suspicious</i>.")
    if amount in NOT_IN_USE:
        extra += "<br><br>The analysis below only judges how the note was printed. It does not change the note's status above."
    if amount_src in ("design", "colour"):
        extra += f"<br><br><span style='opacity:.75;font-size:.85rem'>Denomination {SOURCE_LABEL[amount_src]}.</span>"
    if amount_votes:
        extra += ("<br><span style='opacity:.6;font-size:.78rem'>Denomination votes: "
                  + ", ".join(f"₹{v}: {w:.2f}" for v, w in sorted(amount_votes.items(), key=lambda kv: -kv[1]))
                  + (f" (design match for ₹{amount_veto} ignored: the note's colour contradicts it)" if amount_veto else "") + "</span>")
    st.markdown(banner_html(banner, extra), unsafe_allow_html=True)
    st.write("")
if res.get("decisive"):
    why = "".join(f"<li><b>{html.escape(n)}</b>: {html.escape(d)}</li>" for n, d in res["decisive"])
    st.markdown(f'<div class="note bad"><span class="badge">COUNTERFEIT EVIDENCE</span>'
                f'<ul style="margin:.4rem 0 .6rem 1.1rem;padding:0">{why}</ul>'
                '<span style="opacity:.75;font-size:.85rem">The verdict and the low score come from this evidence, not from averaging the checks. '
                'The other checks below can still be green because a toy or counterfeit note can copy the design of a real one; '
                'they only show that the design matches, not that the note is genuine.</span></div>', unsafe_allow_html=True)
    st.write("")
left, right = st.columns([1, 1.25], gap="large")
with left:
    st.markdown(
        f'<div class="vcard">{gauge}<div><div class="lbl">Verdict</div>'
        f'<div class="word" style="color:{col}">{res["verdict"].title()}</div>'
        f'<div class="sub">{passed} of {len(active)} checks passed'
        f'{"<br>verdict set by decisive evidence" if res.get("decisive") else ""}</div></div></div>'
        f'<div class="clist">{rows}</div>', unsafe_allow_html=True)
with right:
    show(rgb(res["annotated"]), caption="Rectified note. Green regions passed, red regions failed.")

st.write("")
t1, t2, t3 = st.tabs(["Evidence", "Pipeline", "Report"])

with t1:
    st.caption("Open a check to see what was measured and the visual proof.")
    for r in res["results"]:
        tag = "skipped" if r.score is None else f"{r.score*100:.0f}%"
        mark = "○" if r.score is None else ("●" if r.passed else "○")
        with st.expander(f"{mark}  {r.name}  -  {tag}"):
            a, b = st.columns([1, 1.2], gap="large")
            a.markdown(f"**{r.metric or 'n/a'}**")
            a.caption(r.detail)
            if r.debug is not None:
                with b:
                    show(rgb(r.debug))

with t2:
    a, b, c = st.columns(3, gap="medium")
    with a: show(rgb(res["outline"]), caption="1  Detected outline")
    with b: show(res["edges"], caption="2  Canny edges")
    with c: show(rgb(res["note"]), caption="3  Perspective-corrected note")
    st.caption("Input → edges → contour → 4-corner fit → perspective warp → 7 checks → weakest-link fusion → verdict")

with t3:
    report = {"denomination": amount, "denomination_source": amount_src,
              "status": banner["title"] if banner else None,
              "verdict": res["verdict"], "score": round(res["score"], 3),
              "checks": [{"name": r.name, "score": None if r.score is None else round(r.score, 3),
                          "metric": r.metric, "detail": r.detail} for r in res["results"]]}
    st.json(report, expanded=False)
    st.download_button("Download JSON report", json.dumps(report, indent=2), "report.json", "application/json")
    st.caption("Screening aid only. Phone cameras cannot capture UV, microprinting or tactile features, "
               "so suspicious notes should be verified by a bank.")