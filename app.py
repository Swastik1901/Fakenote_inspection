import hashlib
import html
import json
import time

import cv2
import numpy as np
import streamlit as st

from detector import config as C
from detector.pipeline import analyze, get_references
from detector.gate import check_banknote
from detector.preprocess import rectify
from simulate_fake import photocopy, inkjet, lowres, print_watermark, backlit

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
    '<div class="eyebrow">Currency forensics &middot; classical image processing</div>'
    '<div class="title">FakeNote Inspector</div>'
    '<div class="tagline">Every security feature of the note is checked on its own and shown with evidence. '
    'No black box, just explainable computer vision.</div><div class="rule"></div>',
    unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Input")
    mode = st.radio("Source", ["Upload", "Webcam"], horizontal=True, label_visibility="collapsed")
    file = st.file_uploader("Note photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed") \
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
    refs = get_references()
    st.caption(f"{len(refs)} genuine reference image(s) in references/")

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
img = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
# Attacks are applied to the NOTE itself (straightened first), like photocopying a real note,
# so the checks stay aligned and only the genuine effect of the attack is measured.
fn = {"Photocopy": photocopy, "Inkjet reprint": inkjet, "Low-res scan": lowres,
      "Printed fake watermark": print_watermark}.get(attack)
img0 = img.copy()
r0 = rectify(img)
base = r0["note"] if (r0["found"] or cropped) else None      # None -> no note in the image
pre_rect = False
if fn and base is not None:
    img, pre_rect = fn(base), True

# ---- is it a banknote at all? (checked on the original, before any simulated attack)
cand = r0["note"] if r0["found"] else (img0 if cropped else None)
if gate_on and cand is not None:
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
                        'Upload a photo of the note itself. This build is tuned for the ₹500 Mahatma Gandhi series. '
                        'Turn off <i>Reject non-banknote images</i> in the sidebar to analyse it anyway.</div>',
                        unsafe_allow_html=True)
        with right:
            show(rgb(img0), caption="Uploaded image")
        st.stop()

backlit_img = None
if wm_file is not None:
    backlit_img = cv2.imdecode(np.frombuffer(wm_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
elif wm_sim != "Off":
    backlit_img = backlit(base if base is not None else img, "genuine" if wm_sim == "Genuine watermark" else "none")

with st.spinner("Analysing security features..."):
    res = analyze(img, refs, weights, g_at, s_at, backlit_bgr=backlit_img, assume_cropped=pre_rect)

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
col = {"GENUINE": GREEN, "SUSPICIOUS": AMBER}.get(res["verdict"], RED)
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

left, right = st.columns([1, 1.25], gap="large")
with left:
    st.markdown(
        f'<div class="vcard">{gauge}<div><div class="lbl">Verdict</div>'
        f'<div class="word" style="color:{col}">{res["verdict"].title()}</div>'
        f'<div class="sub">{passed} of {len(active)} checks passed</div></div></div>'
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
    report = {"verdict": res["verdict"], "score": round(res["score"], 3),
              "checks": [{"name": r.name, "score": None if r.score is None else round(r.score, 3),
                          "metric": r.metric, "detail": r.detail} for r in res["results"]]}
    st.json(report, expanded=False)
    st.download_button("Download JSON report", json.dumps(report, indent=2), "report.json", "application/json")
    st.caption("Screening aid only. Phone cameras cannot capture UV, microprinting or tactile features, "
               "so suspicious notes should be verified by a bank.")