import streamlit as st
import requests
from PIL import Image
import io
import base64

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Root variables ── */
:root {
    --bg:       #0a0c10;
    --surface:  #111318;
    --border:   #1e2230;
    --accent:   #00e5ff;
    --accent2:  #7c3aed;
    --danger:   #ff3b5c;
    --success:  #00e676;
    --warn:     #ffab00;
    --text:     #e8eaf6;
    --muted:    #5c6278;
    --mono:     'Space Mono', monospace;
    --sans:     'Syne', sans-serif;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 720px; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 0.5rem;
    filter: drop-shadow(0 0 24px var(--accent));
    animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { filter: drop-shadow(0 0 12px var(--accent)); }
    50%       { filter: drop-shadow(0 0 32px var(--accent)); }
}
.hero h1 {
    font-family: var(--sans) !important;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    letter-spacing: -0.03em;
    transform: scaleY(1.7);
    display: inline-block;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
    padding: 0 !important;
}
.hero p {
    color: var(--muted) !important;
    font-size: 0.9rem;
    font-family: var(--mono) !important;
    margin-top: 0.5rem;
}

/* ── Section labels ── */
.section-label {
    font-family: var(--mono) !important;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
    display: block;
}

/* ── Cards / panels ── */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* ── Inputs & selects ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0, 229, 255, 0.15) !important;
}

/* ── File uploader ── */
.stFileUploader > div {
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
    background: var(--bg) !important;
}
.stFileUploader > div:hover {
    border-color: var(--accent) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
.stButton > button:hover  { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0px) !important; }

/* ── Radio tabs ── */
.stRadio > div { gap: 0.5rem; }
.stRadio > div > label {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0.35rem 1rem !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    cursor: pointer;
    transition: border-color 0.2s;
}
.stRadio > div > label:hover { border-color: var(--accent) !important; }

/* ── Result card ── */
.result-card {
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    border-radius: 4px 0 0 4px;
}
.result-benign   { background: rgba(0,230,118,.08); border: 1px solid rgba(0,230,118,.25); }
.result-benign::before { background: var(--success); }
.result-malignant{ background: rgba(255,59,92,.08); border: 1px solid rgba(255,59,92,.25); }
.result-malignant::before { background: var(--danger); }
.result-unknown  { background: rgba(255,171,0,.08);  border: 1px solid rgba(255,171,0,.25); }
.result-unknown::before  { background: var(--warn); }

.result-label {
    font-family: var(--mono) !important;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.result-pred {
    font-family: var(--sans) !important;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.result-meta {
    font-family: var(--mono) !important;
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.6rem;
}
.result-meta span { color: var(--text); }

/* ── Confidence bar ── */
.conf-bar-bg {
    background: var(--border);
    border-radius: 99px;
    height: 6px;
    margin-top: 1rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}

/* ── Error / info boxes ── */
.stAlert { border-radius: 8px !important; font-family: var(--mono) !important; font-size: 0.82rem !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Spinner text ── */
.stSpinner > div { color: var(--accent) !important; }

/* ── API URL field label fix ── */
label { color: var(--muted) !important; font-size: 0.75rem !important; font-family: var(--mono) !important; }
</style>
""", unsafe_allow_html=True)


# ── Helper ────────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: ("No Tumor",        "result-benign",    "🟢"),
    1: ("Glioma",          "result-malignant", "⚠️"),
    2: ("Meningioma",      "result-malignant", "⚠️"),
    3: ("Pituitary Tumor", "result-unknown",   "🟡"),
}

def get_class_info(cls: int):
    return CLASS_NAMES.get(cls, (f"Class {cls}", "result-unknown", "🔵"))

def render_result(data: dict):
    cls      = data.get("predicted_class", -1)
    pred     = data.get("prediction", "Unknown")
    conf     = data.get("confidence", 0.0)
    method   = data.get("method", "—")
    label, css, icon = get_class_info(cls)
    conf_pct = conf * 100
    bar_color = "#00e676" if "benign" in css else ("#ff3b5c" if "malignant" in css else "#ffab00")

    st.markdown(f"""
    <div class="result-card {css}">
        <div class="result-label">Diagnosis Result</div>
        <div class="result-pred">{icon} {pred}</div>
        <div class="result-meta">
            Class index: <span>{cls}</span> &nbsp;·&nbsp;
            Method: <span>{method}</span>
        </div>
        <div class="result-meta" style="margin-top:.4rem">
            Confidence: <span>{conf_pct:.1f}%</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill"
                 style="width:{conf_pct:.1f}%; background:{bar_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-icon">🧠</span>
    <h1>Brain Tumor Classifier</h1>
    <p>MRI Analysis via Deep Learning · SVM / Gompertz / Mitscherlich</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar: API config ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ API Configuration")
    api_base = st.text_input(
        "API Base URL",
        value="http://localhost:8000",
        help="Base URL of the FastAPI server",
    )
    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#5c6278; line-height:1.7">
    <b style="color:#00e5ff">Endpoints</b><br>
    POST /predict/upload<br>
    POST /predict/url<br>
    POST /predict<br>
    GET  /health
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔍 Check Health"):
        try:
            r = requests.get(f"{api_base}/health", timeout=5)
            if r.status_code == 200:
                st.success("Server is online ✅")
            else:
                st.error(f"Server returned {r.status_code}")
        except Exception as e:
            st.error(f"Cannot reach server:\n{e}")


# ── Method selector ───────────────────────────────────────────────────────────
st.markdown('<span class="section-label">Classification Method</span>', unsafe_allow_html=True)
method = st.selectbox(
    "method",
    options=["Gompertz", "SVM", "Mitscherlich"],
    index=0,
    label_visibility="collapsed",
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Input mode tabs ───────────────────────────────────────────────────────────
st.markdown('<span class="section-label">Input Mode</span>', unsafe_allow_html=True)
mode = st.radio(
    "input_mode",
    options=["📁  Upload File", "🔗  Image URL"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Upload mode ───────────────────────────────────────────────────────────────
if mode == "📁  Upload File":
    st.markdown('<span class="section-label">MRI Image</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop MRI scan here",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded:
        col_img, col_info = st.columns([1, 1])
        with col_img:
            img = Image.open(uploaded)
            st.image(img, caption=uploaded.name, use_container_width=True)
        with col_info:
            st.markdown(f"""
            <div style="font-family:'Space Mono',monospace; font-size:0.75rem;
                        color:#5c6278; line-height:2; padding-top:0.5rem">
            <b style="color:#e8eaf6">File</b><br>{uploaded.name}<br><br>
            <b style="color:#e8eaf6">Size</b><br>{uploaded.size / 1024:.1f} KB<br><br>
            <b style="color:#e8eaf6">Type</b><br>{uploaded.type}<br><br>
            <b style="color:#e8eaf6">Dimensions</b><br>{img.width} × {img.height} px
            </div>
            """, unsafe_allow_html=True)

    if st.button("🧬 Run Classification", key="btn_upload"):
        if not uploaded:
            st.warning("Please upload an MRI image first.")
        else:
            with st.spinner("Analysing image…"):
                try:
                    uploaded.seek(0)
                    files   = {"file":   (uploaded.name, uploaded, uploaded.type)}
                    payload = {"method": method}
                    r = requests.post(
                        f"{api_base}/predict/upload",
                        files=files,
                        data=payload,
                        timeout=30,
                    )
                    if r.status_code == 200:
                        render_result(r.json())
                    else:
                        detail = r.json().get("detail", r.text)
                        st.error(f"**API Error {r.status_code}:** {detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the API. Is the server running?")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

# ── URL mode ──────────────────────────────────────────────────────────────────
else:
    st.markdown('<span class="section-label">Image URL</span>', unsafe_allow_html=True)
    image_url = st.text_input(
        "url",
        placeholder="https://example.com/brain_mri.jpg",
        label_visibility="collapsed",
    )

    if image_url and (image_url.startswith("http://") or image_url.startswith("https://")):
        try:
            preview_r = requests.get(image_url, timeout=10)
            if preview_r.status_code == 200 and "image" in preview_r.headers.get("content-type", ""):
                img_bytes = preview_r.content
                img = Image.open(io.BytesIO(img_bytes))
                st.image(img, caption="Preview", use_container_width=True)
        except Exception:
            st.caption("⚠️ Could not load preview (image will still be sent to API).")
    elif image_url:
        st.warning("URL must start with http:// or https://")

    if st.button("🧬 Run Classification", key="btn_url"):
        if not image_url:
            st.warning("Please enter an image URL.")
        elif not (image_url.startswith("http://") or image_url.startswith("https://")):
            st.error("Invalid URL format.")
        else:
            with st.spinner("Analysing image…"):
                try:
                    payload = {"url": image_url, "method": method}
                    r = requests.post(
                        f"{api_base}/predict/url",
                        json=payload,
                        timeout=30,
                    )
                    if r.status_code == 200:
                        render_result(r.json())
                    else:
                        detail = r.json().get("detail", r.text)
                        st.error(f"**API Error {r.status_code}:** {detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the API. Is the server running?")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; font-family:'Space Mono',monospace;
           font-size:0.68rem; color:#2a2e3d; letter-spacing:0.1em">
BRAIN TUMOR CLASSIFIER · FASTAPI + STREAMLIT
</p>
""", unsafe_allow_html=True)