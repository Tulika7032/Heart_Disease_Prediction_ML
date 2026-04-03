import streamlit as st
import numpy as np
import joblib
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioScan · Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
            
/* ── SIDEBAR WIDGET LABELS (Age, BP, etc.) ── */
section[data-testid="stSidebar"] label {
    font-size: 1.25rem !important; 
    font-weight: 600 !important;
    color: #cbd5e1 !important;
}

/* Widget labels (Age (years), Biological Sex, etc.) */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: 1.75rem !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important;
    display: block !important;
    margin-bottom: 6px !important;
}


/* Slider thumb tooltip / current value */
section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
}

/* Target slider value more broadly */
section[data-testid="stSidebar"] .stSlider div {
    font-size: 1.25rem !important;
}
            

/* Selectbox text (selected value) */
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    font-size: 1.15rem !important;
}

/* Selectbox selected value inner span */
section[data-testid="stSidebar"] .stSelectbox span {
    font-size: 1.15rem !important;
}


div[data-baseweb="popover"] span {
    font-size: 1.5rem !important;
}

/* Spacing in dropdown */
div[data-baseweb="popover"] ul li {
    font-size: 1.5rem !important;
    padding: 12px 16px !important;
    line-height: 1.6 !important;
}

/* Small helper text / paragraphs */
section[data-testid="stSidebar"] p {
    font-size: 1.1rem !important;
}
            

/* Bold section titles like "Cardiovascular", "Clinical" */
section[data-testid="stSidebar"] .stMarkdown p strong {
    font-size: 2.75rem !important;
    color: #e2e8f0 !important;
}
            

/* Target the actual displayed value */
section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    font-size: 1.5rem !important;
}

/* Increase selectbox height */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    min-height: 50px !important;
    height: 50px !important;
    display: flex !important;
    align-items: center !important;
}

/* Increase padding inside */
section[data-testid="stSidebar"] div[data-baseweb="select"] {
    padding-top: 4px !important;
    padding-bottom: 4px !important;
            
}           
       

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #FFFFFF;
    color: #e8e9ed;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #13161e;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: #f0655a;
    font-family: 'DM Serif Display', serif;
    font-size: 2.75rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

/* ── Metric cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.metric-card:hover {border-color: #f0655a55;
    box-shadow: 0 8px 20px rgba(0,0,0,0.10);
    transform: translateY(-2px);}
            
.metric-label {
    font-size: 1.8rem;
    letter-spacing: 0.0em;
    text-transform: uppercase;
    color: #444444;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-weight: 600;
    font-size: 2.25rem;
    color: #000000;
    line-height: 1;
}
.metric-unit {
    font-size: 1.65rem;
    color: #666666;
    margin-top: 4px;
}

/* ── Section headings ── */
.section-head {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #000000;
    border-left: 3px solid #f0655a;
    padding-left: 12px;
    margin-bottom: 16px;
}

/* ── Risk gauge wrapper ── */
.gauge-wrapper {
    background: #13161e;
    border: 1px solid #1e2130;
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
}

/* ── Result banner ── */
.banner-high {
    background: linear-gradient(135deg, #3b1215 0%, #1a0b0c 100%);
    border: 1px solid #f0655a55;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}
.banner-low {
    background: linear-gradient(135deg, #0b2418 0%, #071610 100%);
    border: 1px solid #3ecf8e55;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}
.banner-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    margin-bottom: 6px;
}
.banner-sub {
    font-size: 0.85rem;
    color: #8a8fa8;
}

/* ── Predict button ── */
.stButton > button {
    background: #1a56db !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 18px 40px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.25rem !important;
    letter-spacing: 0.04em !important;
    width: 100% !important;
    transition: background 0.2s !important;
    box-shadow: 0 2px 8px #1a56db55 !important;
}
.stButton > button:hover { background: #1648c0 !important; }
            
/* ── Divider ── */
hr { border-color: #1e2130 !important; }

</style>
""", unsafe_allow_html=True)


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    candidates = [
        "models/model.pkl",
        os.path.join(os.path.dirname(__file__), "models", "model.pkl"),
        os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return joblib.load(path)
    return None

model = load_model()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 8px 0 24px 0">
  <span style="font-family:'DM Serif Display',serif; font-size:5rem; color:#000103">
    🫀 CardioScan
  </span>
  <span style="font-size:3rem; color:#000103; margin-left:14px">
    Heart Disease Risk Predictor
  </span>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file not found. Please train and save the model first (`python main.py`).")
    st.stop()


# ── Sidebar: Patient Inputs ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧾 Patient Details")
    st.markdown("---")

    age        = st.slider("Age (years)", 20, 80, 45)
    sex_label  = st.selectbox("Biological Sex", ["Female", "Male"])
    sex        = 1 if sex_label == "Male" else 0

    st.markdown("**Cardiovascular**")
    trestbps   = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol       = st.slider("Serum Cholesterol (mg/dl)", 100, 400, 220)
    thalach    = st.slider("Max Heart Rate Achieved", 70, 210, 150)
    oldpeak    = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1)

    st.markdown("**Clinical**")
    cp         = st.selectbox("Chest Pain Type",
                              [0, 1, 2, 3],
                              format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",2:"Non-Anginal",3:"Asymptomatic"}[x])
    fbs_label  = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    fbs        = 1 if fbs_label == "Yes" else 0
    restecg    = st.selectbox("Resting ECG Result",
                              [0, 1, 2],
                              format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"LV Hypertrophy"}[x])
    exang_label= st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    exang      = 1 if exang_label == "Yes" else 0
    slope      = st.selectbox("ST Slope",
                              [0, 1, 2],
                              format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
    ca         = st.selectbox("Major Vessels Colored (CA)", [0, 1, 2, 3, 4])
    thal       = st.selectbox("Thalassemia",
                              [0, 1, 2, 3],
                              format_func=lambda x: {0:"Unknown",1:"Normal",2:"Fixed Defect",3:"Reversable Defect"}[x])


# ── Input array ───────────────────────────────────────────────────────────────
input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak,
                        slope, ca, thal]])


# ── Patient Vitals (full width) ───────────────────────────────────────────────
st.markdown('<div class="section-head">Patient Vitals</div>', unsafe_allow_html=True)

r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Age</div>
      <div class="metric-value">{age}</div>
      <div class="metric-unit">years · {sex_label}</div>
    </div>""", unsafe_allow_html=True)
with r1c2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Blood Pressure</div>
      <div class="metric-value">{trestbps}</div>
      <div class="metric-unit">mm Hg</div>
    </div>""", unsafe_allow_html=True)
with r1c3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Cholesterol</div>
      <div class="metric-value">{chol}</div>
      <div class="metric-unit">mg/dl</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Max Heart Rate</div>
      <div class="metric-value">{thalach}</div>
      <div class="metric-unit">bpm</div>
    </div>""", unsafe_allow_html=True)
with r2c2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">ST Depression</div>
      <div class="metric-value">{oldpeak}</div>
      <div class="metric-unit">Oldpeak</div>
    </div>""", unsafe_allow_html=True)
with r2c3:
    chest_labels = {0:"Typical",1:"Atypical",2:"Non-Anginal",3:"Asymp."}
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Chest Pain</div>
      <div class="metric-value">{chest_labels[cp]}</div>
      <div class="metric-unit">Type {cp}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-head">Additional Flags</div>', unsafe_allow_html=True)
flag_cols = st.columns(4)
flags = [
    ("FBS > 120", "Yes" if fbs else "No"),
    ("Angina", "Yes" if exang else "No"),
    ("Vessels (CA)", str(ca)),
    ("Thal", {0:"Unk",1:"Norm",2:"Fixed",3:"Revers."}[thal]),
]
for col, (label, val) in zip(flag_cols, flags):
    color = "#e13427" if val in ("Yes",) else "#0eac65" if val == "No" else "#000000"
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="font-size:1.9rem;font-weight:600;color:{color}">{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([4, 1])

with col1:
    st.markdown('<div class="section-head">Risk Assessment</div>', unsafe_allow_html=True)

with col2:
    predict_clicked = st.button("🔍 Run Prediction")

if predict_clicked:
    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    risk_pct    = int(probability[1] * 100)

    st.markdown("<br>", unsafe_allow_html=True)

    gauge_col, result_col = st.columns([1, 1], gap="large")

    with gauge_col:
        angle       = -90 + (180 * probability[1])
        gauge_color = "#f0655a" if prediction == 1 else "#3ecf8e"
        gauge_html  = f"""
        <div class="gauge-wrapper">
        <svg viewBox="0 0 220 130" xmlns="http://www.w3.org/2000/svg"
             style="width:100%;max-width:300px;margin:0 auto;display:block">
          <path d="M 20 110 A 75 75 0 0 1 200 100" fill="none" stroke="#1e2130" stroke-width="14" stroke-linecap="round"/>
          <path d="M 20 110 A 75 75 0 0 1 200 100" fill="none" stroke="{gauge_color}" stroke-width="14"
                stroke-linecap="round" stroke-dasharray="263" stroke-dashoffset="{int(283*(1-probability[1]))}"/>
          <line x1="110" y1="110"
                x2="{110 + 70*np.cos(np.radians(angle)):.1f}"
                y2="{110 + 70*np.sin(np.radians(angle)):.1f}"
                stroke="#e8e9ed" stroke-width="2.5" stroke-linecap="round"/>
          <circle cx="110" cy="110" r="5" fill="#e8e9ed"/>
          <text x="10"  y="130" font-size="15" fill="#636880" font-weight="600">LOW</text>
          <text x="220" y="130" font-size="15" fill="#636880" text-anchor="end" font-weight="600">HIGH</text>
          <text x="110" y="76" font-size="26" fill="{gauge_color}" text-anchor="middle" font-weight="700">{risk_pct}%</text>
          <text x="110" y="98" font-size="15" fill="#636880" text-anchor="middle" font-weight="600">RISK SCORE</text>
        </svg>
        <br>
        <div style="display:flex;gap:12px;justify-content:center">
          <div class="metric-card" style="flex:1">
            <div class="metric-label">No Disease</div>
            <div class="metric-value" style="color:#3ecf8e">{int(probability[0]*100)}%</div>
            <div class="metric-unit">confidence</div>
          </div>
          <div class="metric-card" style="flex:1">
            <div class="metric-label">Disease</div>
            <div class="metric-value" style="color:#f0655a">{int(probability[1]*100)}%</div>
            <div class="metric-unit">confidence</div>
          </div>
        </div>
        </div>
        """
        st.markdown(gauge_html, unsafe_allow_html=True)

    with result_col:
        if prediction == 1:
            st.markdown(f"""
            <div class="banner-high" style="height:100%;display:flex;flex-direction:column;justify-content:center;min-height:220px">
              <div class="banner-title" style="color:#f0655a;font-size:2.8rem">⚠️ High Risk Detected</div>
              <div class="banner-sub" style="margin-top:12px;font-size:1.7rem;line-height:1.7">
                The model predicts an <b style="color:#f0655a">elevated likelihood</b> of heart disease based on the provided parameters.<br><br>
                Please consult a cardiologist for a full clinical evaluation.
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="banner-low" style="height:100%;display:flex;flex-direction:column;justify-content:center;min-height:220px">
              <div class="banner-title" style="color:#3ecf8e;font-size:2.8rem">✅ Low Risk</div>
              <div class="banner-sub" style="margin-top:12px;font-size:1.7rem;line-height:1.7">
                The model predicts a <b style="color:#3ecf8e">low likelihood</b> of heart disease based on the provided parameters.<br><br>
                Continue regular health check-ups and maintain a heart-healthy lifestyle.
              </div>
            </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:48px 0;color:#636880">
      <div style="font-size:3.5rem">🫀</div>
      <div style="margin-top:16px;font-size:2rem">
        Fill in patient details in the sidebar, then click <b style="color:#1a56db">Run Prediction</b>
      </div>
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#3a3d4d;font-size:1.5rem;padding:8px 0">
  CardioScan · For research & educational use only · Not a substitute for professional medical advice
</div>""", unsafe_allow_html=True)