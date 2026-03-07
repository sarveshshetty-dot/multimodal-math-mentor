import streamlit as st
import tempfile
import os
from input_processing.image_ocr import ImageOCRProcessor
from input_processing.speech_to_text import SpeechProcessor
from input_processing.text_input import TextProcessor


def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #111827 50%, #0a0f1e 100%);
        min-height: 100vh;
    }

    /* Hide default header */
    header[data-testid="stHeader"] {
        background: rgba(13, 13, 26, 0.8);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(99, 102, 241, 0.2);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0d1117 100%) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #9ca3af !important;
        font-size: 0.85rem;
    }
    [data-testid="stSidebar"] h1 {
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Main title */
    h1 {
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.8rem !important;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem !important;
    }

    /* Sub-headings */
    h2, h3 {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    /* Cards / Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
    }

    /* Text input / text area */
    .stTextArea textarea {
        background: rgba(17, 24, 39, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        transition: border-color 0.2s ease;
    }
    .stTextArea textarea:focus {
        border-color: rgba(99, 102, 241, 0.7) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }

    /* Radio buttons */
    .stRadio > label {
        color: #9ca3af !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
    }
    .stRadio [data-baseweb="radio"] {
        background: rgba(99, 102, 241, 0.08) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease;
    }
    .stRadio [data-baseweb="radio"]:hover {
        border-color: rgba(99, 102, 241, 0.5) !important;
        background: rgba(99, 102, 241, 0.15) !important;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2) !important;
        width: auto !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
        filter: brightness(1.1);
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(1px) !important;
    }

    /* Secondary buttons */
    .stButton > button {
        background: rgba(99, 102, 241, 0.1) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        color: #a5b4fc !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: rgba(99, 102, 241, 0.2) !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
    }

    /* Status / spinner */
    [data-testid="stStatusWidget"] {
        background: rgba(17, 24, 39, 0.9) !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 12px !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(99, 102, 241, 0.08) !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 10px !important;
        color: #a5b4fc !important;
        font-weight: 500 !important;
    }
    .streamlit-expanderContent {
        background: rgba(13, 13, 26, 0.5) !important;
        border: 1px solid rgba(99, 102, 241, 0.1) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }

    /* Metric */
    [data-testid="stMetric"] {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricValue"] {
        color: #818cf8 !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
    }

    /* Info / success / warning / error boxes */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 10px !important;
        color: #6ee7b7 !important;
    }
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 10px !important;
    }
    .stWarning {
        background: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
        border-radius: 10px !important;
    }
    .stInfo {
        background: rgba(99, 102, 241, 0.1) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 10px !important;
        color: #a5b4fc !important;
    }

    /* JSON display */
    .stJson {
        background: rgba(13, 13, 26, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 10px !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(99, 102, 241, 0.05) !important;
        border: 2px dashed rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(99, 102, 241, 0.6) !important;
        background: rgba(99, 102, 241, 0.08) !important;
    }

    /* Divider */
    hr {
        border-color: rgba(99, 102, 241, 0.15) !important;
        margin: 24px 0 !important;
    }

    /* Default text color */
    p, li, span {
        color: #d1d5db;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Glow effect on section titles */
    .section-glow {
        color: #818cf8;
        font-weight: 600;
        font-size: 1.1rem;
        text-shadow: 0 0 20px rgba(129, 140, 248, 0.4);
        margin-bottom: 12px;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    inject_custom_css()

    st.sidebar.markdown("""
        <div style='padding: 8px 0; margin-bottom: 8px;'>
            <h1 style='font-size:1.4rem; margin:0;'>🧠 Math Mentor AI</h1>
            <p style='color:#6b7280; font-size:0.78rem; margin-top:4px;'>Offline · Local AI · JEE-Level</p>
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.divider()
    st.sidebar.markdown("""
        <p style='color:#9ca3af; font-size:0.83rem; line-height:1.6;'>
        Solves JEE-level math using<br>
        <span style='color:#818cf8;'>▸ RAG</span> · 
        <span style='color:#c084fc;'>▸ Multi-Agent</span> · 
        <span style='color:#f472b6;'>▸ SymPy</span>
        </p>
    """, unsafe_allow_html=True)

    st.sidebar.divider()
    st.sidebar.markdown("<p class='section-glow'>⚙ Settings</p>", unsafe_allow_html=True)
    st.sidebar.info("Knowledge Base loaded from local FAISS index.")
    st.sidebar.markdown("""
        <div style='margin-top:16px;'>
            <p style='font-size:0.78rem; color:#4b5563;'>
            Models: <span style='color:#818cf8;'>TinyLlama-1.1B</span><br>
            Embeddings: <span style='color:#818cf8;'>all-MiniLM-L6-v2</span><br>
            Vector DB: <span style='color:#818cf8;'>FAISS (local)</span>
            </p>
        </div>
    """, unsafe_allow_html=True)


# input_mode and raw_text logic is now handled in app.py for cleaner layout control


def render_final_output(explanation: str, confidence_data: dict):
    """Renders the final explanation and confidence score."""
    st.markdown("---")

    st.markdown("""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:20px;'>
            <span style='font-size:1.5rem;'>✨</span>
            <span style='font-size:1.2rem; font-weight:700; background:linear-gradient(135deg,#818cf8,#c084fc);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
                Solution & Step-by-Step Explanation
            </span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class='glass-card'>
            {explanation}
        </div>
    """, unsafe_allow_html=True)

    conf = confidence_data.get("confidence", 0)
    reason = confidence_data.get("reason", "")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(label="🎯 Confidence", value=f"{conf}/100")
    with col2:
        if conf >= 90:
            st.success(f"✅ {reason}")
        elif conf >= 70:
            st.warning(f"⚠️ {reason}")
        else:
            st.error(f"🔴 {reason} — Human review recommended.")
