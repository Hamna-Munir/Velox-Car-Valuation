import io
import textwrap
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from velox.config import CURRENT_YEAR, FUEL_TYPES, OWNER_TYPES, SELLER_TYPES, TRANSMISSIONS
from velox.data import get_or_build_clean_dataset
from velox.explain import explain_prediction, global_feature_importance
from velox.logging_config import setup_logging
from velox.model import load_models

setup_logging()

# ==============================================================================
# Page config
# ==============================================================================
st.set_page_config(
    page_title="VELOX — Vehicle Valuation Engine",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# Original detailed car illustrations (hand-built SVG, gradient-shaded — no
# external images, so nothing to break on deploy and nothing to license)
# ==============================================================================
def car_illustration_svg(body_style="sedan", body_color="#E10600", body_dark="#8A0400",
                          uid="car", width="100%"):
    """A gradient-shaded car illustration. body_style: sedan | suv | hatchback | coupe."""
    roofs = {
        "sedan":     "M118 58 Q138 30 190 27 L248 27 Q292 30 308 58 Z",
        "suv":       "M112 50 Q130 18 188 15 L252 15 Q296 18 312 50 Z",
        "hatchback": "M120 58 Q140 32 190 29 L236 29 Q270 32 284 58 Z",
        "coupe":     "M128 62 Q148 40 195 37 L245 37 Q278 40 296 62 Z",
    }
    bodies = {
        "sedan":     "M36 96 Q36 64 74 61 L118 61 L308 61 L346 64 Q372 67 372 96 L372 102 L36 102 Z",
        "suv":       "M32 100 Q32 58 70 55 L112 55 L312 55 L354 58 Q378 62 378 100 L378 108 L32 108 Z",
        "hatchback": "M40 96 Q40 65 76 62 L120 62 L284 62 L318 66 Q340 69 340 96 L340 102 L40 102 Z",
        "coupe":     "M42 98 Q42 70 80 66 L128 66 L296 66 L330 70 Q352 74 352 98 L352 103 L42 103 Z",
    }
    roof = roofs[body_style]
    body = bodies[body_style]
    vb_h = 160 if body_style == "suv" else 150
    wheel_y = 112 if body_style == "suv" else 108
    wheel_r = 27 if body_style == "suv" else 25
    wx1, wx2 = (108, 302) if body_style != "hatchback" else (112, 282)

    return f"""
    <svg viewBox="0 0 410 {vb_h}" width="{width}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="{uid}-body" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="{body_color}"/>
                <stop offset="65%" stop-color="{body_color}"/>
                <stop offset="100%" stop-color="{body_dark}"/>
            </linearGradient>
            <linearGradient id="{uid}-glass" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#3A4048"/>
                <stop offset="100%" stop-color="#12151A"/>
            </linearGradient>
            <radialGradient id="{uid}-wheel" cx="0.35" cy="0.35" r="0.7">
                <stop offset="0%" stop-color="#3A3A3A"/>
                <stop offset="100%" stop-color="#0A0A0A"/>
            </radialGradient>
            <linearGradient id="{uid}-shine" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0"/>
                <stop offset="50%" stop-color="#FFFFFF" stop-opacity="0.35"/>
                <stop offset="100%" stop-color="#FFFFFF" stop-opacity="0"/>
            </linearGradient>
        </defs>
        <ellipse cx="205" cy="{vb_h-12}" rx="185" ry="7" fill="#000000" opacity="0.45"/>
        <path d="{body}" fill="url(#{uid}-body)" stroke="#000000" stroke-opacity="0.3" stroke-width="1"/>
        <path d="{roof}" fill="url(#{uid}-glass)"/>
        <path d="{roof}" fill="none" stroke="{body_color}" stroke-width="2" stroke-opacity="0.6"/>
        <rect x="30" y="70" width="350" height="5" fill="url(#{uid}-shine)" opacity="0.5"/>
        <circle cx="{wx1}" cy="{wheel_y}" r="{wheel_r}" fill="url(#{uid}-wheel)" stroke="#050505" stroke-width="2"/>
        <circle cx="{wx1}" cy="{wheel_y}" r="{wheel_r*0.42:.0f}" fill="#C9C9C9"/>
        <circle cx="{wx1}" cy="{wheel_y}" r="{wheel_r*0.14:.0f}" fill="#3A3A3A"/>
        <circle cx="{wx2}" cy="{wheel_y}" r="{wheel_r}" fill="url(#{uid}-wheel)" stroke="#050505" stroke-width="2"/>
        <circle cx="{wx2}" cy="{wheel_y}" r="{wheel_r*0.42:.0f}" fill="#C9C9C9"/>
        <circle cx="{wx2}" cy="{wheel_y}" r="{wheel_r*0.14:.0f}" fill="#3A3A3A"/>
        <rect x="358" y="75" width="16" height="9" rx="2" fill="#FFFFFF"/>
        <rect x="358" y="75" width="16" height="9" rx="2" fill="#FFFFFF" opacity="0.6"/>
        <rect x="34" y="82" width="12" height="6" rx="2" fill="#FF3B3B"/>
        <rect x="60" y="99" width="290" height="4" fill="#000000" opacity="0.25"/>
    </svg>
    """


HERO_CAR = {"body_style": "coupe", "body_color": "#E10600", "body_dark": "#5A0300", "uid": "hero"}
GALLERY_CARS = [
    {"body_style": "sedan", "body_color": "#F2F2F2", "body_dark": "#9A9A9A", "uid": "g1",
     "caption": "Sedan Segment", "note": "Balanced daily-driver value retention"},
    {"body_style": "suv", "body_color": "#1A1A1A", "body_dark": "#000000", "uid": "g2",
     "caption": "SUV Segment", "note": "Higher resale floor, slower depreciation"},
    {"body_style": "hatchback", "body_color": "#E10600", "body_dark": "#5A0300", "uid": "g3",
     "caption": "Hatchback Segment", "note": "Fastest-moving, most liquid resale market"},
    {"body_style": "coupe", "body_color": "#C9C9C9", "body_dark": "#6E6E6E", "uid": "g4",
     "caption": "Coupe Segment", "note": "Lower volume, higher price variance"},
]

# ==============================================================================
# Design tokens + global CSS — black / white / red racing theme
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

:root{
    --bg-deep:#0A0A0A;
    --bg-panel:#141414;
    --bg-panel-alt:#1B1B1B;
    --border:#2A2A2A;
    --border-soft:#1E1E1E;
    --text:#FFFFFF;
    --text-muted:#9A9A9A;
    --text-faint:#5C5C5C;
    --red:#E10600;
    --red-bright:#FF2A2A;
    --emerald:#34D399;
    --rose:#FB7185;
    --chrome-1:#8A8A8A;
    --chrome-2:#F2F2F2;
    --chrome-3:#5C5C5C;
    --glass-bg: rgba(20,20,20,0.58);
    --glass-border: rgba(255,255,255,0.07);
}

html, body, .stApp { background: var(--bg-deep) !important; }
* { font-family: 'Inter', -apple-system, sans-serif; }

#MainMenu, footer, [data-testid="stHeader"] { visibility: hidden; height: 0; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding-top: 0; padding-bottom: 4rem; max-width: 1320px; }

/* ---------- Hero ---------- */
.hero {
    position: relative; min-height: 560px; margin: 0 -1rem 0 -1rem;
    display: flex; align-items: center; overflow: hidden;
    border-bottom: 1px solid var(--glass-border);
    background:
        radial-gradient(ellipse at 78% 45%, #1D1414 0%, #0A0A0A 68%),
        linear-gradient(180deg, #0C0C0C 0%, #060606 100%);
}
.hero::after {
    content: ''; position: absolute; left: 0; right: 0; bottom: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--red) 20%, var(--red) 80%, transparent);
}
.hero-car {
    position: absolute; right: -2%; top: 50%; transform: translateY(-42%);
    opacity: 0.95; z-index: 1; filter: drop-shadow(0 30px 60px rgba(0,0,0,0.6)) drop-shadow(0 0 40px rgba(225,6,0,0.12));
}
.hero-content { position: relative; z-index: 2; padding: 3rem 3.6rem; width: 58%; }
.hero-eyebrow {
    font-family: 'Oswald', sans-serif; font-size: 0.75rem; letter-spacing: 0.38em;
    text-transform: uppercase; font-weight: 600; margin-bottom: 1rem;
    background: linear-gradient(90deg, var(--chrome-2), var(--chrome-1));
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-title {
    font-family: 'Oswald', sans-serif; font-size: 6rem; font-weight: 700; color: #FFFFFF;
    letter-spacing: 0.01em; line-height: 0.95; text-transform: uppercase; margin: 0;
    text-shadow: 0 4px 30px rgba(0,0,0,0.5);
}
.hero-title span {
    background: linear-gradient(160deg, var(--red-bright) 0%, var(--red) 55%, #6B0000 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub {
    font-size: 1.05rem; color: #C7C7C7; max-width: 520px; margin-top: 1.2rem; line-height: 1.7;
    font-weight: 400;
}
.hero-divider {
    width: 56px; height: 2px; margin: 1.6rem 0 0 0;
    background: linear-gradient(90deg, var(--red), transparent);
}
.hero-stats { display: flex; gap: 3rem; margin-top: 1.8rem; }
.hero-stat-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 700; color: #FFFFFF; }
.hero-stat-label { font-size: 0.64rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--text-muted); margin-top: 0.2rem; }
.hero-stat-item { padding-left: 0.9rem; border-left: 2px solid var(--red); }

/* ---------- Section headers ---------- */
.chapter {
    display: flex; align-items: center; gap: 1.1rem; margin: 4rem 0 1.6rem 0;
    padding-top: 2rem; position: relative;
}
.chapter::before {
    content: ''; position: absolute; top: 0; left: 0; width: 64px; height: 1px;
    background: linear-gradient(90deg, var(--red), transparent);
}
.chapter-num {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 700;
    width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    border: 1px solid var(--glass-border);
    background: linear-gradient(160deg, var(--chrome-2), var(--chrome-3));
    -webkit-background-clip: text; background-clip: text; color: transparent;
    -webkit-text-fill-color: transparent;
}
.chapter-title {
    font-family: 'Oswald', sans-serif; font-size: 1.9rem; font-weight: 700; color: #FFFFFF;
    text-transform: uppercase; letter-spacing: 0.03em;
}
.chapter-sub { font-size: 0.88rem; color: var(--text-muted); margin: -0.9rem 0 1.6rem 3.1rem; font-style: italic; }

/* ---------- Bordered containers as premium glass cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] { background: transparent; }
div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {
    background: var(--glass-bg); border-radius: 10px;
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div) {
    border: 1px solid var(--glass-border) !important; border-radius: 10px !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.35);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div):hover {
    border-color: rgba(225,6,0,0.35) !important;
    box-shadow: 0 16px 48px rgba(0,0,0,0.45), 0 0 0 1px rgba(225,6,0,0.08);
}

/* ---------- Result number ---------- */
.result-label { font-size: 0.7rem; letter-spacing: 0.16em; text-transform: uppercase; color: var(--text-muted); font-weight: 700; }
.result-figure {
    font-family: 'IBM Plex Mono', monospace; font-weight: 700;
    font-size: 3.2rem; color: var(--text); line-height: 1.15; margin: 0.4rem 0 0.25rem 0;
    letter-spacing: -0.01em;
}
.result-range { font-family: 'IBM Plex Mono', monospace; font-size: 0.84rem; color: var(--red-bright); }

.band-wrap { margin-top: 1.8rem; }
.band-labels { display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; color: var(--text-faint); margin-top: 0.5rem; }

.chip-row { display: flex; gap: 1.6rem; flex-wrap: wrap; }
.chip { flex: 1; min-width: 130px; border-left: 2px solid var(--border); padding-left: 0.9rem; transition: border-color 0.2s ease; }
.chip-label { font-size: 0.66rem; letter-spacing: 0.09em; text-transform: uppercase; color: var(--text-muted); font-weight: 600; }
.chip-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.35rem; font-weight: 700; color: var(--text); margin: 0.18rem 0; }
.chip-delta { font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; font-weight: 600; }
.chip-delta.up { color: var(--emerald); }
.chip-delta.down { color: var(--rose); }

.tag-row { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 1rem; }
.tag {
    background: rgba(255,255,255,0.03); border: 1px solid var(--border); color: var(--text);
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
    padding: 0.36rem 0.8rem; border-radius: 4px; backdrop-filter: blur(6px);
}
.tag.red { border-color: var(--red); color: var(--red-bright); }

/* ---------- Gallery ---------- */
.gallery-card {
    position: relative; border-radius: 8px; overflow: hidden; border: 1px solid var(--glass-border);
    height: 210px; background: radial-gradient(ellipse at 60% 45%, #1B1414 0%, #0C0C0C 80%);
    display: flex; align-items: center; justify-content: center;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    box-shadow: 0 8px 28px rgba(0,0,0,0.3);
}
.gallery-card:hover {
    transform: translateY(-4px); border-color: rgba(225,6,0,0.4);
    box-shadow: 0 16px 40px rgba(0,0,0,0.5), 0 0 24px rgba(225,6,0,0.10);
}
.gallery-car-wrap { width: 92%; }
.gallery-caption {
    position: absolute; bottom: 0; left: 0; right: 0; padding: 0.8rem 1rem;
    background: linear-gradient(180deg, transparent, rgba(0,0,0,0.92));
    font-family: 'Oswald', sans-serif; font-size: 0.82rem; font-weight: 600; color: #FFFFFF;
    text-transform: uppercase; letter-spacing: 0.02em;
}

/* ---------- Widgets ---------- */
[data-testid="stWidgetLabel"] p {
    font-size: 0.72rem !important; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--text-muted) !important; font-weight: 700 !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    background: var(--bg-panel-alt); border-color: var(--border); border-radius: 4px;
}
.stSlider [data-baseweb="slider"] { margin-top: 0.4rem; }
.stSlider [role="slider"] { background-color: var(--red) !important; border-color: var(--red) !important; }

[data-testid="stSegmentedControl"] label {
    background: var(--bg-panel-alt) !important; border: 1px solid var(--border) !important;
    color: var(--text-muted) !important; border-radius: 4px !important;
}
[data-testid="stSegmentedControl"] label[data-checked="true"] {
    background: var(--red) !important; color: #FFFFFF !important; border-color: var(--red) !important;
}

div.stButton > button {
    background: var(--red); color: #FFFFFF; border: none; border-radius: 4px;
    padding: 0.75rem 1.4rem; font-family: 'Oswald', sans-serif; font-weight: 700; font-size: 0.85rem;
    letter-spacing: 0.08em; text-transform: uppercase; width: 100%;
    transition: all 0.15s ease;
}
div.stButton > button:hover { background: var(--red-bright); box-shadow: 0 8px 26px rgba(225,6,0,0.4); transform: translateY(-1px); }

div[data-testid="stDownloadButton"] > button {
    background: transparent; color: var(--red-bright); border: 1px solid var(--red);
    border-radius: 4px; padding: 0.7rem 1.4rem; font-family: 'Oswald', sans-serif; font-weight: 700; font-size: 0.8rem;
    letter-spacing: 0.08em; text-transform: uppercase; width: 100%; transition: all 0.15s ease;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: rgba(225,6,0,0.10); box-shadow: 0 4px 18px rgba(225,6,0,0.20);
}

[data-testid="stMetricValue"] { color: var(--text); font-family: 'IBM Plex Mono', monospace; }
[data-testid="stMetricLabel"] { color: var(--text-muted); }
[data-testid="stMetricDelta"] svg { display: none; }

hr { border-color: var(--border-soft); }
[data-testid="stDataFrame"] { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 4px; }
::-webkit-scrollbar { height: 8px; width: 8px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

.credit-line { font-size: 0.68rem; color: var(--text-faint); font-family: 'IBM Plex Mono', monospace; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

DARK_LAYOUT = dict(
    plot_bgcolor="#141414",
    paper_bgcolor="#141414",
    font=dict(color="#9A9A9A", family="IBM Plex Mono"),
    xaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#2A2A2A"),
    yaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#2A2A2A"),
    margin=dict(l=10, r=10, t=10, b=10),
)
RED_SCALE = ["#3A0000", "#6B0000", "#A30000", "#E10600", "#FF6B60"]


def chapter(number, title, subtitle=None):
    st.markdown(f"""
    <div class="chapter">
        <span class="chapter-num">{number}</span>
        <span class="chapter-title">{title}</span>
    </div>
    """, unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="chapter-sub">{subtitle}</div>', unsafe_allow_html=True)


# ==============================================================================
# Load model + data
# ==============================================================================
@st.cache_resource
def load_model():
    """Point-estimate model, quantile models (P10/P50/P90), and training metrics."""
    return load_models()


@st.cache_data
def load_data():
    return get_or_build_clean_dataset()


model, quantile_models, metrics = load_model()
df = load_data()
BRANDS = sorted(df["brand"].unique().tolist())


def format_inr(value):
    return f"Rs {value:,.0f}"


def build_pdf_report(profile: dict, prediction: float, low: float, high: float,
                      band_lo: float, band_p50: float, band_hi: float) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                             leftMargin=0.75 * inch, rightMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    red = colors.HexColor("#B00500")
    black = colors.HexColor("#0A0A0A")
    muted = colors.HexColor("#5C5C5C")

    title_style = ParagraphStyle("TitleRed", parent=styles["Title"], textColor=black, fontSize=22, spaceAfter=2)
    tagline_style = ParagraphStyle("Tagline", parent=styles["Normal"], textColor=muted, fontSize=9,
                                    spaceAfter=18, alignment=TA_CENTER)
    h2_style = ParagraphStyle("H2Red", parent=styles["Heading2"], textColor=red, fontSize=12, spaceBefore=14, spaceAfter=6)
    figure_style = ParagraphStyle("Figure", parent=styles["Normal"], fontSize=26, textColor=black,
                                   alignment=TA_CENTER, spaceAfter=2, fontName="Helvetica-Bold")
    range_style = ParagraphStyle("Range", parent=styles["Normal"], fontSize=10, textColor=red,
                                  alignment=TA_CENTER, spaceAfter=14)

    elements = [
        Paragraph("VELOX", title_style),
        Paragraph("VEHICLE VALUATION REPORT", tagline_style),
        Paragraph("Estimated Resale Value", ParagraphStyle(
            "Label", parent=styles["Normal"], fontSize=9, textColor=muted, alignment=TA_CENTER)),
        Paragraph(f"Rs {prediction:,.0f}", figure_style),
        Paragraph(f"Likely range: Rs {low:,.0f} &nbsp;–&nbsp; Rs {high:,.0f}", range_style),
        Paragraph("Vehicle Profile", h2_style),
    ]

    profile_table_data = [[k, v] for k, v in profile.items()]
    profile_table = Table(profile_table_data, colWidths=[2.2 * inch, 3.3 * inch])
    profile_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), muted),
        ("TEXTCOLOR", (1, 0), (1, -1), black),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
    ]))
    elements.append(profile_table)

    elements.append(Paragraph("Model-Predicted Price Range (P10-P90, this exact configuration)", h2_style))
    band_data = [["P10", "Median", "P90", "This Estimate"],
                 [f"Rs {band_lo:,.0f}", f"Rs {band_p50:,.0f}", f"Rs {band_hi:,.0f}", f"Rs {prediction:,.0f}"]]
    band_table = Table(band_data, colWidths=[1.3 * inch] * 4)
    band_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, 0), muted),
        ("TEXTCOLOR", (0, 1), (-1, 1), black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#E0E0E0")),
        ("BACKGROUND", (3, 0), (3, 1), colors.HexColor("#FBEAEA")),
    ]))
    elements.append(band_table)

    elements.append(Spacer(1, 18))
    disclaimer = (
        "This estimate is generated by a Gradient Boosting model trained on real-world "
        "used car listings from CarDekho (India, 1992-2020), with prices scaled to "
        "estimated 2026 market levels using a documented ~8%/year used-car price "
        "appreciation rate. It cannot see the vehicle's actual condition, accident "
        "history, or service records, so treat this as a market benchmark, not a "
        "guaranteed sale price."
    )
    elements.append(Paragraph(disclaimer, ParagraphStyle(
        "Disclaimer", parent=styles["Normal"], fontSize=8, textColor=muted, leading=12)))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y')} · VELOX Vehicle Valuation Engine",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=7.5, textColor=muted, alignment=TA_CENTER)
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# ==============================================================================
# HERO
# ==============================================================================
st.markdown(f"""
<div class="hero">
    <div class="hero-car">{car_illustration_svg(**HERO_CAR, width="620")}</div>
    <div class="hero-content">
        <div class="hero-eyebrow">Vehicle Valuation Engine</div>
        <div class="hero-title">VEL<span>O</span>X</div>
        <div class="hero-divider"></div>
        <div class="hero-sub">
            Know what your car is really worth. VELOX benchmarks resale value against
            {len(df):,} real used-car listings, scaled to estimated {CURRENT_YEAR} market levels.
        </div>
        <div class="hero-stats">
            <div class="hero-stat-item"><div class="hero-stat-value">{len(df):,}</div><div class="hero-stat-label">Real Listings</div></div>
            <div class="hero-stat-item"><div class="hero-stat-value">{df['brand'].nunique()}</div><div class="hero-stat-label">Brands Covered</div></div>
            <div class="hero-stat-item"><div class="hero-stat-value">{metrics['r2']:.2f}</div><div class="hero-stat-label">Model R² Score</div></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# CHAPTER 01 — CONFIGURE + VALUATION
# ==============================================================================
chapter("01", "Configure Your Vehicle", "Enter the details below — the valuation updates instantly.")

col_form, col_result = st.columns([1, 1.35], gap="large")

with col_form:
    with st.container(border=True):
        brand = st.selectbox("Brand", BRANDS, index=BRANDS.index("Maruti") if "Maruti" in BRANDS else 0)
        year = st.slider("Manufacturing Year", 1992, 2026, 2021, 1)
        km_driven = st.slider("Kilometers Driven", 0, 300000, 40000, 1000)
        fuel = st.segmented_control("Fuel Type", FUEL_TYPES, default=FUEL_TYPES[0])
        transmission = st.segmented_control("Transmission", TRANSMISSIONS, default=TRANSMISSIONS[0])
        seller_type = st.segmented_control("Seller Type", SELLER_TYPES, default=SELLER_TYPES[0])
        owner = st.selectbox("Ownership", OWNER_TYPES, index=0)

fuel = fuel or FUEL_TYPES[0]
transmission = transmission or TRANSMISSIONS[0]
seller_type = seller_type or SELLER_TYPES[0]
car_age = max(CURRENT_YEAR - year, 0)

input_df = pd.DataFrame([{
    "brand": brand, "car_age": car_age, "km_driven": km_driven, "fuel": fuel,
    "seller_type": seller_type, "transmission": transmission, "owner": owner,
}])

prediction = max(model.predict(input_df)[0], 0)
q_p10 = max(float(quantile_models["p10"].predict(input_df)[0]), 0)
q_p50 = max(float(quantile_models["p50"].predict(input_df)[0]), 0)
q_p90 = max(float(quantile_models["p90"].predict(input_df)[0]), 0)
# Guard against quantile crossing (rare, but possible with independently trained models)
q_p10, q_p90 = min(q_p10, q_p90), max(q_p10, q_p90)
low, high = min(q_p10, prediction), max(q_p90, prediction)

brand_df = df[df["brand"] == brand]
brand_median = brand_df["price_inr"].median() if len(brand_df) >= 5 else df["price_inr"].median()

fuel_df = df[df["fuel"] == fuel]
fuel_median = fuel_df["price_inr"].median() if len(fuel_df) >= 5 else df["price_inr"].median()
overall_median = df["price_inr"].median()

with col_result:
    with st.container(border=True):
        st.markdown(
            f'<div class="result-label">Estimated Resale Value</div>'
            f'<div class="result-figure">{format_inr(prediction)}</div>'
            f'<div class="result-range">Range {format_inr(low)} &nbsp;–&nbsp; {format_inr(high)}</div>',
            unsafe_allow_html=True
        )

        band_lo, band_hi = float(q_p10), float(q_p90)
        span = max(band_hi - band_lo, 1)
        pos_pct = min(max((prediction - band_lo) / span * 100, 1), 99)
        median_pct = min(max((q_p50 - band_lo) / span * 100, 1), 99)

        gauge_svg = (
            '<div class="band-wrap">'
            '<svg viewBox="0 0 500 46" width="100%" height="46" xmlns="http://www.w3.org/2000/svg">'
            '<defs><linearGradient id="bandgrad" x1="0" y1="0" x2="1" y2="0">'
            '<stop offset="0%" stop-color="#3A0000"/>'
            '<stop offset="100%" stop-color="#FF6B60"/>'
            '</linearGradient></defs>'
            '<rect x="0" y="16" width="500" height="8" rx="4" fill="url(#bandgrad)" opacity="0.6"/>'
            f'<line x1="{median_pct*5}" y1="8" x2="{median_pct*5}" y2="32" stroke="#9A9A9A" stroke-width="1.5" stroke-dasharray="3,2"/>'
            f'<circle cx="{pos_pct*5}" cy="20" r="8" fill="#0A0A0A" stroke="#E10600" stroke-width="3"/>'
            '</svg>'
            '<div class="band-labels">'
            f'<span>{format_inr(band_lo)} (P10)</span>'
            f'<span style="color:#9A9A9A;">median {format_inr(q_p50)}</span>'
            f'<span>{format_inr(band_hi)} (P90)</span>'
            '</div></div>'
        )
        st.markdown(gauge_svg, unsafe_allow_html=True)
        st.caption(f"Model-predicted P10–P90 range for **this exact configuration** "
                   f"(quantile regression) — not just {brand}'s historical spread.")

        st.write("")
        tags = (
            '<div class="tag-row">'
            f'<span class="tag red">{brand}</span>'
            f'<span class="tag red">{year}</span>'
            f'<span class="tag">{km_driven:,} km</span>'
            f'<span class="tag">{fuel}</span>'
            f'<span class="tag">{transmission}</span>'
            f'<span class="tag">{seller_type}</span>'
            '</div>'
        )
        st.markdown(tags, unsafe_allow_html=True)

    st.write("")
    with st.container(border=True):
        st.markdown("**How This Vehicle Compares**")

        def chip(label, value, delta):
            cls = "up" if delta >= 0 else "down"
            arrow = "▲" if delta >= 0 else "▼"
            return (
                '<div class="chip">'
                f'<div class="chip-label">{label}</div>'
                f'<div class="chip-value">{format_inr(value)}</div>'
                f'<div class="chip-delta {cls}">{arrow} {format_inr(abs(delta))}</div>'
                '</div>'
            )

        chips_html = '<div class="chip-row">' + \
            chip(f"{brand} median", brand_median, prediction - brand_median) + \
            chip(f"{fuel} median", fuel_median, prediction - fuel_median) + \
            chip("Overall median", overall_median, prediction - overall_median) + \
            '</div>'
        st.markdown(chips_html, unsafe_allow_html=True)

    st.write("")
    with st.container(border=True):
        st.markdown("**Why This Estimate**")
        st.caption("SHAP contribution of each field to this specific prediction — how much it pushed the "
                   "price up or down relative to the model's average baseline.")

        contrib = explain_prediction(model, input_df)
        colors_shap = ["#E10600" if v >= 0 else "#4A5568" for v in contrib["shap_value"]]
        fig_shap = go.Figure(go.Bar(
            x=contrib["shap_value"], y=contrib["label"], orientation="h",
            marker_color=colors_shap,
            text=[f"{'+' if v >= 0 else ''}{format_inr(v)}" for v in contrib["shap_value"]],
            textposition="outside", textfont=dict(color="#FFFFFF", family="IBM Plex Mono", size=10)
        ))
        fig_shap.update_layout(**DARK_LAYOUT, height=240, xaxis_title="Impact on Price (Rs)", yaxis_title="")
        st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})

st.write("")
predict_clicked = st.button("Estimate Resale Value", use_container_width=True)

profile_dict = {
    "Brand": brand, "Manufacturing Year": str(year), "Kilometers Driven": f"{km_driven:,} km",
    "Fuel Type": fuel, "Transmission": transmission, "Seller Type": seller_type, "Ownership": owner,
}
pdf_bytes = build_pdf_report(profile_dict, prediction, low, high, float(q_p10), float(q_p50), float(q_p90))
st.download_button(
    label="Download Valuation Report (PDF)", data=pdf_bytes,
    file_name=f"VELOX_Report_{brand.replace(' ', '_')}_{year}.pdf",
    mime="application/pdf", use_container_width=True,
)

# ==============================================================================
# CHAPTER 02 — DEPRECIATION STORY
# ==============================================================================
chapter("02", "The Depreciation Story", f"How a {brand} in this configuration loses value over its lifetime.")

with st.container(border=True):
    age_points = [0, 2, 4, 6, 8, 10, 15, 20, 25]
    dep_rows = []
    for a in age_points:
        row = pd.DataFrame([{
            "brand": brand, "car_age": a, "km_driven": km_driven, "fuel": fuel,
            "seller_type": seller_type, "transmission": transmission, "owner": owner,
        }])
        dep_rows.append({"Age": a, "Predicted Price": max(model.predict(row)[0], 0)})
    dep_df = pd.DataFrame(dep_rows)

    fig_dep = go.Figure()
    fig_dep.add_trace(go.Scatter(
        x=dep_df["Age"], y=dep_df["Predicted Price"], mode="lines+markers",
        line=dict(color="#E10600", width=3), marker=dict(color="#FF6B60", size=8),
        fill="tozeroy", fillcolor="rgba(225,6,0,0.08)"
    ))
    fig_dep.add_vline(x=car_age, line_width=2, line_dash="dash", line_color="#FFFFFF",
                       annotation_text="This vehicle", annotation_font_color="#FFFFFF", annotation_font_size=10)
    fig_dep.update_layout(**DARK_LAYOUT, height=280,
                           xaxis_title="Vehicle Age (years)", yaxis_title="Predicted Price (Rs)")
    st.plotly_chart(fig_dep, use_container_width=True, config={"displayModeBar": False})

    value_at_0 = dep_df["Predicted Price"].iloc[0]
    value_at_10 = dep_df[dep_df["Age"] == 10]["Predicted Price"].iloc[0]
    drop_pct = (value_at_0 - value_at_10) / value_at_0 * 100 if value_at_0 else 0
    st.caption(f"This {brand} model is projected to lose **{drop_pct:,.0f}%** of its value in the first 10 years.")

# ==============================================================================
# CHAPTER 03 — SHOWROOM GALLERY (real photos + market data interleaved)
# ==============================================================================
chapter("03", "Showroom Gallery", "A look across body styles in the market, with the data behind each segment.")

gcols = st.columns(4)
for gcol, car in zip(gcols, GALLERY_CARS):
    with gcol:
        svg = car_illustration_svg(
            body_style=car["body_style"], body_color=car["body_color"],
            body_dark=car["body_dark"], uid=car["uid"], width="100%"
        )
        st.markdown(f"""
        <div class="gallery-card">
            <div class="gallery-car-wrap">{svg}</div>
            <div class="gallery-caption">{car['caption']}</div>
        </div>
        <div class="credit-line">{car['note']}</div>
        """, unsafe_allow_html=True)

st.write("")
col1, col2 = st.columns(2, gap="large")
with col1:
    with st.container(border=True):
        st.markdown("**Price by Brand**")
        brand_order = df.groupby("brand")["price_inr"].median().sort_values(ascending=False).index
        fig1 = px.box(df, x="price_inr", y="brand", category_orders={"brand": list(brand_order)},
                       color_discrete_sequence=["#E10600"])
        fig1.update_layout(**DARK_LAYOUT, height=360, xaxis_title="Price (Rs)", yaxis_title="")
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

with col2:
    with st.container(border=True):
        st.markdown("**Price by Fuel Type**")
        fig2 = px.box(df, x="fuel", y="price_inr", color="fuel", color_discrete_sequence=RED_SCALE)
        fig2.update_layout(**DARK_LAYOUT, height=360, showlegend=False, xaxis_title="", yaxis_title="Price (Rs)")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

col3, col4 = st.columns(2, gap="large")
with col3:
    with st.container(border=True):
        st.markdown("**Price vs Vehicle Age**")
        age_trend = df.groupby("car_age")["price_inr"].median().reset_index()
        fig3 = px.line(age_trend, x="car_age", y="price_inr", markers=True, color_discrete_sequence=["#E10600"])
        fig3.update_traces(line_width=3, marker_size=6)
        fig3.update_layout(**DARK_LAYOUT, height=360, xaxis_title="Vehicle Age (years)", yaxis_title="Median Price (Rs)")
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

with col4:
    with st.container(border=True):
        st.markdown("**Price vs Kilometers Driven**")
        fig4 = px.scatter(df, x="km_driven", y="price_inr", color="transmission",
                           opacity=0.6, color_discrete_sequence=["#E10600", "#9A9A9A"])
        fig4.update_layout(**DARK_LAYOUT, height=360, xaxis_title="Kilometers Driven", yaxis_title="Price (Rs)")
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

with st.container(border=True):
    st.markdown("**Raw Data Sample**")
    st.dataframe(df.sample(20, random_state=1).reset_index(drop=True), use_container_width=True)

# ==============================================================================
# CHAPTER 04 — UNDER THE HOOD
# ==============================================================================
chapter("04", "Under the Hood", "How the model works, and what it's honest about.")

col_a, col_b = st.columns([1.2, 1], gap="large")
with col_a:
    with st.container(border=True):
        st.markdown("**About VELOX**")
        st.markdown(textwrap.dedent("""\
        VELOX estimates **used car resale value** from brand, age, kilometers
        driven, fuel type, transmission, seller type, and ownership history,
        using a **Gradient Boosting Regressor** trained on the real
        [Car Details from Car Dekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
        dataset (4,340 listings, India, 1992–2020).
        """))
        st.write("")
        st.markdown("**Data preparation**")
        st.markdown(textwrap.dedent(f"""\
        - Brand extracted from the listing name, reduced to the 12 most frequent + "Other"
        - Vehicle age derived from manufacturing year (relative to {CURRENT_YEAR})
        - Top/bottom 0.5% of prices dropped as implausible outliers
        - Prices are in Indian Rupees (Rs), as listed on CarDekho
        """))

    st.write("")
    with st.container(border=True):
        st.markdown("**A Note on 2026 Pricing**")
        if metrics and "market_adjustment" in metrics:
            pct = (metrics["market_adjustment"] - 1) * 100
            st.markdown(textwrap.dedent(f"""\
            CarDekho's public listings reflect the India used-car market as of
            roughly {metrics['data_reference_year']} — genuinely fresh 2026 listings
            aren't available as an open dataset. Rather than show stale {metrics['data_reference_year']}
            prices, every price here is scaled up **{pct:,.0f}%** to estimated
            {metrics['market_year']} market levels, using a documented ~{metrics['annual_appreciation']*100:.0f}%/year
            used-car price appreciation rate for India (Cars24/Team-BHP and Mordor
            Intelligence market reports, 2024–2025).

            This is a transparent **estimate**, not real 2026 transaction data —
            treat it the same way you'd treat an inflation-adjusted historical
            price, not a live market quote.
            """))
        else:
            st.caption("Market adjustment details unavailable — run train_model.py.")

    st.write("")
    with st.container(border=True):
        st.markdown("**A Note on Accuracy**")
        r2_text = f"{metrics['r2']:.2f}" if metrics else "unavailable"
        st.markdown(textwrap.dedent(f"""\
        This model's R² is **{r2_text}** — solid for a real-world pricing problem,
        but not perfect. Brand, age, mileage, and fuel type explain most of the
        variance in used-car prices, but this model still can't see a vehicle's
        actual condition, accident history, service records, or negotiation
        dynamics. Treat estimates as a **market benchmark**, not a guaranteed
        sale price.
        """))

with col_b:
    with st.container(border=True):
        st.markdown("**Model Performance**")
        if metrics:
            st.metric("Test R² Score", f"{metrics['r2']:.3f}")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("5-Fold CV R²", f"{metrics.get('cv_r2', 0):.3f}")
            with m2:
                coverage = metrics.get("coverage", {}).get("p10_p90_coverage")
                st.metric("P10–P90 Coverage", f"{coverage:.0%}" if coverage else "—")
            st.metric("Mean Absolute Error", format_inr(metrics["mae"]))
            st.metric("RMSE", format_inr(metrics["rmse"]))
            st.caption("Point-estimate model tuned via RandomizedSearchCV (25 iterations, "
                       "5-fold CV) on training data only; all metrics above are on the held-out "
                       "test split. P10–P90 coverage is the share of true test-set prices that "
                       "actually fall inside the model's predicted range — ~80% is well-calibrated.")
        else:
            st.caption("Metrics unavailable — run train_model.py.")

    st.write("")
    with st.container(border=True):
        st.markdown("**Tech Stack**")
        stack_tags = "".join(f'<span class="tag">{t}</span>' for t in
                              ["scikit-learn", "SHAP", "FastAPI", "Pandas", "Streamlit", "Plotly", "ReportLab"])
        st.markdown(f'<div class="tag-row">{stack_tags}</div>', unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("**What Drives the Prediction — Overall**")
    st.caption("Global feature importance across all training data, vs. the per-vehicle "
               "SHAP breakdown shown in \"Why This Estimate\" above.")
    try:
        imp_df = global_feature_importance(model)
        fig_imp = go.Figure(go.Bar(
            x=imp_df["importance"], y=imp_df["label"], orientation="h", marker_color="#E10600",
            text=[f"{v*100:.1f}%" for v in imp_df["importance"]], textposition="outside",
            textfont=dict(color="#FFFFFF", family="IBM Plex Mono", size=11)
        ))
        fig_imp.update_layout(**DARK_LAYOUT, height=280, xaxis_title="Relative Importance", yaxis_title="")
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        st.caption("Feature importance unavailable for this model configuration.")

# ==============================================================================
# Footer
# ==============================================================================
st.write("")
st.markdown("""
<div style="border-top: 1px solid #1E1E1E; margin-top: 2rem; padding-top: 1.2rem;">
    <div style="display: flex; justify-content: space-between; align-items: center;
                flex-wrap: wrap; gap: 0.6rem;">
        <div style="font-size: 0.72rem; color: #5C5C5C;">
            Developed by <span style="color:#9A9A9A; font-weight:600;">Hamna Munir</span>
            &nbsp;·&nbsp; Software Engineering &amp; AI/ML
        </div>
        <div style="font-size: 0.68rem; color: #5C5C5C; letter-spacing: 0.04em;">
            VELOX &nbsp;·&nbsp; BUILT WITH STREAMLIT + SCIKIT-LEARN
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
