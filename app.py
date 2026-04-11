"""
app.py — Streamlit UI for the Contract Intelligence Engine.
Design: Premium legal-tech — deep navy, warm gold, sharp typographic hierarchy.
"""

import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="Lexis — Contract Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.extractor import ContractExtractor
from src.joint_extractor import JointContractExtractor
from src.config import OPENAI_API_KEY


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

        :root {
            --navy:       #0b1220;
            --navy-mid:   #111c30;
            --navy-card:  #162035;
            --navy-light: #1e2d47;
            --navy-hover: #243454;
            --gold:       #c9a84c;
            --gold-light: #e8c97a;
            --gold-pale:  #f5ecd4;
            --gold-dim:   #6b5520;
            --border:     #243050;
            --border-mid: #2e3e5c;
            --border-gold:#5a4010;
            --text:       #edf0f5;
            --text-sub:   #8fa0bc;
            --text-muted: #4a5e7a;
            --green:      #3ecf8e;
            --green-dim:  #0e3d28;
            --red:        #f06565;
            --red-dim:    #3d1010;
            --blue:       #6eb3f7;
            --serif:      'Playfair Display', Georgia, serif;
            --sans:       'DM Sans', system-ui, sans-serif;
            --mono:       'DM Mono', 'Courier New', monospace;
        }

        html, body, [class*="css"], .stApp {
            font-family: var(--sans) !important;
            background-color: var(--navy) !important;
            color: var(--text) !important;
        }
        .main .block-container {
            background-color: var(--navy) !important;
            padding-top: 0 !important;
            padding-bottom: 3rem !important;
            max-width: 1340px !important;
        }
        #MainMenu, footer, header { visibility: hidden; }
        section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

        /* SIDEBAR */
        [data-testid="stSidebar"] {
            background: var(--navy-mid) !important;
            border-right: 1px solid var(--border) !important;
        }
        [data-testid="stSidebar"] * { font-family: var(--sans) !important; }
        [data-testid="stSidebar"] .stMarkdown h3 {
            font-family: var(--serif) !important;
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            color: var(--gold) !important;
            letter-spacing: 0.04em !important;
            margin: 20px 0 8px !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: var(--border) !important;
            margin: 10px 0 !important;
        }

        .sb-brand {
            padding: 28px 20px 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 4px;
        }
        .sb-brand-name {
            font-family: var(--serif);
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--gold);
            letter-spacing: 0.02em;
        }
        .sb-brand-sub {
            font-size: 0.68rem;
            color: var(--text-muted);
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-top: 3px;
        }

        [data-testid="stRadio"] label,
        [data-testid="stToggle"] label {
            font-size: 0.82rem !important;
            color: var(--text-sub) !important;
        }

        /* TOPBAR */
        .cie-topbar {
            background: linear-gradient(135deg, var(--navy-mid) 0%, var(--navy-card) 100%);
            border-bottom: 1px solid var(--border-gold);
            padding: 28px 40px 24px;
            margin-bottom: 36px;
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
        }
        .cie-logo-group { display: flex; align-items: center; gap: 16px; }
        .cie-seal {
            width: 50px; height: 50px;
            background: var(--gold-dim);
            border: 1.5px solid var(--gold);
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.4rem;
            flex-shrink: 0;
        }
        .cie-wordmark {
            font-family: var(--serif);
            font-size: 1.9rem;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -0.01em;
            line-height: 1;
        }
        .cie-wordmark span { color: var(--gold); }
        .cie-tagline {
            font-size: 0.7rem;
            color: var(--text-muted);
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-top: 5px;
        }
        .cie-topbar-meta {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .cie-badge {
            font-family: var(--mono);
            font-size: 0.66rem;
            color: var(--gold);
            background: var(--gold-dim);
            border: 1px solid var(--border-gold);
            padding: 4px 12px;
            border-radius: 3px;
            letter-spacing: 0.04em;
        }
        .cie-status {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .cie-dot {
            width: 6px; height: 6px;
            border-radius: 50%;
            background: var(--green);
        }
        .cie-status-txt {
            font-size: 0.68rem;
            color: var(--text-muted);
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        /* SECTION HEADER */
        .section-header {
            display: flex;
            align-items: center;
            gap: 14px;
            margin: 36px 0 18px;
        }
        .section-num {
            font-family: var(--mono);
            font-size: 0.62rem;
            color: var(--gold);
            background: var(--gold-dim);
            border: 1px solid var(--border-gold);
            padding: 2px 8px;
            border-radius: 2px;
            letter-spacing: 0.06em;
        }
        .section-title {
            font-family: var(--serif);
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--text);
            margin: 0;
        }
        .section-rule {
            flex: 1;
            height: 1px;
            background: var(--border);
        }

        /* UPLOAD */
        [data-testid="stFileUploader"] {
            background: var(--navy-card) !important;
            border: 1.5px dashed var(--border-mid) !important;
            border-radius: 8px !important;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: var(--gold-dim) !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: transparent !important;
            padding: 28px !important;
        }
        [data-testid="stFileUploaderDropzone"] > div {
            color: var(--text-sub) !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
            background: var(--gold) !important;
            color: var(--navy) !important;
            border: none !important;
            border-radius: 5px !important;
            font-family: var(--sans) !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
        }

        /* FILE INFO CARD */
        .file-info-card {
            background: var(--navy-card);
            border: 1px solid var(--border);
            border-left: 3px solid var(--gold);
            border-radius: 6px;
            padding: 14px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 14px;
        }
        .file-name {
            font-size: 0.9rem;
            color: var(--text);
            font-weight: 500;
        }
        .file-meta {
            font-family: var(--mono);
            font-size: 0.66rem;
            color: var(--text-muted);
            margin-top: 3px;
        }
        .mode-pill {
            font-family: var(--mono);
            font-size: 0.62rem;
            padding: 3px 10px;
            border-radius: 3px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .mode-joint {
            color: var(--green);
            background: var(--green-dim);
            border: 1px solid rgba(62,207,142,0.25);
        }
        .mode-individual {
            color: var(--blue);
            background: rgba(110,179,247,0.08);
            border: 1px solid rgba(110,179,247,0.25);
        }

        /* BUTTONS */
        .stButton > button {
            background: var(--gold) !important;
            color: var(--navy) !important;
            border: none !important;
            font-family: var(--sans) !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            padding: 10px 28px !important;
            border-radius: 5px !important;
            width: 100% !important;
            letter-spacing: 0.02em !important;
            transition: background 0.2s !important;
        }
        .stButton > button:hover { background: var(--gold-light) !important; }
        .stButton > button:disabled { opacity: 0.35 !important; }
        .stDownloadButton > button {
            background: transparent !important;
            color: var(--gold) !important;
            border: 1px solid var(--border-gold) !important;
            font-family: var(--sans) !important;
            font-weight: 500 !important;
            font-size: 0.82rem !important;
            padding: 8px 20px !important;
            border-radius: 5px !important;
        }
        .stDownloadButton > button:hover { background: var(--gold-dim) !important; }

        /* PROGRESS */
        .stProgress > div > div { background: var(--gold) !important; }

        /* STATS ROW */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin: 0 0 32px;
        }
        .stat-card {
            background: var(--navy-card);
            border: 1px solid var(--border);
            border-top: 2px solid var(--border-mid);
            border-radius: 8px;
            padding: 18px 20px;
        }
        .stat-card.hl { border-top-color: var(--gold); }
        .stat-value {
            font-family: var(--serif);
            font-size: 2rem;
            font-weight: 700;
            color: var(--text);
            line-height: 1;
            margin-bottom: 6px;
        }
        .stat-value.gold { color: var(--gold); }
        .stat-sup {
            font-family: var(--sans);
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-left: 2px;
        }
        .stat-label {
            font-size: 0.68rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        /* CONTRACT TYPE */
        .ct-card {
            background: var(--navy-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 24px 28px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
            margin-bottom: 12px;
        }
        .ct-value {
            font-family: var(--serif);
            font-size: 1.6rem;
            font-weight: 600;
            color: var(--text);
        }
        .ct-badge {
            font-family: var(--mono);
            font-size: 0.62rem;
            color: var(--gold);
            background: var(--gold-dim);
            border: 1px solid var(--border-gold);
            padding: 4px 12px;
            border-radius: 3px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        /* CONFIDENCE BAR */
        .conf-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 10px;
        }
        .conf-label {
            font-family: var(--mono);
            font-size: 0.6rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            min-width: 76px;
        }
        .conf-track {
            flex: 1;
            height: 3px;
            background: var(--border);
            border-radius: 2px;
            overflow: hidden;
        }
        .conf-fill { height: 100%; border-radius: 2px; }
        .conf-pct {
            font-family: var(--mono);
            font-size: 0.66rem;
            min-width: 32px;
            text-align: right;
        }

        /* CLAUSE CARD */
        .clause-card {
            background: var(--navy-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px 22px;
            margin-bottom: 14px;
            transition: border-color 0.15s, background 0.15s;
        }
        .clause-card:hover { border-color: var(--border-mid); background: var(--navy-hover); }
        .clause-card.validated   { border-top: 2px solid var(--green); }
        .clause-card.not-validated { border-top: 2px solid var(--red); }
        .clause-card.null-clause { border-top: 2px solid var(--border-mid); opacity: 0.7; }
        .clause-hdr {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }
        .clause-name {
            font-family: var(--mono);
            font-size: 0.62rem;
            font-weight: 500;
            color: var(--text-muted);
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        .clause-badge {
            font-family: var(--mono);
            font-size: 0.58rem;
            padding: 2px 8px;
            border-radius: 3px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .b-validated {
            color: var(--green); background: var(--green-dim);
            border: 1px solid rgba(62,207,142,0.2);
        }
        .b-not-validated {
            color: var(--red); background: var(--red-dim);
            border: 1px solid rgba(240,101,101,0.2);
        }
        .b-null {
            color: var(--text-muted);
            background: rgba(255,255,255,0.04);
            border: 1px solid var(--border);
        }
        .clause-val {
            font-size: 0.88rem;
            color: var(--text);
            line-height: 1.65;
            margin-bottom: 12px;
        }
        .clause-null {
            font-style: italic;
            color: var(--text-muted);
            font-size: 0.84rem;
        }
        .clause-meta {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-top: 10px;
        }
        .meta-k {
            font-family: var(--mono);
            font-size: 0.58rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-right: 4px;
        }
        .meta-v {
            font-family: var(--mono);
            font-size: 0.66rem;
            color: var(--blue);
        }
        .source-lbl {
            font-family: var(--mono);
            font-size: 0.58rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 12px 0 5px;
        }
        .source-block {
            background: var(--navy-light);
            border: 1px solid var(--border);
            border-left: 2px solid var(--gold-dim);
            border-radius: 4px;
            padding: 10px 14px;
            font-family: var(--mono);
            font-size: 0.72rem;
            color: var(--text-sub);
            line-height: 1.7;
            white-space: pre-wrap;
            word-break: break-word;
        }

        /* FIELD CARDS */
        .field-card {
            background: var(--navy-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .field-label {
            font-family: var(--mono);
            font-size: 0.58rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 8px;
        }
        .field-val {
            font-size: 0.9rem;
            color: var(--text);
            font-weight: 500;
        }
        .field-null {
            font-style: italic;
            color: var(--text-muted);
            font-size: 0.84rem;
        }

        /* EFFICIENCY PANEL */
        .eff-panel {
            background: var(--navy-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 24px 28px;
        }
        .eff-title {
            font-family: var(--serif);
            font-size: 1rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 20px;
        }
        .eff-metrics {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 24px;
        }
        .eff-metric {
            background: var(--navy-light);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 14px 18px;
            min-width: 110px;
        }
        .eff-m-lbl {
            font-family: var(--mono);
            font-size: 0.58rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }
        .eff-m-val {
            font-family: var(--serif);
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text);
            line-height: 1;
        }
        .eff-m-val.green { color: var(--green); }
        .eff-m-val.gold  { color: var(--gold); }

        .compare-tbl {
            width: 100%;
            border-collapse: collapse;
            font-family: var(--mono);
            font-size: 0.72rem;
        }
        .compare-tbl th {
            background: var(--navy-light);
            color: var(--text-muted);
            font-size: 0.58rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            padding: 9px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        .compare-tbl td {
            padding: 8px 14px;
            border-bottom: 1px solid var(--border);
            color: var(--text-sub);
        }
        .compare-tbl tr:last-child td {
            border-bottom: none;
            font-weight: 600;
            color: var(--text);
        }
        .c-red { color: var(--red); }
        .c-grn { color: var(--green); }

        .call-log-wrap {
            background: var(--navy-light);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px 14px;
        }
        .call-log-row {
            display: flex;
            align-items: baseline;
            gap: 10px;
            padding: 4px 0;
            border-bottom: 1px solid var(--border);
            font-family: var(--mono);
            font-size: 0.68rem;
            color: var(--text-sub);
        }
        .call-log-row:last-child { border-bottom: none; }
        .call-log-n {
            font-size: 0.56rem;
            color: var(--text-muted);
            min-width: 18px;
        }

        /* CHUNK */
        .chunk-item {
            background: var(--navy-light);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px 14px;
            margin-bottom: 8px;
        }
        .chunk-hdr { display: flex; justify-content: space-between; margin-bottom: 6px; }
        .chunk-lbl {
            font-family: var(--mono); font-size: 0.58rem;
            color: var(--gold); text-transform: uppercase; letter-spacing: 0.08em;
        }
        .chunk-pg { font-family: var(--mono); font-size: 0.58rem; color: var(--text-muted); }
        .chunk-txt {
            font-size: 0.75rem; color: var(--text-sub);
            line-height: 1.65; font-family: var(--mono);
        }

        /* SECTION LABELS */
        .sb-section-label {
            font-family: var(--mono);
            font-size: 0.58rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            padding-bottom: 4px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 8px;
        }

        /* PIPELINE */
        .pipe-step {
            display: flex; align-items: flex-start; gap: 10px; padding: 3px 0;
        }
        .pipe-num {
            font-family: var(--mono); font-size: 0.58rem;
            color: var(--gold-dim); min-width: 14px; margin-top: 1px;
        }
        .pipe-text {
            font-family: var(--sans); font-size: 0.76rem;
            color: var(--text-muted); line-height: 1.5;
        }
        .pipe-text.active { color: var(--gold); }

        /* SUPPORTED TYPES */
        .supported-type {
            font-size: 0.76rem; color: var(--text-muted);
            padding: 3px 0; display: flex; align-items: center; gap: 8px;
        }
        .supported-type::before {
            content: ''; display: inline-block;
            width: 4px; height: 4px;
            border-radius: 50%; background: var(--gold-dim); flex-shrink: 0;
        }

        /* EMPTY STATE */
        .empty-state { text-align: center; padding: 80px 20px 60px; }
        .empty-emblem { font-size: 3.5rem; margin-bottom: 20px; opacity: 0.12; }
        .empty-heading {
            font-family: var(--serif); font-size: 1.4rem;
            font-weight: 500; color: var(--text-sub); margin-bottom: 8px;
        }
        .empty-sub {
            font-size: 0.8rem; color: var(--text-muted); letter-spacing: 0.04em;
        }
        .empty-features {
            display: inline-flex; gap: 12px;
            flex-wrap: wrap; justify-content: center; margin-top: 32px;
        }
        .empty-feature {
            background: var(--navy-card); border: 1px solid var(--border);
            border-radius: 6px; padding: 16px 20px;
            font-size: 0.76rem; color: var(--text-muted);
            text-align: left; min-width: 170px;
        }
        .empty-feature strong {
            display: block; color: var(--text-sub);
            font-weight: 500; margin-bottom: 4px; font-size: 0.82rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def conf_color(c: float) -> str:
    if c >= 0.8: return "#3ecf8e"
    if c >= 0.5: return "#c9a84c"
    return "#f06565"


def render_confidence_bar(conf: float, label: str = "Confidence"):
    pct = int(conf * 100)
    clr = conf_color(conf)
    st.markdown(
        f"""<div class="conf-row">
          <span class="conf-label">{label}</span>
          <div class="conf-track"><div class="conf-fill" style="width:{pct}%;background:{clr};"></div></div>
          <span class="conf-pct" style="color:{clr};">{pct}%</span>
        </div>""",
        unsafe_allow_html=True,
    )


def render_section_header(title: str, num: str = ""):
    num_html = f'<span class="section-num">{num}</span>' if num else ''
    st.markdown(
        f"""<div class="section-header">
          {num_html}<span class="section-title">{title}</span>
          <div class="section-rule"></div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_contract_type(contract_type: dict):
    value = contract_type.get("value") or "Unknown"
    confidence = contract_type.get("confidence", 0.0)
    st.markdown(
        f"""<div class="ct-card">
          <div class="ct-value">{value}</div>
          <span class="ct-badge">Identified Type</span>
        </div>""",
        unsafe_allow_html=True,
    )
    render_confidence_bar(confidence)


def render_clause_card(clause_name: str, clause: dict):
    labels = {
        "governing_law": "Governing Law",
        "audit_rights": "Audit Rights",
        "non_compete": "Non-Compete",
        "non_solicitation": "Non-Solicitation",
    }
    label = labels.get(clause_name, clause_name.replace("_", " ").title())
    value      = clause.get("value")
    exact_text = clause.get("exact_text")
    page       = clause.get("page")
    confidence = clause.get("confidence", 0.0)
    validated  = clause.get("validated", False)

    if value is None:
        card_cls, badge_cls, badge_txt = "clause-card null-clause", "clause-badge b-null", "Not Found"
    elif validated:
        card_cls, badge_cls, badge_txt = "clause-card validated",   "clause-badge b-validated",     "Validated"
    else:
        card_cls, badge_cls, badge_txt = "clause-card not-validated","clause-badge b-not-validated", "Unverified"

    val_html = (
        f'<div class="clause-val">{value}</div>' if value
        else '<div class="clause-null">Not found in document</div>'
    )
    src_html = ""
    if exact_text:
        src_html = f'<div class="source-lbl">Source Text</div><div class="source-block">{exact_text}</div>'

    meta = ""
    if page is not None:
        meta += f'<span><span class="meta-k">Page</span><span class="meta-v">p.{page}</span></span>'
    if confidence:
        clr = conf_color(confidence)
        meta += (f'<span><span class="meta-k">Confidence</span>'
                 f'<span class="meta-v" style="color:{clr};">{int(confidence*100)}%</span></span>')

    st.markdown(
        f"""<div class="{card_cls}">
          <div class="clause-hdr">
            <span class="clause-name">{label}</span>
            <span class="{badge_cls}">{badge_txt}</span>
          </div>
          {val_html}{src_html}
          <div class="clause-meta">{meta}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_fields(fields: dict):
    field_defs = [
        ("jurisdiction",  "Jurisdiction"),
        ("payment_terms", "Payment Terms"),
        ("notice_period", "Notice Period"),
        ("liability_cap", "Liability Cap"),
    ]
    col1, col2 = st.columns(2)
    cols = [col1, col2, col1, col2]
    for i, (key, label) in enumerate(field_defs):
        val = fields.get(key)
        with cols[i]:
            inner = (
                f'<div class="field-val">{val}</div>' if val
                else '<div class="field-null">Not identified</div>'
            )
            st.markdown(
                f'<div class="field-card"><div class="field-label">{label}</div>{inner}</div>',
                unsafe_allow_html=True,
            )


def render_stats_bar(result: dict):
    clauses = result.get("clauses", {})
    n_found = sum(1 for c in clauses.values() if c.get("value") is not None)
    n_val   = sum(1 for c in clauses.values() if c.get("validated"))
    avg_c   = (sum(c.get("confidence", 0) for c in clauses.values() if c.get("value"))
               / max(n_found, 1))
    n_fld   = sum(1 for v in result.get("fields", {}).values() if v is not None)

    data = [
        (str(n_found), "/ 4", "Clauses Found",   "hl", "gold"),
        (str(n_val),   "",    "Validated",        "",   ""),
        (f"{int(avg_c*100)}%", "", "Avg Confidence", "", ""),
        (str(n_fld),   "/ 4", "Fields Extracted", "",   ""),
    ]
    cols = st.columns(4)
    for col, (val, sup, lbl, card_cls, val_cls) in zip(cols, data):
        with col:
            st.markdown(
                f"""<div class="stat-card {card_cls}">
                  <div class="stat-value {val_cls}">{val}<span class="stat-sup">{sup}</span></div>
                  <div class="stat-label">{lbl}</div>
                </div>""",
                unsafe_allow_html=True,
            )


def render_retrieved_chunks(retrieved_chunks: dict):
    if not retrieved_chunks:
        st.info("No retrieval data available.")
        return
    names = {
        "governing_law": "Governing Law", "audit_rights": "Audit Rights",
        "non_compete": "Non-Compete",     "non_solicitation": "Non-Solicitation",
        "contract_type": "Contract Type",
    }
    for cname, (texts, pages) in retrieved_chunks.items():
        if not texts: continue
        dn = names.get(cname, cname)
        with st.expander(f"{dn} — {len(texts)} chunks retrieved"):
            for i, (txt, pg) in enumerate(zip(texts, pages)):
                preview = txt[:400] + "..." if len(txt) > 400 else txt
                st.markdown(
                    f"""<div class="chunk-item">
                      <div class="chunk-hdr">
                        <span class="chunk-lbl">Chunk {i+1}</span>
                        <span class="chunk-pg">Page {pg}</span>
                      </div>
                      <div class="chunk-txt">{preview}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )


def render_llm_efficiency_panel(mode: str, call_count: int, call_log: list, elapsed: float):
    calls_saved = max(0, 10 - call_count)

    log_html = "".join(
        f'<div class="call-log-row"><span class="call-log-n">{i+1}</span>{lbl}</div>'
        for i, lbl in enumerate(call_log)
    ) or '<div class="call-log-row"><span class="call-log-n">—</span>No calls recorded</div>'

    rows = [
        ("Contract type",       "1 call",  "1 call"),
        ("4 clauses",           "4 calls", "1 call"),
        ("4 structured fields", "1 call",  "1 call"),
        ("Validation (4×)",     "4 calls", "1 call"),
        ("Total",
         "<span class='c-red'>10 calls</span>",
         "<span class='c-grn'>4 calls</span>"),
    ]
    tbl_rows = "".join(
        f"<tr><td>{l}</td><td class='c-red'>{o}</td><td class='c-grn'>{n}</td></tr>"
        for l, o, n in rows
    )

    st.markdown(
        f"""<div class="eff-panel">
          <div class="eff-title">Extraction Efficiency</div>
          <div class="eff-metrics">
            <div class="eff-metric">
              <div class="eff-m-lbl">LLM Calls Made</div>
              <div class="eff-m-val {'green' if mode == 'joint' else ''}">{call_count}</div>
            </div>
            <div class="eff-metric">
              <div class="eff-m-lbl">Calls Saved</div>
              <div class="eff-m-val green">{calls_saved}</div>
            </div>
            <div class="eff-metric">
              <div class="eff-m-lbl">Time Elapsed</div>
              <div class="eff-m-val gold">{elapsed:.1f}s</div>
            </div>
          </div>
          <div style="display:flex;gap:20px;flex-wrap:wrap;">
            <div style="flex:1;min-width:200px;">
              <div class="sb-section-label" style="margin-bottom:8px;">Call Log</div>
              <div class="call-log-wrap">{log_html}</div>
            </div>
            <div style="flex:1.2;min-width:240px;">
              <div class="sb-section-label" style="margin-bottom:8px;">Mode Comparison</div>
              <table class="compare-tbl">
                <thead><tr><th>Task</th><th>Individual</th><th>Joint</th></tr></thead>
                <tbody>{tbl_rows}</tbody>
              </table>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_extractor():
    return ContractExtractor()

@st.cache_resource(show_spinner=False)
def get_joint_extractor():
    return JointContractExtractor()


def main():
    inject_css()

    # TOP BAR
    st.markdown(
        """<div class="cie-topbar">
          <div class="cie-logo-group">
            <div class="cie-seal">⚖</div>
            <div>
              <div class="cie-wordmark">Lex<span>is</span></div>
              <div class="cie-tagline">Contract Intelligence Engine</div>
            </div>
          </div>
          <div class="cie-topbar-meta">
            <span class="cie-badge">gpt-4o-mini</span>
            <span class="cie-badge">v2.0.0</span>
            <div class="cie-status">
              <div class="cie-dot"></div>
              <span class="cie-status-txt">System Ready</span>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    if not OPENAI_API_KEY:
        st.error("**OPENAI_API_KEY not configured.** Add `OPENAI_API_KEY=sk-...` to your `.env` and restart.")
        st.stop()

    # SIDEBAR
    with st.sidebar:
        st.markdown(
            '<div class="sb-brand">'
            '<div class="sb-brand-name">⚖ Lexis</div>'
            '<div class="sb-brand-sub">Contract Intelligence</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Configuration")
        st.markdown('<div class="sb-section-label">Extraction Mode</div>', unsafe_allow_html=True)
        extraction_mode = st.radio(
            "mode",
            options=["Joint (Optimized)", "Individual (Baseline)"],
            index=0,
            label_visibility="collapsed",
            help="Joint: 4 LLM calls total. Individual: ~10 calls (baseline).",
        )
        use_joint = "Joint" in extraction_mode

        st.markdown("---")
        show_llm_stats = st.toggle("Show Efficiency Report", value=True)
        show_chunks    = st.toggle("Show Retrieved Chunks",  value=False)
        show_raw_json  = st.toggle("Show Raw JSON",          value=True)

        st.markdown("---")
        st.markdown("### Pipeline")
        pipeline = [
            ("01", "PDF parsing"),
            ("02", "Text chunking"),
            ("03", "Embedding"),
            ("04", "FAISS indexing"),
            ("05", "2-pass joint retrieval" if use_joint else "Per-clause retrieval"),
            ("06", "Joint clause extraction" if use_joint else "Clause extraction"),
            ("07", "Joint field extraction"  if use_joint else "Field extraction"),
            ("08", "Joint validation"        if use_joint else "Per-clause validation"),
            ("09", "JSON output"),
        ]
        joint_steps = {"05","06","07","08"}
        for num, step in pipeline:
            active = use_joint and num in joint_steps
            cls = "active" if active else ""
            st.markdown(
                f'<div class="pipe-step"><span class="pipe-num">{num}</span>'
                f'<span class="pipe-text {cls}">{step}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### Supported Types")
        for ct in ["Service Agreement", "Lease Agreement", "IP Agreement", "Supply Agreement"]:
            st.markdown(f'<div class="supported-type">{ct}</div>', unsafe_allow_html=True)

    # UPLOAD
    render_section_header("Upload Contract", "01")
    uploaded_file = st.file_uploader(
        "Drop a contract PDF here",
        type=["pdf"],
        help="Supports multi-page PDFs up to 200MB.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        mode_pill = (
            '<span class="mode-pill mode-joint">Joint Mode</span>'
            if use_joint else
            '<span class="mode-pill mode-individual">Individual Mode</span>'
        )
        st.markdown(
            f"""<div class="file-info-card">
              <div>
                <div class="file-name">{uploaded_file.name}</div>
                <div class="file-meta">{uploaded_file.size / 1024:.1f} KB &nbsp;·&nbsp; PDF Document</div>
              </div>
              <div>{mode_pill}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        _, col_btn = st.columns([4, 1])
        with col_btn:
            analyze_clicked = st.button("Run Analysis", key="analyze")
    else:
        analyze_clicked = False

    # ANALYSIS
    mode_key = "joint" if use_joint else "individual"
    if uploaded_file is not None and analyze_clicked:
        pdf_bytes  = uploaded_file.read()
        result_key = f"result_{mode_key}_{uploaded_file.name}_{len(pdf_bytes)}"

        progress_bar = st.progress(0)
        status_text  = st.empty()
        start_time   = time.time()

        def set_status(msg: str, pct: int):
            progress_bar.progress(pct)
            status_text.markdown(
                f"<div style='font-family:DM Mono,monospace;font-size:0.76rem;"
                f"color:#4a5e7a;padding:4px 0;letter-spacing:0.02em;'>{msg}</div>",
                unsafe_allow_html=True,
            )

        try:
            if use_joint:
                set_status("Initialising joint extractor…", 5)
                extractor  = get_joint_extractor()
                set_status("Parsing · embedding · retrieval…", 20)
                result_obj = extractor.process_bytes(pdf_bytes, uploaded_file.name)
                set_status("Assembling structured output…", 90)
                result_dict = result_obj.to_output_dict()
                call_stats  = extractor.call_stats
            else:
                set_status("Parsing PDF and building index…", 10)
                extractor  = get_extractor()
                set_status("Running per-clause extraction…", 40)
                result_obj = extractor.process_bytes(pdf_bytes, uploaded_file.name)
                set_status("Assembling structured output…", 90)
                result_dict = result_obj.to_output_dict()
                call_stats  = type("Stats", (), {
                    "total_calls": 10,
                    "call_log": [
                        "Contract type classification",
                        "Governing law extraction",
                        "Audit rights extraction",
                        "Non-compete extraction",
                        "Non-solicitation extraction",
                        "Structured fields extraction",
                        "Validation: governing law",
                        "Validation: audit rights",
                        "Validation: non-compete",
                        "Validation: non-solicitation",
                    ],
                })()

            elapsed = time.time() - start_time
            progress_bar.progress(100)
            status_text.empty()

            st.session_state[result_key] = {
                "result":           result_dict,
                "retrieved_chunks": extractor.retrieved_chunks,
                "elapsed":          elapsed,
                "mode":             mode_key,
                "call_count":       call_stats.total_calls,
                "call_log":         list(call_stats.call_log),
            }

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Extraction failed: {str(e)}")
            st.exception(e)
            st.stop()

        st.success(f"Analysis complete — {st.session_state[result_key]['elapsed']:.1f}s")

    # RESULTS
    result_data = None
    if uploaded_file:
        result_key = f"result_{mode_key}_{uploaded_file.name}_{uploaded_file.size}"
        if result_key in st.session_state:
            result_data = st.session_state[result_key]

    if result_data:
        result           = result_data["result"]
        retrieved_chunks = result_data.get("retrieved_chunks", {})
        result_mode      = result_data.get("mode", "individual")
        call_count       = result_data.get("call_count", 0)
        call_log         = result_data.get("call_log", [])
        elapsed          = result_data.get("elapsed", 0)

        render_stats_bar(result)

        render_section_header("Contract Classification", "02")
        render_contract_type(result["contract_type"])

        render_section_header("Extracted Clauses", "03")
        col_a, col_b = st.columns(2)
        for i, (name, data) in enumerate(result["clauses"].items()):
            with col_a if i % 2 == 0 else col_b:
                render_clause_card(name, data)

        render_section_header("Structured Fields", "04")
        render_fields(result["fields"])

        if show_llm_stats:
            render_section_header("Efficiency Report", "05")
            render_llm_efficiency_panel(result_mode, call_count, call_log, elapsed)

        if show_chunks:
            render_section_header("Retrieval Context", "06")
            render_retrieved_chunks(retrieved_chunks)

        if show_raw_json:
            render_section_header("Raw JSON Output", "07")
            st.json(result, expanded=False)
            json_str = json.dumps(result, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{Path(uploaded_file.name).stem}_extracted.json",
                mime="application/json",
            )

    elif uploaded_file is None:
        st.markdown(
            """<div class="empty-state">
              <div class="empty-emblem">⚖</div>
              <div class="empty-heading">Ready to analyse your contract</div>
              <div class="empty-sub">Upload a PDF to extract clauses, structured fields, and contract classification</div>
              <div class="empty-features">
                <div class="empty-feature">
                  <strong>Clause Extraction</strong>
                  Governing law, audit rights, non-compete, non-solicitation
                </div>
                <div class="empty-feature">
                  <strong>Field Detection</strong>
                  Jurisdiction, payment terms, notice period, liability cap
                </div>
                <div class="empty-feature">
                  <strong>LLM Validation</strong>
                  Cross-referenced, confidence-scored results
                </div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()