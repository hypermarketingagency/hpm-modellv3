import csv
import pandas as pd
import streamlit as st

from analytics.rollups import build_rollups


def _detect_csv_header_and_delimiter(uploaded_file, encoding):
    uploaded_file.seek(0)
    sample_bytes = uploaded_file.read(65536)
    uploaded_file.seek(0)
    sample_text = sample_bytes.decode(encoding, errors="ignore")
    lines = [line for line in sample_text.splitlines() if line.strip()]
    header_keywords = [
        "kampány",
        "campaign",
        "kampány neve",
        "kampány név",
        "nap",
        "date",
    ]
    header_index = 0
    header_line = lines[0] if lines else ""
    for idx, line in enumerate(lines[:50]):
        lowered = line.lower()
        if any(keyword in lowered for keyword in header_keywords):
            header_index = idx
            header_line = line
            break
    try:
        dialect = csv.Sniffer().sniff(header_line)
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = None
    return header_index, delimiter


def _read_csv_with_fallbacks(uploaded_file, encoding):
    skiprows, delimiter = _detect_csv_header_and_delimiter(uploaded_file, encoding)
    read_kwargs = {
        "encoding": encoding,
        "engine": "python",
        "on_bad_lines": "skip",
        "skiprows": skiprows,
    }
    if delimiter:
        read_kwargs["sep"] = delimiter
    else:
        read_kwargs["sep"] = None
    return pd.read_csv(uploaded_file, **read_kwargs)


def load_uploaded_dataframe(uploaded_file, clean_excel_structure):
    if uploaded_file.name.endswith(".csv"):
        try:
            df = _read_csv_with_fallbacks(uploaded_file, "utf-8-sig")
        except (pd.errors.ParserError, UnicodeDecodeError):
            uploaded_file.seek(0)
            df = _read_csv_with_fallbacks(uploaded_file, "cp1250")
        except Exception:
            uploaded_file.seek(0)
            df = _read_csv_with_fallbacks(uploaded_file, "latin-1")
    else:
        df = pd.read_excel(uploaded_file)
    if uploaded_file.name.endswith((".xlsx", ".xls")):
        df = clean_excel_structure(df)
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\u00a0", " ", regex=False)
        .str.strip()
    )
    return df


def detect_breakdown_type(df):
    if "geo_city" in df.columns and df["geo_city"].notna().any():
        return "geo"
    if "geo_region" in df.columns and df["geo_region"].notna().any():
        return "geo"
    if "age_group" in df.columns or "gender" in df.columns:
        return "demography"
    return "general"


def annotate_source_columns(df, filename):
    df = df.copy()
    df["source_file"] = filename
    df["breakdown_type"] = detect_breakdown_type(df)
    return df


def render_pulsing_logo(logo_url):
    pulsing_html = f"""
        <style>
        .pulse-logo {{
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 12px 0 20px 0;
        }}
        .pulse-logo img {{
            width: 110px;
            height: auto;
            animation: pulse 1.4s ease-in-out infinite;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(0.95); opacity: 0.6; }}
            50% {{ transform: scale(1.05); opacity: 1; }}
            100% {{ transform: scale(0.95); opacity: 0.6; }}
        }}
        </style>
        <div class="pulse-logo">
            <img src="{logo_url}" alt="HYPER logo" />
        </div>
    """
    container = st.empty()
    container.markdown(pulsing_html, unsafe_allow_html=True)
    return container


@st.cache_data(show_spinner=False)
def build_dashboard_summary(df):
    if "platform" not in df.columns:
        return pd.DataFrame()
    metrics = {
        "spend": "sum",
        "conversions": "sum",
        "conversion_value": "sum",
        "impressions": "sum",
    }
    metrics = {key: value for key, value in metrics.items() if key in df.columns}
    if not metrics:
        return df[["platform"]].drop_duplicates().reset_index(drop=True)
    summary = df.groupby("platform").agg(metrics).round(2)
    return summary.reset_index()


@st.cache_data(show_spinner=False)
def build_segment_rollup(df, group_cols):
    metrics = {
        "spend": "sum",
        "conversions": "sum",
        "conversion_value": "sum",
        "impressions": "sum",
    }
    return df.groupby(group_cols).agg(metrics).reset_index()


@st.cache_data(show_spinner=False)
def cached_rollups(df):
    return build_rollups(df)
