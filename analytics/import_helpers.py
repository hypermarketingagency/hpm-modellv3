import pandas as pd
import streamlit as st

from analytics.rollups import build_rollups


def load_uploaded_dataframe(uploaded_file, clean_excel_structure):
    if uploaded_file.name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except pd.errors.ParserError:
            uploaded_file.seek(0)
            return pd.read_csv(
                uploaded_file,
                encoding="utf-8-sig",
                sep=None,
                engine="python",
                on_bad_lines="skip",
            )
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(
                uploaded_file,
                encoding="latin-1",
                sep=None,
                engine="python",
                on_bad_lines="skip",
            )
    raw_df = pd.read_excel(uploaded_file)
    if uploaded_file.name.endswith((".xlsx", ".xls")):
        raw_df = clean_excel_structure(raw_df)
    return raw_df


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
    summary = df.groupby("platform").agg({
        "spend": "sum",
        "conversions": "sum",
        "conversion_value": "sum",
        "impressions": "sum",
    }).round(2)
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
