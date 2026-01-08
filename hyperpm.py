import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from thefuzz import fuzz
except ImportError:
    st.error("Hiányzik: pip install thefuzz python-Levenshtein")
    st.stop()

import io

# ============================================================================
# 🎨 HYPER App - Neuromarketing ROAS Predictor v3.9
# FÁZIS 1: CSV Importer & Intelligent Mapper
# ============================================================================

st.set_page_config(
    page_title="HYPER - Marketing Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 📊 CONFIG & SCHEMA
# ============================================================================

UNIFIED_SCHEMA = {
    "mandatory": [
        ("date_start", "date", "Jelentés kezdete (dátum)"),
        ("date_end", "date", "Jelentés vége (dátum)"),
        ("campaign_name", "string", "Kampány neve"),
        ("platform", "string", "Platform (Facebook/Google Ads/TikTok)"),
        ("campaign_status", "string", "Kampány státusza"),
        ("spend", "float", "Elköltött összeg (HUF)"),
        ("conversions", "int", "Konverziók / Vásárlások"),
        ("conversion_value", "float", "Konverziós érték (HUF)"),
    ],
    "recommended": [
        ("impressions", "int", "Megjelenések"),
        ("clicks", "int", "Kattintások / Interakciók"),
        ("ctr_percent", "float", "CTR (%)"),
        ("cpc", "float", "CPC (HUF)"),
        ("cpa", "float", "CPA (HUF, számított)"),
        ("conv_cost", "float", "Konverziós költség (HUF)"),
        ("roas", "float", "ROAS (x)"),
        ("reach", "int", "Elérés"),
        ("frequency", "float", "Gyakoriság"),
        ("add_to_cart", "int", "Kosárba helyezések"),
        ("cost_per_addtocart", "float", "Kosárba helyezés egységnyi költsége (HUF)"),
        ("results", "int", "Eredmények (FB results oszlop)"),
        ("adset_budget", "float", "Hirdetéssorozat költségkerete (HUF)"),
        ("adset_budget_type", "string", "Hirdetéssorozat költségkeretének típusa"),
    ],
    "optional": [
        ("video_views", "int", "Videó megtekintések"),
        ("engagement", "int", "Engagement"),
        ("notes", "string", "Megjegyzések"),
    ],
}

# FONTOS: „Eredmény jelzése” NINCS a mintákban → nem lesz felismerve.
COLUMN_PATTERNS = {
    "cpc": ["cpc (összes) (huf)", "cpc (összes)", "cpc"],
    "ctr_percent": ["ctr (átkattintási arány)", "ctr"],
    "spend": ["elköltött összeg (huf)", "elköltött összeg", "spend"],
    "reach": ["elérés"],
    # Eredmények: csak az a sor, aminek a neve PONTOSAN „Eredmények”
    "results": ["eredmények"],
    "conv_cost": ["eredményenkénti költség", "cost per result"],
    "frequency": ["gyakoriság"],
    "adset_budget": ["hirdetéssorozat költségkerete"],
    "adset_budget_type": ["hirdetéssorozat költségkeretének típusa"],
    "date_start": ["jelentés kezdete"],
    "date_end": ["jelentés vége", "vége"],
    "campaign_name": ["kampány neve"],
    "campaign_status": ["kampány teljesítése"],
    "cost_per_addtocart": ["kosárba helyezés egységnyi költsége"],
    "add_to_cart": ["kosárba helyezések"],
    "conversion_value": [
        "vásárlások konverziós értéke",
        "kosárba helyezések konverziós értéke",
    ],
    "impressions": ["megjelenések"],
    "roas": ["vásárlási hirdetésmegtérülés"],
    # conversions: csak „Vásárlások”/„Konverziók” típusú oszlopokra ül rá,
    # nem fog ráülni a „Vásárlások konverziós értéke” oszlopra.
    "conversions": ["vásárlások", "konverziók", "purchases", "orders"],
    "clicks": ["link click", "clicks", "kattintás", "kattintások"],
}

# ============================================================================
# 🔧 HELPERS
# ============================================================================

def find_matching_column(csv_column, patterns_dict, threshold=80):
    """Szándékosan magasabb threshold, hogy ne keverje össze „Eredmény jelzése” vs „Eredmények”."""
    csv_col_lower = csv_column.lower().strip()
    best_match = None
    best_score = 0
    for unified_field, patterns in patterns_dict.items():
        for pattern in patterns:
            score = fuzz.partial_ratio(csv_col_lower, pattern.lower())
            if score > best_score:
                best_score = score
                best_match = unified_field
    if best_score >= threshold:
        return best_match, best_score
    return None, best_score


def intelligently_map_columns(df_columns):
    mapping = {}
    unmapped = []
    for col in df_columns:
        matched_field, score = find_matching_column(col, COLUMN_PATTERNS)
        if matched_field:
            mapping[col] = matched_field
        else:
            unmapped.append(col)
    return mapping, unmapped


def parse_numeric_value(val):
    if pd.isna(val) or val == "" or val == "–" or val == "--":
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    s = s.replace("\u00a0", "").replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_percentage_value(val):
    if pd.isna(val) or val == "" or val == "–":
        return np.nan
    s = str(val).strip().replace("%", "")
    s = s.replace("\u00a0", "").replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_date(val):
    if pd.isna(val):
        return None
    date_formats = [
        "%Y-%m-%d",
        "%Y.%m.%d",
        "%d.%m.%Y",
        "%d-%m-%Y",
        "%m/%d/%Y",
    ]
    for fmt in date_formats:
        try:
            return pd.to_datetime(val, format=fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(val)
    except Exception:
        return None


def normalize_data(df, mapping, user_adjustments=None, platform_hint=None):
    if user_adjustments:
        mapping = {**mapping, **user_adjustments}

    normalized_df = pd.DataFrame()

    for csv_col, unified_col in mapping.items():
        if csv_col not in df.columns:
            continue

        field_info = None
        for section in [UNIFIED_SCHEMA["mandatory"], UNIFIED_SCHEMA["recommended"], UNIFIED_SCHEMA["optional"]]:
            for field in section:
                if field[0] == unified_col:
                    field_info = field
                    break

        if not field_info:
            continue

        field_name, field_type, _ = field_info
        raw_data = df[csv_col]

        if field_name == "ctr_percent":
            normalized_df[field_name] = raw_data.apply(parse_percentage_value)
        elif field_type == "float":
            normalized_df[field_name] = raw_data.apply(parse_numeric_value)
        elif field_type == "int":
            normalized_df[field_name] = raw_data.apply(
                lambda x: int(parse_numeric_value(x))
                if not pd.isna(parse_numeric_value(x))
                else np.nan
            )
        elif field_type == "date":
            normalized_df[field_name] = raw_data.apply(parse_date)
        elif field_type == "string":
            normalized_df[field_name] = raw_data.astype(str)
        else:
            normalized_df[field_name] = raw_data

    if "date_start" in normalized_df.columns and "date_end" not in normalized_df.columns:
        normalized_df["date_end"] = normalized_df["date_start"]

    if "platform" not in normalized_df.columns:
        normalized_df["platform"] = platform_hint if platform_hint else "Unknown"

    if "spend" in normalized_df.columns and "conversion_value" in normalized_df.columns:
        if "roas" not in normalized_df.columns:
            normalized_df["roas"] = normalized_df["conversion_value"] / normalized_df["spend"]
            normalized_df["roas"] = normalized_df["roas"].replace([np.inf, -np.inf], np.nan)

    if "spend" in normalized_df.columns and "conversions" in normalized_df.columns:
        if "cpa" not in normalized_df.columns:
            normalized_df["cpa"] = normalized_df["spend"] / normalized_df["conversions"]
            normalized_df["cpa"] = normalized_df["cpa"].replace([np.inf, -np.inf], np.nan)

    if "clicks" in normalized_df.columns and "impressions" in normalized_df.columns:
        if "ctr_percent" not in normalized_df.columns:
            normalized_df["ctr_percent"] = (
                normalized_df["clicks"] / normalized_df["impressions"] * 100
            )
            normalized_df["ctr_percent"] = normalized_df["ctr_percent"].replace(
                [np.inf, -np.inf], np.nan
            )

    if "spend" in normalized_df.columns and "clicks" in normalized_df.columns:
        if "cpc" not in normalized_df.columns:
            normalized_df["cpc"] = normalized_df["spend"] / normalized_df["clicks"]
            normalized_df["cpc"] = normalized_df["cpc"].replace([np.inf, -np.inf], np.nan)

    return normalized_df


def validate_data(df):
    issues = []
    mandatory_fields = [f[0] for f in UNIFIED_SCHEMA["mandatory"]]
    for field in mandatory_fields:
        if field not in df.columns:
            issues.append(f"❌ Hiányzik: {field}")
        elif df[field].isna().sum() > len(df) * 0.5:
            issues.append(f"⚠️ Túl sok hiányzik: {field} ({df[field].isna().sum()} / {len(df)})")

    if "roas" in df.columns:
        invalid_roas = df[(df["roas"] < 0) | (df["roas"] > 100)].shape[0]
        if invalid_roas > 0:
            issues.append(f"⚠️ Érvénytelen ROAS értékek: {invalid_roas}")

    if "cpa" in df.columns:
        invalid_cpa = df[(df["cpa"] < 0)].shape[0]
        if invalid_cpa > 0:
            issues.append(f"⚠️ Negatív CPA értékek: {invalid_cpa}")

    return issues


# ============================================================================
# 🎨 STREAMLIT UI
# ============================================================================

st.title("🎯 HYPER - Marketing Campaign Analyzer")
st.markdown("## Fázis 1: Intelligens CSV Importer")

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None
if "mapping" not in st.session_state:
    st.session_state.mapping = {}
if "normalized_data" not in st.session_state:
    st.session_state.normalized_data = None

tab1, tab2, tab3, tab4 = st.tabs(
    ["📥 Feltöltés & Mapping", "✅ Validáció", "📊 Előnézet", "💾 Mentés"]
)

# ---------------------------------------------------------------------------
# TAB 1 – Feltöltés & Mapping
# ---------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("1️⃣ CSV/Excel Feltöltés")
        uploaded_file = st.file_uploader(
            "Válassz CSV vagy Excel fájlt",
            type=["csv", "xlsx", "xls"],
            help="Facebook, Google Ads vagy TikTok export",
        )

    with col2:
        st.subheader("ℹ️ Támogatott formátumok")
        st.markdown(
            """
        - ✅ Facebook Ads Manager
        - ✅ Google Ads
        - ⏳ TikTok (hamarosan)
        """
        )

    platform_hint = st.selectbox(
        "Melyik platformról származik ez a fájl?",
        ["Facebook", "Google Ads", "TikTok", "Ismeretlen"],
        index=0,
        help="Ez bekerül a 'platform' oszlopba, ha a CSV-ben nincs ilyen mező.",
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            else:
                raw_df = pd.read_excel(uploaded_file)

            st.session_state.uploaded_data = raw_df

            st.success(f"✅ Betöltve: {uploaded_file.name}")
            st.info(f"📊 Sorok: {len(raw_df)}, Oszlopok: {len(raw_df.columns)}")

            st.subheader("2️⃣ Automata Oszlop Felismerés")
            initial_mapping, unmapped = intelligently_map_columns(raw_df.columns)
            st.session_state.mapping = initial_mapping

            st.markdown("#### 🔄 Automatikusan felismert oszlopok:")
            mapped_cols = st.expander("✅ Leképezett oszlopok", expanded=True)
            with mapped_cols:
                mapping_display = [
                    {"CSV Oszlop": csv_col, "Unified Field": unified_col}
                    for csv_col, unified_col in sorted(initial_mapping.items())
                ]
                if mapping_display:
                    st.dataframe(pd.DataFrame(mapping_display), use_container_width=True)
                else:
                    st.warning("Nincs automata felismerés :(")

            if unmapped:
                unmapped_cols = st.expander(f"⚠️ Felismeretlen oszlopok ({len(unmapped)})")
                with unmapped_cols:
                    st.warning("A következő oszlopok nem kerültek besorolásra:")
                    for col in unmapped:
                        st.text(f"• {col}")

            st.subheader("3️⃣ Manuális Korrekció (opcionális)")
            st.markdown("Ha valamelyik mező rossz helyre került, itt tudod javítani.")

            manual_corrections = {}
            all_fields = ["--Nincs--"]
            for section in [
                UNIFIED_SCHEMA["mandatory"],
                UNIFIED_SCHEMA["recommended"],
                UNIFIED_SCHEMA["optional"],
            ]:
                for field in section:
                    all_fields.append(field[0])

            with st.expander("Manuális mapping szerkesztése"):
                for csv_col in raw_df.columns:
                    current_mapping = initial_mapping.get(csv_col, "--Nincs--")
                    new_mapping = st.selectbox(
                        csv_col,
                        all_fields,
                        index=all_fields.index(current_mapping)
                        if current_mapping in all_fields
                        else 0,
                        key=f"manual_map_{csv_col}",
                    )
                    if new_mapping != "--Nincs--":
                        manual_corrections[csv_col] = new_mapping

            if manual_corrections:
                st.session_state.mapping.update(manual_corrections)

            st.subheader("📋 Adatok Előnézete (RAW)")
            st.dataframe(raw_df.head(10), use_container_width=True)

            st.session_state.platform_hint = (
                platform_hint if platform_hint != "Ismeretlen" else None
            )

        except Exception as e:
            st.error(f"❌ Hiba a fájl feldolgozásakor: {str(e)}")

# ---------------------------------------------------------------------------
# TAB 2 – Validáció
# ---------------------------------------------------------------------------
with tab2:
    if st.session_state.uploaded_data is not None:
        st.subheader("✅ Adatok Normalizálása & Validálása")
        try:
            normalized_df = normalize_data(
                st.session_state.uploaded_data,
                st.session_state.mapping,
                platform_hint=getattr(st.session_state, "platform_hint", None),
            )
            st.session_state.normalized_data = normalized_df

            validation_issues = validate_data(normalized_df)

            if validation_issues:
                st.warning("### ⚠️ Validációs Figyelmeztetések")
                for issue in validation_issues:
                    st.warning(issue)
            else:
                st.success("### ✅ Minden OK! Az adatok készen állnak.")

            st.info(
                f"Normalizált adatok: {len(normalized_df)} sor × {len(normalized_df.columns)} oszlop"
            )
        except Exception as e:
            st.error(f"❌ Hiba a normalizálás során: {str(e)}")
    else:
        st.info("Először töltsd fel az adatokat a '📥 Feltöltés & Mapping' fülön!")

# ---------------------------------------------------------------------------
# TAB 3 – Előnézet
# ---------------------------------------------------------------------------
with tab3:
    if st.session_state.normalized_data is not None:
        st.subheader("📊 Normalizált Adatok Előnézete")
        try:
            df = st.session_state.normalized_data

            col1, col2, col3 = st.columns(3)
            with col1:
                if "spend" in df.columns:
                    st.metric("💰 Teljes Költség (HUF)", f"{df['spend'].sum():,.0f}")
            with col2:
                if "conversion_value" in df.columns:
                    st.metric(
                        "💵 Konverziós Érték (HUF)",
                        f"{df['conversion_value'].sum():,.0f}",
                    )
            with col3:
                if "roas" in df.columns:
                    st.metric("📈 Átlag ROAS (x)", f"{df['roas'].mean():.2f}")

            st.subheader("Adatok Táblázat")
            df_display = df.copy()

            def fmt_int(x):
                if pd.isna(x):
                    return ""
                return f"{int(round(x)):,}".replace(",", " ")

            def fmt_huf(x):
                if pd.isna(x):
                    return ""
                return f"{int(round(x)):,}".replace(",", " ")

            def fmt_ctr(x):
                if pd.isna(x):
                    return ""
                return f"{x:.2f}".replace(".", ",") + "%"

            for col in ["conversions", "impressions", "clicks", "add_to_cart", "reach", "results"]:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(fmt_int)

            if "frequency" in df_display.columns:
                df_display["frequency"] = df_display["frequency"].apply(
                    lambda x: "" if pd.isna(x) else f"{x:.4f}"
                )

            huf_cols = {
                "spend": "spend (HUF)",
                "conversion_value": "conversion_value (HUF)",
                "cpc": "cpc (HUF)",
                "cpa": "cpa (HUF, számított)",
                "conv_cost": "conv_cost (HUF)",
                "cost_per_addtocart": "cost_per_addtocart (HUF)",
                "adset_budget": "adset_budget (HUF)",
            }
            for src, dst in huf_cols.items():
                if src in df_display.columns:
                    df_display[dst] = df_display[src].apply(fmt_huf)
                    del df_display[src]

            if "ctr_percent" in df_display.columns:
                df_display["ctr_percent (%)"] = df_display["ctr_percent"].apply(fmt_ctr)
                del df_display["ctr_percent"]

            st.dataframe(df_display, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Hiba az előnézet során: {str(e)}")
    else:
        st.info("Először töltsd fel és normalizáld az adatokat.")

# ---------------------------------------------------------------------------
# TAB 4 – Mentés
# ---------------------------------------------------------------------------
with tab4:
    if st.session_state.normalized_data is not None:
        st.subheader("💾 Adatok Exportálása")
        df = st.session_state.normalized_data

        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 CSV letöltés",
                data=csv,
                file_name=f"hyper_normalized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        with col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Campaigns")
            st.download_button(
                "📥 Excel letöltés",
                data=buffer.getvalue(),
                file_name=f"hyper_normalized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.info("Először töltsd fel és normalizáld az adatokat.")

st.divider()
st.markdown(
    """
**HYPER App v3.9** | Neuromarketing ROAS Predictor  
Fázis 1 kész – jöhet a Fázis 2 (Creative Analyzer + ML modell).
"""
)
