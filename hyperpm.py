import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

st.set_page_config(
    page_title="HYPER - Marketing Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 📊 HARDKÓDOLT PLATFORM-SPECIFIKUS MAPPINGEK (v7.1 - Google Ads Fix)
# ============================================================================

FACEBOOK_EXACT_MAPPING = {
    "Jelentés kezdete": "date_start",
    "Kampány neve": "campaign_name",
    "Kampány teljesítése": "campaign_status",
    "Elköltött összeg (HUF)": "spend",
    "Megjelenések": "impressions",
    "Elérés": "reach",
    "CPC (összes) (HUF)": "cpc",
    "CTR (átkattintási arány)": "ctr_percent",
    "Vásárlási hirdetésmegtérülés (ROAS)": "roas",
    "Gyakoriság": "frequency",
    "Eredmények": "results_count",
    "Eredményenkénti költség": "conv_cost",
    "Vásárlások": "conversions",
    "Vásárlások konverziós értéke": "conversion_value",
    "Kosárba helyezések": "add_to_cart",
    "Kosárba helyezés egységnyi költsége (HUF)": "add_to_cart_cost",
    "Kosárba helyezések konverziós értéke": "add_to_cart_value",
    "Engagement": "engagement",
}

FACEBOOK_SKIP_COLUMNS = {
    "Jelentés vége",
    "Hirdetéssorozat költségkerete",
    "Hirdetéssorozat költségkeretének típusa",
    "Vége",
    "Eredmény jelzése",
    "Hozzárendelés beállítása",
}

# Google Ads oszlopnevei (Magyar)
GOOGLE_ADS_EXACT_MAPPING = {
    "Kampány": "campaign_name",
    "Kampány állapota": "campaign_status",
    "Költség": "spend",
    "Interakciók": "clicks",
    "Interakciós arány": "ctr_percent",
    "Konverziók": "conversions",
    "Konverziós érték": "conversion_value",
    "Konverziós érték/költség": "roas",
    "Átl. CPC": "cpc",
    "Megjel.": "impressions",
    "Költség/konv.": "cpa",
}

GOOGLE_ADS_SKIP_COLUMNS = {
    "Kampány állapota",
    "Költségkeret",
    "Költségkeret neve",
    "Költségkerettípus azonosítója",
    "Pénznem kód",
    "Állapot",
    "Állapot okai",
    "Optimalizálási pontszám",
    "Kampánytípus",
    "Ajánlattételi stratégia típusa",
    "Keresési megj. arány",
    "Eredeti konv. érték",
}

UNIFIED_SCHEMA = {
    "mandatory": [
        ("date_start", "date", "Jelentés kezdete (dátum)"),
        ("campaign_name", "string", "Kampány neve"),
        ("platform", "string", "Platform (Facebook/Google Ads)"),
        ("campaign_status", "string", "Kampány státusza"),
        ("spend", "float", "Elköltött összeg (HUF)"),
        ("conversions", "int", "Konverziók (darabszám)"),
        ("conversion_value", "float", "Konverziós érték (HUF)"),
    ],
    "recommended": [
        ("impressions", "int", "Megjelenések"),
        ("clicks", "int", "Kattintások"),
        ("ctr_percent", "float", "CTR (%)"),
        ("cpc", "float", "CPC (HUF)"),
        ("cpa", "float", "CPA (HUF)"),
        ("conv_cost", "float", "Eredményenkénti költség (HUF)"),
        ("roas", "float", "ROAS (x)"),
        ("reach", "int", "Elérés"),
        ("frequency", "float", "Gyakoriság"),
        ("results_count", "int", "Eredmények (darabszám)"),
    ],
}

# ============================================================================
# 🔧 HELPERS
# ============================================================================

def clean_excel_structure(df):
    """
    Google Ads Excel-ből kihagyja az üres/szerkezeti sorokat.
    Az első igazi header-t keresi meg.
    """
    # Keresünk egy sort, ahol "Kampány" vagy "Költség" oszlop van
    for idx, row in df.iterrows():
        if any(col in str(row.values) for col in ["Kampány", "Költség", "Konverziók", "Megjel."]):
            # Ez a header sor!
            df_clean = df.iloc[idx+1:].reset_index(drop=True)
            df_clean.columns = row.values
            # Üres oszlopokat eltávolítjuk
            df_clean = df_clean.loc[:, ~df_clean.columns.str.contains("Unnamed")]
            return df_clean
    return df

def parse_numeric_value(val):
    if pd.isna(val) or val == "" or val == "–" or val == "--" or val == "folyamatban":
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace("\u00a0", "").replace(" ", "").replace("\"", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

def parse_percentage_value(val):
    if pd.isna(val) or val == "" or val == "–" or val == "< 10%" or val == "--":
        return np.nan
    s = str(val).strip().replace("%", "").replace(",", ".").replace(" ", "")
    try:
        return float(s)
    except:
        return np.nan

def parse_date(val):
    if pd.isna(val):
        return None
    formats = ["%Y-%m-%d", "%Y.%m.%d", "%d.%m.%Y", "%d-%m-%Y", "%m/%d/%Y"]
    for fmt in formats:
        try:
            return pd.to_datetime(val, format=fmt)
        except:
            pass
    try:
        return pd.to_datetime(val)
    except:
        return None

def detect_platform(df_columns):
    """Platform detektálása."""
    facebook_matches = sum(1 for col in df_columns if col in FACEBOOK_EXACT_MAPPING)
    google_matches = sum(1 for col in df_columns if col in GOOGLE_ADS_EXACT_MAPPING)
    
    if facebook_matches > google_matches:
        return "facebook"
    elif google_matches > facebook_matches:
        return "google_ads"
    else:
        return "unknown"

def create_mapping_from_platform(df_columns, platform):
    """Platform alapú mapping."""
    mapping = {}
    unmapped = []
    
    if platform == "facebook":
        skip_cols = FACEBOOK_SKIP_COLUMNS
        exact_map = FACEBOOK_EXACT_MAPPING
    elif platform == "google_ads":
        skip_cols = GOOGLE_ADS_SKIP_COLUMNS
        exact_map = GOOGLE_ADS_EXACT_MAPPING
    else:
        return {}, list(df_columns)
    
    for col in df_columns:
        if col in skip_cols or "Unnamed" in col:
            continue
        elif col in exact_map:
            mapping[col] = exact_map[col]
        else:
            unmapped.append(col)
    
    return mapping, unmapped

def normalize_data(df, mapping, platform):
    """Adatok normalizálása."""
    normalized_df = pd.DataFrame()

    for csv_col, unified_col in mapping.items():
        if csv_col not in df.columns:
            continue

        field_info = None
        for section in [UNIFIED_SCHEMA["mandatory"], UNIFIED_SCHEMA["recommended"]]:
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

    if "platform" not in normalized_df.columns:
        normalized_df["platform"] = platform.replace("_", " ").title()

    # CPA számítása, ha nincs
    if platform == "facebook":
        if "spend" in normalized_df.columns and "conversions" in normalized_df.columns:
            if "cpa" not in normalized_df.columns:
                normalized_df["cpa"] = normalized_df["spend"] / normalized_df["conversions"]
                normalized_df["cpa"] = normalized_df["cpa"].replace([np.inf, -np.inf], np.nan)

    # Intelligens "results" típus
    results_type_map = []
    for idx, row in normalized_df.iterrows():
        if not pd.isna(row.get("conversions")) and row.get("conversions", 0) > 0:
            results_type_map.append("Vásárlások")
        else:
            results_type_map.append("Egyéb")
    
    normalized_df["results"] = results_type_map

    return normalized_df

# ============================================================================
# 🎨 STREAMLIT UI
# ============================================================================

st.title("🎯 HYPER - Marketing Campaign Analyzer")
st.markdown("## Fázis 1: Intelligens CSV/Excel Importer")

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None
if "mapping" not in st.session_state:
    st.session_state.mapping = {}
if "platform" not in st.session_state:
    st.session_state.platform = None
if "normalized_data" not in st.session_state:
    st.session_state.normalized_data = None

tab1, tab2, tab3, tab4 = st.tabs(
    ["📥 Feltöltés & Mapping", "✅ Validáció", "📊 Előnézet", "💾 Mentés"]
)

with tab1:
    st.subheader("1️⃣ CSV/Excel Feltöltés")
    uploaded_file = st.file_uploader(
        "Válassz CSV vagy Excel fájlt",
        type=["csv", "xlsx", "xls"],
        help="Facebook vagy Google Ads export",
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            else:
                raw_df = pd.read_excel(uploaded_file)

            # Google Ads Excel tisztítása
            if uploaded_file.name.endswith((".xlsx", ".xls")):
                raw_df = clean_excel_structure(raw_df)

            st.session_state.uploaded_data = raw_df

            st.success(f"✅ Betöltve: {uploaded_file.name}")
            st.info(f"📊 Sorok: {len(raw_df)}, Oszlopok: {len(raw_df.columns)}")

            # Platform detektálása
            detected_platform = detect_platform(raw_df.columns)
            st.session_state.platform = detected_platform

            if detected_platform == "unknown":
                st.warning("⚠️ Nem sikerült felismerni a platform típusát. Válassz manuálisan:")
                selected_platform = st.selectbox("Platform:", ["Facebook", "Google Ads"])
                st.session_state.platform = "facebook" if selected_platform == "Facebook" else "google_ads"
            else:
                platform_name = "Facebook" if detected_platform == "facebook" else "Google Ads"
                st.success(f"✅ Felismert platform: {platform_name}")

            st.subheader("2️⃣ Automata Oszlop Felismerés")
            mapping, unmapped = create_mapping_from_platform(raw_df.columns, st.session_state.platform)
            st.session_state.mapping = mapping

            st.markdown("#### ✅ Leképezett oszlopok:")
            mapping_display = [
                {"CSV Oszlop": csv_col, "Unified Field": unified_col}
                for csv_col, unified_col in sorted(mapping.items())
            ]
            if mapping_display:
                st.dataframe(pd.DataFrame(mapping_display), use_container_width=True)
            else:
                st.warning("Nincs felismerés")

            if unmapped:
                st.markdown(f"#### ⚠️ Felismeretlen oszlopok ({len(unmapped)}):")
                for col in unmapped:
                    st.text(f"• {col}")

            st.subheader("📋 Adatok Előnézete")
            st.dataframe(raw_df.head(3), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Hiba: {str(e)}")

with tab2:
    if st.session_state.uploaded_data is not None:
        st.subheader("✅ Normalizálása")
        try:
            normalized_df = normalize_data(
                st.session_state.uploaded_data,
                st.session_state.mapping,
                st.session_state.platform,
            )
            st.session_state.normalized_data = normalized_df

            st.success("✅ Sikeres normalizálás!")
            st.info(f"Adatok: {len(normalized_df)} sor × {len(normalized_df.columns)} oszlop")
        except Exception as e:
            st.error(f"❌ Hiba: {str(e)}")
    else:
        st.info("Először töltsd fel az adatokat!")

with tab3:
    if st.session_state.normalized_data is not None:
        st.subheader("📊 Normalizált Adatok")
        try:
            df = st.session_state.normalized_data

            col1, col2, col3 = st.columns(3)
            with col1:
                if "spend" in df.columns:
                    st.metric("💰 Költség", f"{df['spend'].sum():,.0f} HUF")
            with col2:
                if "conversion_value" in df.columns:
                    st.metric("💵 Érték", f"{df['conversion_value'].sum():,.0f} HUF")
            with col3:
                if "roas" in df.columns:
                    st.metric("📈 ROAS", f"{df['roas'].mean():.2f}x")

            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Hiba: {str(e)}")
    else:
        st.info("Először normalizáld az adatokat!")

with tab4:
    if st.session_state.normalized_data is not None:
        st.subheader("💾 Exportálás")
        df = st.session_state.normalized_data

        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button("📥 CSV", csv, f"hyper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        with col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Kampanyok")
            st.download_button(
                "📥 Excel",
                buffer.getvalue(),
                f"hyper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("Először normalizáld az adatokat!")

st.divider()
st.markdown("**HYPER App v7.1** | Multi-Platform Support with Excel Structure Fix\n✅ Facebook + Google Ads - Auto-clean Excel headers")
