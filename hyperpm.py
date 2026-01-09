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
# 📊 HARDKÓDOLT PLATFORM-SPECIFIKUS MAPPINGEK (v8.0 - TikTok Support)
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
    "Átl. költség": "avg_cost",
    "Konv. arány": "conversion_rate",
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

# TikTok Ads oszlopnevei (Angol)
TIKTOK_EXACT_MAPPING = {
    "Campaign name": "campaign_name",
    "Primary status": "campaign_status",
    "Cost": "spend",
    "Impressions": "impressions",
    "Clicks (destination)": "clicks",
    "CTR (destination)": "ctr_percent",
    "Purchases (website)": "conversions",
    "Purchase value (website)": "conversion_value",
    "Purchase ROAS (app)": "roas_app",
    "Payment completion ROAS (website)": "roas",
    "Checkouts initiated (website)": "checkouts_initiated",
    "Adds to cart (website)": "add_to_cart",
    "Video views": "video_views",
    "Video views at 50%": "video_views_50",
    "Video views at 100%": "video_views_100",
    "2-second video views": "video_views_2s",
    "6-second video views": "video_views_6s",
}

TIKTOK_SKIP_COLUMNS = {
    "Currency",
    "Total of",
}

UNIFIED_SCHEMA = {
    "mandatory": [
        ("date_start", "date", "Jelentés kezdete (dátum)"),
        ("campaign_name", "string", "Kampány neve"),
        ("platform", "string", "Platform (Facebook/Google Ads/TikTok)"),
        ("campaign_status", "string", "Kampány státusza"),
        ("spend", "float", "Elköltött összeg (HUF)"),
        ("conversions", "int", "Konverziók (darabszám)"),
        ("conversion_value", "float", "Konverziós érték (HUF)"),
    ],
    "recommended": [
        ("impressions", "int", "Megjelenések"),
        ("clicks", "int", "Kattintások"),
        ("ctr_percent", "percentage", "CTR (%)"),
        ("cpc", "float", "CPC (HUF)"),
        ("cpa", "float", "CPA (HUF)"),
        ("avg_cost", "float", "Átlagos költség (HUF)"),
        ("conversion_rate", "percentage", "Konverziós ráta (%)"),
        ("conv_cost", "float", "Eredményenkénti költség (HUF)"),
        ("roas", "float", "ROAS (x)"),
        ("roas_app", "float", "ROAS App (x)"),
        ("reach", "int", "Elérés"),
        ("frequency", "float", "Gyakoriság"),
        ("results_count", "int", "Eredmények (darabszám)"),
        ("add_to_cart", "int", "Kosárba helyezések (darabszám)"),
        ("checkouts_initiated", "int", "Kezdeményezett fizetések (darabszám)"),
        ("video_views", "int", "Videó megtekintések"),
        ("video_views_50", "int", "50% videó megtekintések"),
        ("video_views_100", "int", "100% videó megtekintések"),
        ("video_views_2s", "int", "2 mp videó megtekintések"),
        ("video_views_6s", "int", "6 mp videó megtekintések"),
    ],
}

# ============================================================================
# 🔧 HELPERS
# ============================================================================

def clean_excel_structure(df):
    """Google Ads/TikTok Excel-ből kihagyja az üres/szerkezeti sorokat."""
    for idx, row in df.iterrows():
        if any(col in str(row.values) for col in ["Kampány", "Költség", "Konverziók", "Megjel.", "Campaign name", "Cost"]):
            df_clean = df.iloc[idx+1:].reset_index(drop=True)
            df_clean.columns = row.values
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
    """Percentáge értékek feldolgozása - szorzás 100-zal ha szükséges."""
    if pd.isna(val) or val == "" or val == "–" or val == "< 10%" or val == "--":
        return np.nan
    s = str(val).strip().replace("%", "").replace(",", ".").replace(" ", "")
    try:
        num = float(s)
        if 0 <= num <= 1:
            return num * 100
        return num
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
    tiktok_matches = sum(1 for col in df_columns if col in TIKTOK_EXACT_MAPPING)
    
    max_matches = max(facebook_matches, google_matches, tiktok_matches)
    
    if max_matches == 0:
        return "unknown"
    elif facebook_matches == max_matches:
        return "facebook"
    elif google_matches == max_matches:
        return "google_ads"
    elif tiktok_matches == max_matches:
        return "tiktok"
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
    elif platform == "tiktok":
        skip_cols = TIKTOK_SKIP_COLUMNS
        exact_map = TIKTOK_EXACT_MAPPING
    else:
        return {}, list(df_columns)
    
    for col in df_columns:
        if col in skip_cols or "Unnamed" in col or "Total of" in col:
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

        if field_type == "percentage":
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
        platform_display = platform.replace("_", " ").title()
        if platform == "tiktok":
            platform_display = "TikTok"
        normalized_df["platform"] = platform_display

    # CPA számítása, ha nincs
    if "spend" in normalized_df.columns and "conversions" in normalized_df.columns:
        if "cpa" not in normalized_df.columns or normalized_df["cpa"].isna().all():
            normalized_df["cpa"] = normalized_df["spend"] / normalized_df["conversions"]
            normalized_df["cpa"] = normalized_df["cpa"].replace([np.inf, -np.inf], np.nan)

    # Intelligens "results" típus
    results_type_map = []
    for idx, row in normalized_df.iterrows():
        if not pd.isna(row.get("conversions")) and row.get("conversions", 0) > 0:
            results_type_map.append("Vásárlások")
        elif not pd.isna(row.get("add_to_cart")) and row.get("add_to_cart", 0) > 0:
            results_type_map.append("Kosárba helyezések")
        elif not pd.isna(row.get("video_views")) and row.get("video_views", 0) > 0:
            results_type_map.append("Videó megtekintések")
        else:
            results_type_map.append("Egyéb")
    
    normalized_df["results"] = results_type_map

    return normalized_df

# ============================================================================
# 📊 FORMÁZÁS - Percentage oszlopok %-ás megjelenítéshez
# ============================================================================

def format_dataframe_for_display(df):
    """Streamlit dataframe formázása - percentage oszlopok %-ás kijelzéshez."""
    display_df = df.copy()
    
    percentage_cols = ["ctr_percent", "conversion_rate"]
    for col in percentage_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "–")
    
    return display_df

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
        help="Facebook, Google Ads vagy TikTok export",
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            else:
                raw_df = pd.read_excel(uploaded_file)

            # Excel tisztítása (Google Ads/TikTok)
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
                selected_platform = st.selectbox("Platform:", ["Facebook", "Google Ads", "TikTok"])
                platform_map = {"Facebook": "facebook", "Google Ads": "google_ads", "TikTok": "tiktok"}
                st.session_state.platform = platform_map[selected_platform]
            else:
                platform_names = {"facebook": "Facebook", "google_ads": "Google Ads", "tiktok": "TikTok"}
                platform_name = platform_names[detected_platform]
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
                if "roas" in df.columns and df["roas"].notna().any():
                    st.metric("📈 ROAS", f"{df['roas'].mean():.2f}x")
                elif "video_views" in df.columns and df["video_views"].notna().any():
                    st.metric("🎬 Videó nézetek", f"{df['video_views'].sum():,.0f}")

            display_df = format_dataframe_for_display(df)
            st.dataframe(display_df, use_container_width=True)
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
st.markdown("**HYPER App v8.0** | Multi-Platform Support: Facebook + Google Ads + TikTok\n✅ Automata platform detektálás • % formázás • Video metrics support")
