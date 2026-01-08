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
# 📊 HARDKÓDOLT FACEBOOK MAPPING (v6.0 - EXAKT EGYEZÉSEK)
# ============================================================================

# Ez a lista az EXAKT oszlop neveket tartalmazza a Facebooktól
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
    
    # Eredmények oszlop - a darabszám
    "Eredmények": "results_count",
    
    # Eredményenkénti költség
    "Eredményenkénti költség": "conv_cost",
    
    # Vásárlások
    "Vásárlások": "conversions",
    "Vásárlások konverziós értéke": "conversion_value",
    
    # Kosárba helyezések
    "Kosárba helyezések": "add_to_cart",
    "Kosárba helyezés egységnyi költsége (HUF)": "add_to_cart_cost",
    "Kosárba helyezések konverziós értéke": "add_to_cart_value",
    
    # Engagement
    "Engagement": "engagement",
    
    # Kihagyandó oszlopok (nem mappeljük)
    # "Jelentés vége" → SKIP
    # "Hirdetéssorozat költségkerete" → SKIP
    # "Hirdetéssorozat költségkeretének típusa" → SKIP
    # "Vége" → SKIP
    # "Eredmény jelzése" → SKIP
    # "Hozzárendelés beállítása" → SKIP
}

# Azok az oszlop nevek, amelyeket NEM mappelünk
SKIP_COLUMNS = {
    "Jelentés vége",
    "Hirdetéssorozat költségkerete",
    "Hirdetéssorozat költségkeretének típusa",
    "Vége",
    "Eredmény jelzése",
    "Hozzárendelés beállítása",
}

UNIFIED_SCHEMA = {
    "mandatory": [
        ("date_start", "date", "Jelentés kezdete (dátum)"),
        ("campaign_name", "string", "Kampány neve"),
        ("platform", "string", "Platform (Facebook/Google Ads/TikTok)"),
        ("campaign_status", "string", "Kampány státusza"),
        ("spend", "float", "Elköltött összeg (HUF)"),
        ("conversions", "int", "Konverziók / Vásárlások (darabszám)"),
        ("conversion_value", "float", "Vásárlások konverziós értéke (HUF)"),
    ],
    "recommended": [
        ("impressions", "int", "Megjelenések"),
        ("ctr_percent", "float", "CTR (%)"),
        ("cpc", "float", "CPC (HUF)"),
        ("cpa", "float", "CPA (HUF, számított)"),
        ("conv_cost", "float", "Eredményenkénti költség (HUF)"),
        ("roas", "float", "ROAS (x)"),
        ("reach", "int", "Elérés"),
        ("frequency", "float", "Gyakoriság"),
        ("results_count", "int", "Eredmények (darabszám)"),
        ("add_to_cart", "int", "Kosárba helyezések (darabszám)"),
        ("add_to_cart_cost", "float", "Kosárba helyezés egységnyi költsége (HUF)"),
        ("add_to_cart_value", "float", "Kosárba helyezések konverziós értéke (HUF)"),
        ("engagement", "int", "Engagement"),
    ],
    "optional": [],
}

# ============================================================================
# 🔧 HELPERS
# ============================================================================

def parse_numeric_value(val):
    if pd.isna(val) or val == "" or val == "–" or val == "--" or val == "folyamatban":
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace("\u00a0", "").replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

def parse_percentage_value(val):
    if pd.isna(val) or val == "" or val == "–":
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

def create_mapping_from_facebook(df_columns):
    """
    Facebook export oszlopaiból automatikus mapping létrehozása.
    EXAKT egyezéseket használ, nem fuzzy matching-et!
    """
    mapping = {}
    unmapped = []
    
    for col in df_columns:
        if col in SKIP_COLUMNS:
            # Ezeket nem mappeljük
            continue
        elif col in FACEBOOK_EXACT_MAPPING:
            # Exakt egyezés
            mapping[col] = FACEBOOK_EXACT_MAPPING[col]
        else:
            # Nem tudtuk felismerni
            unmapped.append(col)
    
    return mapping, unmapped

def normalize_data(df, mapping, platform_hint=None):
    """Adatok normalizálása."""
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

    if "platform" not in normalized_df.columns:
        normalized_df["platform"] = platform_hint if platform_hint else "Facebook"

    # Intelligens "results" típus
    results_type_map = []
    for idx, row in normalized_df.iterrows():
        if not pd.isna(row.get("conversions")) and row.get("conversions", 0) > 0:
            results_type_map.append("Vásárlások")
        elif not pd.isna(row.get("add_to_cart")) and row.get("add_to_cart", 0) > 0:
            results_type_map.append("Kosárba helyezések")
        elif not pd.isna(row.get("engagement")) and row.get("engagement", 0) > 0:
            results_type_map.append("Engagement")
        elif not pd.isna(row.get("results_count")) and row.get("results_count", 0) > 0:
            results_type_map.append("Egyéb")
        else:
            results_type_map.append("Nincs adat")
    
    normalized_df["results"] = results_type_map

    # Számított mezők
    if "spend" in normalized_df.columns and "conversion_value" in normalized_df.columns:
        if "roas" not in normalized_df.columns:
            normalized_df["roas"] = normalized_df["conversion_value"] / normalized_df["spend"]
            normalized_df["roas"] = normalized_df["roas"].replace([np.inf, -np.inf], np.nan)

    if "spend" in normalized_df.columns and "conversions" in normalized_df.columns:
        if "cpa" not in normalized_df.columns:
            normalized_df["cpa"] = normalized_df["spend"] / normalized_df["conversions"]
            normalized_df["cpa"] = normalized_df["cpa"].replace([np.inf, -np.inf], np.nan)

    return normalized_df

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

with tab1:
    st.subheader("1️⃣ CSV/Excel Feltöltés")
    uploaded_file = st.file_uploader(
        "Válassz CSV vagy Excel fájlt",
        type=["csv", "xlsx", "xls"],
        help="Facebook, Google Ads export",
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

            st.subheader("2️⃣ Automata Facebook Oszlop Felismerés")
            mapping, unmapped = create_mapping_from_facebook(raw_df.columns)
            st.session_state.mapping = mapping

            st.markdown("#### ✅ Leképezett oszlopok (EXAKT):")
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
                platform_hint="Facebook",
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
st.markdown("**HYPER App v6.0** | Hardkódolt Facebook Mapping\n✅ Exakt oszlopneveken alapuló párosítás - NO fuzzy matching!")
