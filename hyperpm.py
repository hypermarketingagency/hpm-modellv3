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

st.set_page_config(
    page_title="HYPER - Marketing Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 📊 CONFIG & SCHEMA (v5.0 – HARDKÓDOLT, PRECÍZ PÁROSÍTÁS)
# ============================================================================

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
        ("clicks", "int", "Kattintások / Interakciók"),
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
    ],
    "optional": [
        ("engagement", "int", "Engagement"),
        ("notes", "string", "Megjegyzések"),
    ],
}

# ✅ VÉGLEGESEN JAVÍTOTT COLUMN_PATTERNS - Facebook specifikus
# Az összes szóköz és speciális karakter kezelve
COLUMN_PATTERNS = {
    # Dátumok
    "date_start": ["jelentés kezdete", "report start", "start date"],
    
    # Kampány adatok
    "campaign_name": ["kampány neve", "campaign name"],
    "campaign_status": ["kampány teljesítése", "campaign performance", "teljesítése"],
    
    # Költség
    "spend": ["elköltött összeg (huf)", "elköltött összeg", "spend", "költség"],
    
    # Megjelenések és elérés
    "impressions": ["megjelenések", "impressions"],
    "reach": ["elérés", "reach"],
    
    # CPC és CTR
    "cpc": ["cpc (összes) (huf)", "cpc"],
    "ctr_percent": ["ctr (átkattintási arány)", "ctr", "átkattintási arány"],
    
    # ROAS
    "roas": ["vásárlási hirdetésmegtérülés", "roas"],
    
    # Gyakoriság
    "frequency": ["gyakoriság", "frequency"],
    
    # ✅ KRITIKUS - PRESÍZS PÁROSÍTÁSOK
    # Ez az oszlop a darabszámot tartalmazza (az eredménytípustól függően)
    "results_count": ["eredmények", "results"],
    
    # Eredményenkénti költség
    "conv_cost": ["eredményenkénti költség", "cost per result"],
    
    # Vásárlások - DARABSZÁM
    "conversions": ["vásárlások", "purchases"],
    
    # Vásárlások konverziós értéke
    "conversion_value": ["vásárlások konverziós értéke", "purchase value"],
    
    # Kosárba helyezések - DARABSZÁM
    "add_to_cart": ["kosárba helyezések", "add to cart"],
    
    # Kosárba helyezés egységnyi költsége
    "add_to_cart_cost": ["kosárba helyezés egységnyi költsége", "add to cart cost"],
    
    # Kosárba helyezések konverziós értéke
    "add_to_cart_value": ["kosárba helyezések konverziós értéke", "add to cart value"],
    
    # Engagement
    "engagement": ["engagement", "post engagement"],
}

# ============================================================================
# 🔧 HELPERS
# ============================================================================

def parse_numeric_value(val):
    """Szöveg → szám konverzió (HUF, tizedes pont vagy vessző)."""
    if pd.isna(val) or val == "" or val == "–" or val == "--" or val == "folyamatban":
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
    """Szöveg → százalék konverzió."""
    if pd.isna(val) or val == "" or val == "–":
        return np.nan
    s = str(val).strip().replace("%", "").replace(",", ".")
    s = s.replace("\u00a0", "").replace(" ", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_date(val):
    """Szöveg → dátum konverzió."""
    if pd.isna(val):
        return None
    date_formats = ["%Y-%m-%d", "%Y.%m.%d", "%d.%m.%Y", "%d-%m-%Y", "%m/%d/%Y"]
    for fmt in date_formats:
        try:
            return pd.to_datetime(val, format=fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(val)
    except Exception:
        return None


def find_matching_column(csv_column, patterns_dict, threshold=75):
    """Fuzzy matching a CSV oszlop és az Unified Field között."""
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
    """Automata felismerés az összes oszlophoz."""
    mapping = {}
    unmapped = []
    for col in df_columns:
        matched_field, score = find_matching_column(col, COLUMN_PATTERNS, threshold=75)
        if matched_field:
            mapping[col] = matched_field
        else:
            unmapped.append(col)
    return mapping, unmapped


def normalize_data(df, mapping, user_adjustments=None, platform_hint=None):
    """Adatok normalizálása az Unified Schema alapján."""
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

    if "platform" not in normalized_df.columns:
        normalized_df["platform"] = platform_hint if platform_hint else "Unknown"

    # Intelligens "results" típus felismerés
    results_type_map = []
    for idx, row in normalized_df.iterrows():
        if not pd.isna(row.get("conversions")) and row.get("conversions", 0) > 0:
            results_type_map.append("Vásárlások")
        elif not pd.isna(row.get("add_to_cart")) and row.get("add_to_cart", 0) > 0:
            results_type_map.append("Kosárba helyezések")
        elif not pd.isna(row.get("engagement")) and row.get("engagement", 0) > 0:
            results_type_map.append("Engagement")
        elif not pd.isna(row.get("results_count")) and row.get("results_count", 0) > 0:
            results_type_map.append("Egyéb eredmények")
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


def validate_data(df):
    """Validáció."""
    issues = []
    mandatory_fields = [f[0] for f in UNIFIED_SCHEMA["mandatory"]]
    for field in mandatory_fields:
        if field not in df.columns:
            issues.append(f"❌ Hiányzik: {field}")
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
        st.markdown("""- ✅ Facebook Ads Manager\n- ✅ Google Ads\n- ⏳ TikTok""")

    platform_hint = st.selectbox(
        "Melyik platformról származik?",
        ["Facebook", "Google Ads", "TikTok", "Ismeretlen"],
        index=0,
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
                    {"CSV Oszlop": csv_col, "Unified Field": unified_col, "Score": int(find_matching_column(csv_col, COLUMN_PATTERNS, threshold=75)[1])}
                    for csv_col, unified_col in sorted(initial_mapping.items())
                ]
                if mapping_display:
                    st.dataframe(pd.DataFrame(mapping_display), use_container_width=True)
                else:
                    st.warning("Nincs automata felismerés :(")

            if unmapped:
                unmapped_cols = st.expander(f"⚠️ Felismeretlen oszlopok ({len(unmapped)})")
                with unmapped_cols:
                    for col in unmapped:
                        st.text(f"• {col}")

            st.subheader("3️⃣ Manuális Korrekció")
            manual_corrections = {}
            all_fields = ["--Nincs--"]
            for section in [UNIFIED_SCHEMA["mandatory"], UNIFIED_SCHEMA["recommended"], UNIFIED_SCHEMA["optional"]]:
                for field in section:
                    all_fields.append(field[0])

            with st.expander("Manuális mapping szerkesztése"):
                for csv_col in raw_df.columns:
                    current_mapping = initial_mapping.get(csv_col, "--Nincs--")
                    new_mapping = st.selectbox(
                        csv_col,
                        all_fields,
                        index=all_fields.index(current_mapping) if current_mapping in all_fields else 0,
                        key=f"manual_map_{csv_col}",
                    )
                    if new_mapping != "--Nincs--":
                        manual_corrections[csv_col] = new_mapping

            if manual_corrections:
                st.session_state.mapping.update(manual_corrections)

            st.subheader("📋 Adatok Előnézete (RAW)")
            st.dataframe(raw_df.head(5), use_container_width=True)

            st.session_state.platform_hint = platform_hint if platform_hint != "Ismeretlen" else None

        except Exception as e:
            st.error(f"❌ Hiba: {str(e)}")

with tab2:
    if st.session_state.uploaded_data is not None:
        st.subheader("✅ Normalizálása & Validálása")
        try:
            normalized_df = normalize_data(
                st.session_state.uploaded_data,
                st.session_state.mapping,
                platform_hint=getattr(st.session_state, "platform_hint", None),
            )
            st.session_state.normalized_data = normalized_df

            validation_issues = validate_data(normalized_df)

            if validation_issues:
                st.warning("### ⚠️ Figyelmezetések")
                for issue in validation_issues:
                    st.warning(issue)
            else:
                st.success("### ✅ Minden OK!")

            st.info(f"Normalizált adatok: {len(normalized_df)} sor × {len(normalized_df.columns)} oszlop")
        except Exception as e:
            st.error(f"❌ Hiba: {str(e)}")
    else:
        st.info("Először töltsd fel az adatokat!")

with tab3:
    if st.session_state.normalized_data is not None:
        st.subheader("📊 Normalizált Adatok Előnézete")
        try:
            df = st.session_state.normalized_data

            col1, col2, col3 = st.columns(3)
            with col1:
                if "spend" in df.columns:
                    st.metric("💰 Teljes Költség", f"{df['spend'].sum():,.0f} HUF")
            with col2:
                if "conversion_value" in df.columns:
                    st.metric("💵 Konverziós Érték", f"{df['conversion_value'].sum():,.0f} HUF")
            with col3:
                if "roas" in df.columns:
                    st.metric("📈 Átlag ROAS", f"{df['roas'].mean():.2f}x")

            st.subheader("Adatok Táblázat")
            df_display = df.copy()

            def fmt_int(x):
                return "" if pd.isna(x) else f"{int(round(x)):,}".replace(",", " ")

            def fmt_huf(x):
                return "" if pd.isna(x) else f"{int(round(x)):,}".replace(",", " ")

            def fmt_pct(x):
                return "" if pd.isna(x) else f"{x:.2f}%"

            # Formázás
            for col in ["conversions", "impressions", "reach", "results_count", "add_to_cart", "engagement"]:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(fmt_int)

            for col in ["spend", "conversion_value", "cpc", "cpa", "conv_cost", "add_to_cart_cost", "add_to_cart_value"]:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(fmt_huf)

            if "ctr_percent" in df_display.columns:
                df_display["ctr_percent"] = df_display["ctr_percent"].apply(fmt_pct)

            st.dataframe(df_display, use_container_width=True)
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
            st.download_button(
                "📥 CSV",
                csv,
                f"hyper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
            )
        with col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Kampanyok")
            st.download_button(
                "📥 Excel",
                buffer.getvalue(),
                f"hyper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.info("Először normalizáld az adatokat!")

st.divider()
st.markdown("**HYPER App v5.0** | Neuromarketing ROAS Predictor\n✅ Hardkódolt, precíz párosítások")
