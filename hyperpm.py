import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from analytics.rollups import filter_segment, segment_summary
from analytics.import_helpers import (
    annotate_source_columns,
    build_dashboard_summary,
    cached_rollups,
    load_uploaded_dataframe,
    render_pulsing_logo,
)
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 🎨 PAGE CONFIG & LOGO
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HYPER - Marketing Campaign Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

logo_url = "https://raw.githubusercontent.com/hypermarketingagency/hpm-modellv3/main/hyper_logo_2025_eredeti.png"

# Header with Logo
col_logo, col_title = st.columns([0.12, 0.88])
with col_logo:
    st.image(logo_url, width=70, use_column_width=False)
with col_title:
    st.markdown("<h1 style='margin-top: -10px;'>HYPER - Marketing Campaign Analyzer</h1>", unsafe_allow_html=True)

st.markdown("**Fázis 1-3: Normalizálás → Analízis → Predikció**")

# ---------------------------------------------------------------------------
# 📊 HARDKÓDOLT MAPPINGEK (Platform-specifikus)
# ---------------------------------------------------------------------------

FACEBOOK_EXACT_MAPPING = {
    "Nap": "date_start",
    "Jelentés kezdete": "date_start",
    "Kampány neve": "campaign_name",
    "Hirdetéssorozat neve": "adset_name",
    "Hirdetés neve": "ad_name",
    "Életkor": "age_group",
    "Kampány teljesítése": "campaign_status",
    "Megjelenítési állapot": "campaign_status",
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
    "Korosztály": "age_group",
    "Nem": "gender",
    "Város": "geo_city",
    "Régió": "geo_region",
    "Megye": "geo_region",
    "Eszköz": "device",
    "Elhelyezés": "placement",
}

FACEBOOK_SKIP_COLUMNS = {
    "Jelentés vége", "Hirdetéssorozat költségkerete", "Hirdetéssorozat költségkeretének típusa",
    "Vége", "Eredmény jelzése", "Hozzárendelés beállítása",
}

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
    "Korosztály": "age_group",
    "Age range": "age_group",
    "Gender": "gender",
    "Nem": "gender",
    "City": "geo_city",
    "Város": "geo_city",
    "Region": "geo_region",
    "Régió": "geo_region",
    "Device": "device",
    "Eszköz": "device",
    "Placement": "placement",
}

GOOGLE_ADS_SKIP_COLUMNS = {
    "Kampány állapota", "Költségkeret", "Költségkeret neve", "Költségkerettípus azonosítója",
    "Pénznem kód", "Állapot", "Állapot okai", "Optimalizálási pontszám", "Kampánytípus",
    "Ajánlattételi stratégia típusa", "Keresési megj. arány", "Eredeti konv. érték",
}

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
    "Age": "age_group",
    "Age group": "age_group",
    "Gender": "gender",
    "City": "geo_city",
    "Location": "geo_city",
    "Region": "geo_region",
    "Régió": "geo_region",
    "Device": "device",
    "Placement": "placement",
}

TIKTOK_SKIP_COLUMNS = {"Currency", "Total of"}

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
        ("adset_name", "string", "Hirdetéssorozat neve"),
        ("ad_name", "string", "Hirdetés neve"),
        ("impressions", "int", "Megjelenések"),
        ("clicks", "int", "Kattintások"),
        ("ctr_percent", "percentage", "CTR (%)"),
        ("cpc", "float", "CPC (HUF)"),
        ("cpa", "float", "CPA (HUF)"),
        ("roas", "float", "ROAS (x)"),
        ("age_group", "string", "Korcsoport"),
        ("gender", "string", "Nem"),
        ("geo_city", "string", "Város"),
        ("geo_region", "string", "Régió/Megye"),
        ("device", "string", "Eszköz"),
        ("placement", "string", "Elhelyezés"),
    ],
}

# ---------------------------------------------------------------------------
# 🔧 HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def clean_excel_structure(df):
    """Excel szerkezeti sorok eltávolítása + Total of sorok szűrése"""
    for idx, row in df.iterrows():
        if any(col in str(row.values) for col in ["Kampány", "Költség", "Campaign name", "Cost"]):
            df_clean = df.iloc[idx+1:].reset_index(drop=True)
            df_clean.columns = row.values
            df_clean = df_clean.loc[:, ~df_clean.columns.str.contains("Unnamed")]
            
            if len(df_clean) > 0 and len(df_clean.columns) > 0:
                first_col = df_clean.columns[0]
                df_col_str = df_clean[first_col].astype(str).str.strip()
                mask = ~(
                    df_col_str.str.lower().isin(['nan', '--', '0', '', 'unnamed: 0']) |
                    df_col_str.str.lower().str.contains('összes|total', regex=True, na=False)
                )
                df_clean = df_clean[mask]
            
            df_clean = df_clean.dropna(how='all')
            
            return df_clean.reset_index(drop=True)
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
    """Platform automatikus detektálása"""
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
    """Platform alapú automata mapping"""
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
    """Adatok normalizálása egységes formátumra"""
    df_filtered = df.copy()
    if len(df_filtered) > 0 and len(df_filtered.columns) > 0:
        first_col = df_filtered.columns[0]
        df_col_str = df_filtered[first_col].astype(str).str.strip()
        
        mask = ~(
            df_col_str.str.lower().isin(['nan', '--', '0', '', 'unnamed: 0']) |
            df_col_str.str.lower().str.contains('összes|total|campaign name', regex=True, na=False)
        )
        df_filtered = df_filtered[mask].reset_index(drop=True)
    
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
                lambda x: int(parse_numeric_value(x)) if not pd.isna(parse_numeric_value(x)) else np.nan
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

    if "spend" in normalized_df.columns and "conversions" in normalized_df.columns:
        if "cpa" not in normalized_df.columns or normalized_df["cpa"].isna().all():
            normalized_df["cpa"] = normalized_df["spend"] / normalized_df["conversions"]
            normalized_df["cpa"] = normalized_df["cpa"].replace([np.inf, -np.inf], np.nan)

    return normalized_df

def format_dataframe_for_display(df):
    """Streamlit megjelenítésre formázás"""
    display_df = df.copy()
    percentage_cols = ["ctr_percent", "conversion_rate"]
    for col in percentage_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "–")
    return display_df


# ---------------------------------------------------------------------------
# 🧠 NEUROMARKETING FUNCTIONS (Fázis 2)
# ---------------------------------------------------------------------------

def analyze_text(text):
    """Szöveg AI-alapú analízise"""
    if not text:
        return 0.5, 0.5, 0, 0.5
    
    text_lower = text.lower()
    
    emotion_words = ['boldogság', 'szeretet', 'bizalom', 'biztonság', 'közösség', 'család',
                     'mosolyog', 'szép', 'amazing', 'fantastic', 'love', 'happy', 'perfect']
    emotion_count = sum(1 for word in emotion_words if word in text_lower)
    emotion_score = min(0.95, 0.3 + (emotion_count * 0.1))
    
    attention_words = ['azonnal', 'most', 'első', 'szenzációs', 'új', 'exkluzív',
                       'revolutionary', 'breakthrough', 'incredible', 'shocking']
    attention_count = sum(1 for word in attention_words if word in text_lower)
    attention_score = min(0.95, 0.3 + (attention_count * 0.08))
    
    urgency_words = ['most', 'azonnal', 'hamar', 'korlátozott', 'csak ma', 'utolsó', 'le fog járni',
                     'limited time', 'hurry', 'urgent']
    urgency_fomo = 1 if any(word in text_lower for word in urgency_words) else 0
    
    personal_words = ['te', 'ön', 'neked', 'nekem', 'mi', 'személyes', 'custom', 'your', 'me', 'personal']
    personal_count = sum(1 for word in personal_words if word in text_lower)
    personalization = min(0.95, 0.2 + (personal_count * 0.12))
    
    return emotion_score, attention_score, urgency_fomo, personalization

def analyze_image(image):
    """Kép analízise (vizuális kontraszt)"""
    try:
        img = Image.open(image).convert('RGB')
        width, height = img.size
        size_score = min(1.0, (width * height) / (1920 * 1080))
        
        pixels = np.array(img.resize((100, 100)))
        contrast = np.std(pixels) / 100
        visual_contrast = min(1.0, contrast)
        
        r_mean, g_mean, b_mean = pixels[:,:,0].mean(), pixels[:,:,1].mean(), pixels[:,:,2].mean()
        color_var = np.var([r_mean, g_mean, b_mean]) / 2000
        color_pop = min(1.0, color_var)
        
        attention_from_image = (size_score * 0.5 + color_pop * 0.5)
        
        return visual_contrast, attention_from_image
    except Exception as e:
        return 0.6, 0.6

@st.cache_resource
def train_model(data):
    """Random Forest modell tanítása"""
    features = ['platform_encoded', 'emotion_score', 'attention_score', 'social_proof',
                'urgency_fomo', 'visual_contrast', 'personalization', 'budget', 'cpc', 'ctr']
    X = data[features].fillna(0)
    y = data['roas'].fillna(0)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    return model, rmse, r2, features

@st.cache_resource
def load_demo_data():
    """Demo adatok generálása"""
    np.random.seed(42)
    n_samples = 500
    data = {
        'platform_encoded': np.random.choice([0,1,2], n_samples),
        'emotion_score': np.random.uniform(0.1, 1.0, n_samples),
        'attention_score': np.random.uniform(0.2, 0.95, n_samples),
        'social_proof': np.random.choice([3,5,10,20], n_samples, p=[0.3,0.4,0.2,0.1]),
        'urgency_fomo': np.random.choice([0,1], n_samples, p=[0.6,0.4]),
        'visual_contrast': np.random.uniform(0.5, 1.0, n_samples),
        'personalization': np.random.uniform(0,1,n_samples),
        'budget': np.random.uniform(100,5000,n_samples),
        'cpc': np.random.uniform(0.5,3.0,n_samples),
        'ctr': np.random.uniform(0.5,5.0,n_samples)/100
    }
    
    neuromarketing_factor = (data['emotion_score']*0.3 + data['attention_score']*0.25 +
                            np.log(data['social_proof']+1)*0.15 + data['urgency_fomo']*0.1 +
                            data['visual_contrast']*0.1 + data['personalization']*0.1)
    data['roas'] = np.clip(2 + neuromarketing_factor*4 + np.log(data['budget'])*0.1 +
                          data['ctr']*20 + np.random.normal(0,0.5,n_samples), 1.0, 10.0)
    
    df = pd.DataFrame(data)
    df['platform'] = df['platform_encoded'].map({0: 'Facebook', 1: 'Google Ads', 2: 'TikTok'})
    return df

# ---------------------------------------------------------------------------
# 💾 SESSION STATE INIT
# ---------------------------------------------------------------------------

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None
if "mapping" not in st.session_state:
    st.session_state.mapping = {}
if "platform" not in st.session_state:
    st.session_state.platform = None
if "normalized_data" not in st.session_state:
    st.session_state.normalized_data = None
if "scores_history" not in st.session_state:
    st.session_state.scores_history = []
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None

# ---------------------------------------------------------------------------
# 🎯 MAIN TAB STRUCTURE
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📥 FÁZIS 1: CSV Import",
    "🖼️ FÁZIS 2: Hirdetés Analyzer",
    "🧠 FÁZIS 3: Model Training",
    "📊 Dashboard"
])

# ---------------------------------------------------------------------------
# TAB 1: FÁZIS 1 - CSV IMPORTER
# ---------------------------------------------------------------------------

with tab1:
    st.markdown("### 📥 Fázis 1: Intelligens CSV/Excel Importer")
    
    st.subheader("1️⃣ CSV/Excel Feltöltés")
    import_mode = st.radio(
        "Import mód",
        ["Egylépcsős feltöltés", "Többlépcsős Facebook (demó + geo)"],
        horizontal=True,
    )

    if import_mode == "Egylépcsős feltöltés":
        uploaded_file = st.file_uploader(
            "Válassz CSV vagy Excel fájlt",
            type=["csv", "xlsx", "xls"],
            help="Facebook, Google Ads vagy TikTok export",
        )

        if uploaded_file:
            try:
                raw_df = load_uploaded_dataframe(uploaded_file, clean_excel_structure)

                st.session_state.uploaded_data = raw_df

                st.success(f"✅ Betöltve: {uploaded_file.name}")
                st.info(f"📊 Sorok: {len(raw_df)}, Oszlopok: {len(raw_df.columns)}")

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

                if unmapped:
                    st.markdown(f"#### ⚠️ Felismeretlen oszlopok ({len(unmapped)}):")
                    for col in unmapped[:5]:
                        st.text(f"• {col}")

                st.subheader("📋 Adatok Előnézete")
                st.dataframe(raw_df.head(3), use_container_width=True)

                if st.button("✅ Normalizálás", type="primary"):
                    pulse_container = render_pulsing_logo(logo_url)
                    try:
                        normalized_df = normalize_data(raw_df, mapping, st.session_state.platform)
                        st.session_state.normalized_data = normalized_df
                        st.success(f"✅ {len(normalized_df)} kampány sikeresen normalizálva!")
                    except Exception as e:
                        st.error(f"❌ Hiba: {str(e)}")
                    finally:
                        pulse_container.empty()

            except Exception as e:
                st.error(f"❌ Hiba: {str(e)}")

    else:
        st.info(
            "ℹ️ Többlépcsős Facebook export esetén több fájlt tölts fel (pl. demográfia + geo). "
            "A rendszer a fájlokat összevonja, de nem végez automatikus összefésülést, hogy elkerülje "
            "a kampány-dátum alapon létrejövő duplikációkat."
        )
        uploaded_files = st.file_uploader(
            "Válassz több CSV vagy Excel fájlt",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            help="Facebook exportok (demográfia + geo)",
        )

        if uploaded_files:
            normalized_dfs = []
            st.subheader("2️⃣ Automata Oszlop Felismerés (több fájl)")

            for idx, file in enumerate(uploaded_files, start=1):
                try:
                    raw_df = load_uploaded_dataframe(file, clean_excel_structure)
                    detected_platform = detect_platform(raw_df.columns)

                    if detected_platform not in ["facebook", "unknown"]:
                        st.warning(f"⚠️ {file.name}: nem Facebook exportnak tűnik.")

                    mapping, unmapped = create_mapping_from_platform(raw_df.columns, "facebook")
                    normalized_df = normalize_data(raw_df, mapping, "facebook")
                    normalized_df = annotate_source_columns(normalized_df, file.name)
                    normalized_dfs.append(normalized_df)

                    with st.expander(f"📄 {idx}. fájl: {file.name}", expanded=idx == 1):
                        st.info(f"📊 Sorok: {len(raw_df)}, Oszlopok: {len(raw_df.columns)}")
                        mapping_display = [
                            {"CSV Oszlop": csv_col, "Unified Field": unified_col}
                            for csv_col, unified_col in sorted(mapping.items())
                        ]
                        if mapping_display:
                            st.markdown("#### ✅ Leképezett oszlopok:")
                            st.dataframe(pd.DataFrame(mapping_display), use_container_width=True)
                        if unmapped:
                            st.markdown(f"#### ⚠️ Felismeretlen oszlopok ({len(unmapped)}):")
                            for col in unmapped[:5]:
                                st.text(f"• {col}")
                        st.markdown("#### 📋 Előnézet")
                        st.dataframe(raw_df.head(3), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ {file.name} betöltési hiba: {str(e)}")

            if st.button("✅ Normalizálás (több fájl)", type="primary"):
                pulse_container = render_pulsing_logo(logo_url)
                if normalized_dfs:
                    combined_df = pd.concat(normalized_dfs, ignore_index=True)
                    st.session_state.normalized_data = combined_df
                    st.success(f"✅ {len(combined_df)} sor sikeresen normalizálva (több fájl)!")
                else:
                    st.warning("⚠️ Nem sikerült feldolgozható fájlt találni.")
                pulse_container.empty()

    if st.session_state.normalized_data is not None:
        st.subheader("📊 Normalizált Adatok")

        if "breakdown_type" in st.session_state.normalized_data.columns:
            st.warning(
                "⚠️ Többlépcsős export esetén az összesítések duplikációt tartalmazhatnak. "
                "Használd a breakdown_type és source_file oszlopokat a szűréshez."
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            if "spend" in st.session_state.normalized_data.columns:
                st.metric("💰 Költség", f"{st.session_state.normalized_data['spend'].sum():,.0f} HUF")
        with col2:
            if "conversion_value" in st.session_state.normalized_data.columns:
                st.metric("💵 Érték", f"{st.session_state.normalized_data['conversion_value'].sum():,.0f} HUF")
        with col3:
            if "roas" in st.session_state.normalized_data.columns and st.session_state.normalized_data["roas"].notna().any():
                st.metric("📈 ROAS", f"{st.session_state.normalized_data['roas'].mean():.2f}x")

        display_df = format_dataframe_for_display(st.session_state.normalized_data)
        st.dataframe(display_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            csv = st.session_state.normalized_data.to_csv(index=False)
            st.download_button("📥 CSV", csv, f"hyper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        with col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                st.session_state.normalized_data.to_excel(writer, index=False, sheet_name="Kampanyok")
            st.download_button(
                "📥 Excel",
                buffer.getvalue(),
                f"hyper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ---------------------------------------------------------------------------
# TAB 2: FÁZIS 2 - HIRDETÉS ANALYZER (TELJES REIMPLEMENTÁCIÓ)
# ---------------------------------------------------------------------------

with tab2:
    st.markdown("### 🖼️ Fázis 2: Hirdetés Neuromarketing Analízis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📸 Hirdetés Kép**")
        uploaded_image = st.file_uploader("Válassz képet", type=["jpg", "jpeg", "png"], key="image_analyzer")
        
        if uploaded_image:
            image_data = Image.open(uploaded_image)
            st.image(image_data, use_column_width=True)
            visual_contrast, attention_img = analyze_image(uploaded_image)
        else:
            visual_contrast, attention_img = 0.6, 0.6
    
    with col2:
        st.markdown("**📝 Hirdetés Szöveg**")
        ad_text = st.text_area("Másold ide a hirdetés szövegét", height=150,
                              placeholder="Pl: 'Csoda módon új megoldás! Csak ma 50% kedvezmény!'", 
                              key="text_analyzer")
        
        if ad_text:
            emotion_txt, attention_txt, urgency_txt, personal_txt = analyze_text(ad_text)
        else:
            emotion_txt, attention_txt, urgency_txt, personal_txt = 0.5, 0.5, 0, 0.5

    if uploaded_image or ad_text:
        st.markdown("---")
        st.subheader("🤖 Automatikus AI Pontozás")
        
        emotion_score = min(0.95, (emotion_txt * 0.7 + attention_img * 0.3))
        attention_score = min(0.95, (attention_txt * 0.6 + visual_contrast * 0.4))
        urgency_fomo = urgency_txt
        personalization = personal_txt
        social_proof_auto = 5
        
        col1, col2 = st.columns(2)
        with col1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("❤️ Emotion Score", f"{emotion_score:.2f}/1.0")
            with col_b:
                st.metric("👁️ Attention Score", f"{attention_score:.2f}/1.0")
        with col2:
            col_c, col_d = st.columns(2)
            with col_c:
                st.metric("🎨 Visual Contrast", f"{visual_contrast:.2f}/1.0")
            with col_d:
                st.metric("🎯 Personalization", f"{personalization:.2f}/1.0")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("👍 Social Proof", f"{social_proof_auto}/20")
        with col2:
            urgency_status = "✅ VAN" if urgency_fomo else "❌ NINCS"
            st.metric("⏰ FOMO/Urgency", urgency_status)

        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Platform**")
            platform_auto = st.selectbox("Platform", ["Facebook", "Google Ads", "TikTok"], key="platform_analyzer")
        
        with col2:
            st.markdown("**Hirdetési Költségvetés (HUF)**")
            budget_auto = st.number_input("Hirdetési Költségvetés", 10000, 5000000, 500000, 10000, key="budget_analyzer")
        
        with col3:
            st.markdown("**Várható CPC (HUF)**")
            cpc_auto = st.number_input("Várható CPC", 10, 1000, 300, 10, key="cpc_analyzer")
        
        ctr_auto = 2.0 + (attention_score * 3)
        
        if st.button("🔮 ROAS Kalkulálás (Auto-Pontok)", type="primary", key="auto_prediction"):
            # ===== AUTO-TRAIN MODELL HA NEM LÉTEZIK =====
            if st.session_state.trained_model is None:
                with st.spinner("🧠 Modell tanítása Demo adatokkal..."):
                    df_demo = load_demo_data()
                    if 'platform' in df_demo.columns:
                        df_demo['platform_encoded'] = df_demo['platform'].map(
                            {'Facebook': 0, 'Google Ads': 1, 'TikTok': 2}
                        ).fillna(0).astype(int)
                    model, rmse, r2, features = train_model(df_demo)
                    st.session_state.trained_model = (model, features)
                    st.success(f"✅ Modell tanítva! R²: {r2:.3f}, RMSE: {rmse:.3f}")
            else:
                model, features = st.session_state.trained_model
            
            plat_enc = {"Facebook": 0, "Google Ads": 1, "TikTok": 2}[platform_auto]
            
            input_data = pd.DataFrame({
                'platform_encoded': [plat_enc],
                'emotion_score': [emotion_score],
                'attention_score': [attention_score],
                'social_proof': [social_proof_auto],
                'urgency_fomo': [int(urgency_fomo)],
                'visual_contrast': [visual_contrast],
                'personalization': [personalization],
                'budget': [budget_auto],
                'cpc': [cpc_auto],
                'ctr': [ctr_auto / 100]
            })
            
            roas_current = model.predict(input_data)[0]
            revenue_current = budget_auto * roas_current
            profit_current = revenue_current - budget_auto
            
            st.markdown("---")
            st.subheader("📊 Jelenlegi Hirdetés - Előrejelzés")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 Várható ROAS", f"{roas_current:.2f}x", delta=f"+{roas_current-1:.2f}x profit")
            with col2:
                st.metric("💵 Bevétel", f"{revenue_current:,.0f} HUF", delta=f"+{profit_current:,.0f} HUF")
            with col3:
                st.metric("🎯 CTR", f"{ctr_auto:.1f}%")
            with col4:
                st.metric("💳 CPC", f"{cpc_auto:.0f} HUF")
            
            st.markdown("---")
            st.subheader("💡 Elemzési Javaslatok")
            
            suggestions = []
            
            if emotion_score < 0.6:
                suggestions.append("📈 **Érzelmi Engagement Növelése**: Erősítsd az érzelmi triggereket (szeretet, közösség, biztonság, boldogság). Potenciális hatás: **+0.5-1.0x ROAS**")
            
            if attention_score < 0.7:
                suggestions.append("👁️ **Figyelem Növelése Az Első 3 Másodpercben**: Használj arcot (azonnal felismerhető), magas kontraszt, mozgás az elején. Potenciális hatás: **+0.3-0.7x ROAS**")
            
            if social_proof_auto < 5:
                suggestions.append("👍 **Social Proof Maximalizálása**: Adj hozzá testimonial videókat, 4.8⭐ értékeléseket, \"500+ elégedett ügyfél\" badget. Potenciális hatás: **+0.4-0.6x ROAS**")
            
            if not urgency_fomo:
                suggestions.append("⏰ **FOMO/Urgency Elem Hozzáadása**: Countdown timer, \"csak 3 db maradt\", \"48 óra akció\", limited offer. Potenciális hatás: **+0.3-0.5x ROAS**")
            
            if visual_contrast < 0.8:
                suggestions.append("🎨 **Vizuális Pop Növelése**: Élénk, kontrasztos színek, before-after képek, animációk. Potenciális hatás: **+0.2-0.4x ROAS**")
            
            if personalization < 0.6:
                suggestions.append("🎯 **Personalizáció Javítása**: Dinamikus szöveg (felhasználó neve), lokális referenciák, targeting finomítása. Potenciális hatás: **+0.2-0.3x ROAS**")
            
            if suggestions:
                for sugg in suggestions:
                    st.info(sugg)
            else:
                st.success("✅ Kiváló paraméterek! Az ad már jól optimalizált!")
            
            st.markdown("---")
            st.subheader("🚀 What-If Szimuláció - Javított Hirdetés")
            st.markdown("**Ha megvalósítod az alább javasolt módosításokat, itt az várható eredmény:**")
            
            emotion_improved = emotion_score
            attention_improved = attention_score
            urgency_improved = urgency_fomo
            personalization_improved = personalization
            visual_improved = visual_contrast
            
            if emotion_score < 0.7:
                emotion_improved = min(0.95, emotion_score + 0.15)
            if attention_score < 0.8:
                attention_improved = min(0.95, attention_score + 0.15)
            if urgency_fomo == 0:
                urgency_improved = 1
            if personalization < 0.6:
                personalization_improved = min(0.95, personalization + 0.15)
            if visual_contrast < 0.8:
                visual_improved = min(0.95, visual_contrast + 0.15)
            
            input_data_improved = pd.DataFrame({
                'platform_encoded': [plat_enc],
                'emotion_score': [emotion_improved],
                'attention_score': [attention_improved],
                'social_proof': [social_proof_auto],
                'urgency_fomo': [int(urgency_improved)],
                'visual_contrast': [visual_improved],
                'personalization': [personalization_improved],
                'budget': [budget_auto],
                'cpc': [cpc_auto],
                'ctr': [ctr_auto / 100]
            })
            
            roas_improved = model.predict(input_data_improved)[0]
            revenue_improved = budget_auto * roas_improved
            profit_improved = revenue_improved - budget_auto
            
            roas_delta = roas_improved - roas_current
            revenue_delta = revenue_improved - revenue_current
            profit_delta = profit_improved - profit_current
            roi_improvement = ((roas_improved - roas_current) / roas_current * 100) if roas_current > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 Javított ROAS", f"{roas_improved:.2f}x",
                         delta=f"+{roas_delta:.2f}x ({roi_improvement:+.1f}%)" if roas_delta != 0 else "Egyezés")
            with col2:
                st.metric("💵 Javított Bevétel", f"{revenue_improved:,.0f} HUF",
                         delta=f"+{revenue_delta:,.0f} HUF" if revenue_delta > 0 else "Nincs változás")
            with col3:
                st.metric("📈 Extra Profit", f"{profit_delta:,.0f} HUF",
                         delta="🎯 Plusz nyereség" if profit_delta > 0 else "Egyezés")
            with col4:
                st.metric("✨ Javítás %", f"{roi_improvement:.1f}%" if roi_improvement > 0 else "—")
            
            st.markdown("---")
            st.subheader("📊 Részletes Összehasonlítás")
            
            comparison_df = pd.DataFrame({
                'Metrika': ['Emotion Score', 'Attention Score', 'Visual Contrast', 'Personalization', 'FOMO/Urgency'],
                'Jelenlegi': [f"{emotion_score:.2f}", f"{attention_score:.2f}", f"{visual_contrast:.2f}",
                            f"{personalization:.2f}", "✅ VAN" if urgency_fomo else "❌ NINCS"],
                'Javított': [f"{emotion_improved:.2f}", f"{attention_improved:.2f}", f"{visual_improved:.2f}",
                           f"{personalization_improved:.2f}", "✅ VAN"],
                'Javulás': [f"+{emotion_improved-emotion_score:.2f}", f"+{attention_improved-attention_score:.2f}",
                          f"+{visual_improved-visual_contrast:.2f}", f"+{personalization_improved-personalization:.2f}",
                          "✅ Hozzáadva" if urgency_improved > urgency_fomo else "—"]
            })
            
            st.table(comparison_df)

# ---------------------------------------------------------------------------
# TAB 3: FÁZIS 3 - MODEL TRAINING
# ---------------------------------------------------------------------------

with tab3:
    st.markdown("### 🧠 Fázis 3: Model Training & Előrejelzés")
    
    st.markdown("**Válaszd ki az adatforrást:**")
    
    col1, col2 = st.columns(2)
    with col1:
        data_source = st.radio("Adatforrás", ["Demo Adatok", "Saját CSV"], key="data_source")
    
    if data_source == "Demo Adatok":
        st.info("📌 Demo adatok használata - ideal teszteléshez")
        df = load_demo_data()
    else:
        st.info("📁 Feltöltsd a saját CSV fájlodat")
        uploaded_train = st.file_uploader("CSV fájl feltöltése", type="csv", key="train_csv")
        
        if uploaded_train:
            df = pd.read_csv(uploaded_train)
            required_cols = ['emotion_score', 'attention_score', 'social_proof', 'urgency_fomo',
                           'visual_contrast', 'personalization', 'budget', 'cpc', 'ctr', 'roas']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Hiányzó oszlopok: {', '.join(missing_cols)}")
                st.stop()
        else:
            st.warning("⚠️ Kérjük, tölts fel egy CSV fájlt!")
            st.stop()
    
    # Platform encoding
    if 'platform' in df.columns:
        df['platform_encoded'] = df['platform'].map(
            {'Facebook': 0, 'Google Ads': 1, 'TikTok': 2}
        ).fillna(0).astype(int)
    else:
        df['platform_encoded'] = 0
        df['platform'] = 'Facebook'
    
    # Model tanítás
    model, rmse, r2, features = train_model(df)
    st.session_state.trained_model = (model, features)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Model Teljesítmény")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("R² Score", f"{r2:.3f}")
    with col2:
        st.metric("RMSE", f"{rmse:.3f}")
    
    st.markdown("---")
    st.subheader("🎯 Manuális ROAS Előrejelzés")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Platform**")
        platform_manual = st.selectbox("Platform", ["Facebook", "Google Ads", "TikTok"], key="platform_manual")
        
        st.markdown("**Emotion Score**")
        emotion_manual = st.slider("Emotion Score", 0.0, 1.0, 0.7, 0.05, key="emotion_manual")
        
        st.markdown("**Attention Score**")
        attention_manual = st.slider("Attention Score", 0.0, 1.0, 0.8, 0.05, key="attention_manual")
    
    with col2:
        st.markdown("**Social Proof**")
        social_proof_manual = st.slider("Social Proof", 0, 20, 5, key="social_proof_manual")
        
        st.markdown("**FOMO/Urgency Element**")
        urgency_manual = st.checkbox("FOMO/Urgency Element", key="urgency_manual")
        
        st.markdown("**Visual Contrast**")
        visual_manual = st.slider("Visual Contrast", 0.0, 1.0, 0.8, 0.05, key="visual_manual")
    
    st.markdown("**Personalizáció & Költségek**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        personal_manual = st.slider("Personalizáció", 0.0, 1.0, 0.6, 0.05, key="personal_manual")
    with col2:
        budget_manual = st.number_input("Hirdetési Költségvetés (HUF)", 10000, 5000000, 500000, 10000, key="budget_manual")
    with col3:
        cpc_manual = st.number_input("Várható CPC (HUF)", 10, 1000, 300, 10, key="cpc_manual")
    
    ctr_manual = st.number_input("Várható CTR (%)", 0.1, 15.0, 2.5, 0.1, key="ctr_manual")
    
    if st.button("🔮 ROAS Előrejelzés", type="primary", key="manual_prediction"):
        plat_enc = {"Facebook": 0, "Google Ads": 1, "TikTok": 2}[platform_manual]
        
        input_data = pd.DataFrame({
            'platform_encoded': [plat_enc],
            'emotion_score': [emotion_manual],
            'attention_score': [attention_manual],
            'social_proof': [social_proof_manual],
            'urgency_fomo': [int(urgency_manual)],
            'visual_contrast': [visual_manual],
            'personalization': [personal_manual],
            'budget': [budget_manual],
            'cpc': [cpc_manual],
            'ctr': [ctr_manual / 100]
        })
        
        roas_pred = model.predict(input_data)[0]
        revenue = budget_manual * roas_pred
        profit = revenue - budget_manual
        
        st.markdown("---")
        st.subheader("📊 Előrejelzés Eredménye")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Várható ROAS", f"{roas_pred:.2f}x", delta=f"+{roas_pred-1:.2f}x profit")
        with col2:
            st.metric("💵 Bevétel", f"{revenue:,.0f} HUF", delta=f"+{profit:,.0f} HUF")
        with col3:
            st.metric("🎯 CTR", f"{ctr_manual:.1f}%")
        with col4:
            st.metric("💳 CPC", f"{cpc_manual:.0f} HUF")

# ---------------------------------------------------------------------------
# TAB 4: DASHBOARD
# ---------------------------------------------------------------------------

with tab4:
    st.markdown("### 📊 Szintetikus Dashboard - Fázis 1-3 Összefoglaló")
    if st.session_state.normalized_data is None:
        st.info("ℹ️ Nincs normalizált adat. Először tölts fel fájlokat a Tab1-ben.")
    else:
        df = st.session_state.normalized_data.copy()

        filter_cols = st.columns(3)
        with filter_cols[0]:
            if "breakdown_type" in df.columns:
                breakdown_options = sorted(df["breakdown_type"].dropna().unique().tolist())
                breakdown_filter = st.multiselect(
                    "Breakdown szűrés",
                    breakdown_options,
                    default=breakdown_options,
                )
            else:
                breakdown_filter = []
        with filter_cols[1]:
            show_full_table = st.checkbox(
                "Teljes tábla megjelenítése (lassú)",
                value=False,
                help="A teljes normalizált táblát rendereli, nagy adatmennyiségnél lassabb lehet.",
            )
        with filter_cols[2]:
            rollup_enabled = st.checkbox(
                "Rollupok számítása (lassabb)",
                value=False,
                help=(
                    "Összesített pivot táblákat számít (havi/regionális/heti/dimenzió), "
                    "nagy adatmennyiségnél lassabb lehet."
                ),
            )

        if breakdown_filter:
            df = df[df["breakdown_type"].isin(breakdown_filter)]

        col1, col2, col3 = st.columns(3)
        with col1:
            if "spend" in df.columns:
                total_spend = df['spend'].sum()
                st.metric("💰 Összes Költség", f"{total_spend:,.0f} HUF")
        with col2:
            if st.session_state.scores_history:
                st.metric("📊 Elemzett Hirdetések", len(st.session_state.scores_history))
        with col3:
            if st.session_state.trained_model:
                st.metric("✅ Modell Status", "🟢 Aktív")

        st.markdown("---")

        st.subheader("📈 Kampányok Összesítése")
        summary = build_dashboard_summary(df)
        st.dataframe(summary, use_container_width=True)

        st.subheader("📋 Mintavétel (gyors nézet)")
        st.dataframe(df.head(200), use_container_width=True)

        if show_full_table:
            st.subheader("📎 Teljes tábla (nagy adatmennyiség)")
            st.dataframe(df, use_container_width=True)

        if rollup_enabled:
            with st.spinner("Rollupok számítása..."):
                rollups = cached_rollups(df)
            with st.expander("📊 Rollup pivotok (havi / regionális / időszaki)"):
                if rollups.get("monthly") is not None:
                    st.markdown("**Havi bontás**")
                    st.dataframe(rollups["monthly"], use_container_width=True)
                if rollups.get("regional") is not None:
                    st.markdown("**Regionális bontás (város/megye)**")
                    st.dataframe(rollups["regional"], use_container_width=True)
                if rollups.get("weekly") is not None:
                    st.markdown("**Heti bontás**")
                    st.dataframe(rollups["weekly"], use_container_width=True)
                if rollups.get("segment_pivot") is not None:
                    st.markdown("**Dimenzió pivot**")
                    st.dataframe(rollups["segment_pivot"], use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Trendek")

        trends_df = df.copy()
        if "date_start" in trends_df.columns:
            trends_df["date_start"] = pd.to_datetime(trends_df["date_start"], errors="coerce")
            trends_df["month_period"] = trends_df["date_start"].dt.to_period("M").astype(str)
            month_labels = {
                1: "Január",
                2: "Február",
                3: "Március",
                4: "Április",
                5: "Május",
                6: "Június",
                7: "Július",
                8: "Augusztus",
                9: "Szeptember",
                10: "Október",
                11: "November",
                12: "December",
            }
            trends_df["month_label"] = trends_df["date_start"].apply(
                lambda x: f"{month_labels.get(x.month, x.month)} {x.year}" if pd.notna(x) else "Ismeretlen"
            )
        else:
            trends_df["month_label"] = "Ismeretlen"
            trends_df["month_period"] = "Ismeretlen"

        def build_options(column_name):
            if column_name not in trends_df.columns:
                return ["Összes"]
            values = (
                trends_df[column_name]
                .dropna()
                .astype(str)
                .replace("nan", pd.NA)
                .dropna()
                .unique()
                .tolist()
            )
            return ["Összes"] + sorted(values)

        geo_column = "geo_region" if "geo_region" in trends_df.columns else "geo_city"
        geo_label = "Régió/Megye" if geo_column == "geo_region" else "Város"
        geo_options = build_options(geo_column)
        age_options = build_options("age_group")
        gender_options = build_options("gender")
        device_options = build_options("device")
        placement_options = build_options("placement")

        min_date = None
        max_date = None
        if "date_start" in trends_df.columns and trends_df["date_start"].notna().any():
            date_series = pd.to_datetime(trends_df["date_start"], errors="coerce")
            min_date = date_series.min().date()
            max_date = date_series.max().date()
        date_picker_available = min_date is not None and max_date is not None

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🔹 Szegmens A**")
            if date_picker_available:
                all_month_a = st.checkbox("Összes hónap", value=True, key="trend_month_all_a")
                month_a = st.date_input(
                    "Hónap",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    disabled=all_month_a,
                    key="trend_month_a",
                )
            else:
                all_month_a = True
                month_a = None
                st.caption("Nincs dátum mező, ezért a hónap szűrés nem elérhető.")
            geo_a = st.selectbox(geo_label, geo_options, key="trend_geo_a")
            age_a = st.selectbox("Korcsoport", age_options, key="trend_age_a")
            gender_a = st.selectbox("Nem", gender_options, key="trend_gender_a")
            device_a = st.selectbox("Eszköz", device_options, key="trend_device_a")
            placement_a = st.selectbox("Elhelyezés", placement_options, key="trend_placement_a")

        with col_b:
            st.markdown("**🔸 Szegmens B**")
            if date_picker_available:
                all_month_b = st.checkbox("Összes hónap", value=True, key="trend_month_all_b")
                month_b = st.date_input(
                    "Hónap",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    disabled=all_month_b,
                    key="trend_month_b",
                )
            else:
                all_month_b = True
                month_b = None
                st.caption("Nincs dátum mező, ezért a hónap szűrés nem elérhető.")
            geo_b = st.selectbox(geo_label, geo_options, key="trend_geo_b")
            age_b = st.selectbox("Korcsoport", age_options, key="trend_age_b")
            gender_b = st.selectbox("Nem", gender_options, key="trend_gender_b")
            device_b = st.selectbox("Eszköz", device_options, key="trend_device_b")
            placement_b = st.selectbox("Elhelyezés", placement_options, key="trend_placement_b")

        def apply_month_filter(df, month_value, include_all):
            if include_all or "date_start" not in df.columns or pd.isna(month_value):
                return df
            month_value = pd.to_datetime(month_value)
            return df[
                (df["date_start"].dt.month == month_value.month)
                & (df["date_start"].dt.year == month_value.year)
            ]

        segment_a = apply_month_filter(trends_df, month_a, all_month_a)
        segment_a = filter_segment(
            segment_a,
            {
                geo_column: geo_a,
                "age_group": age_a,
                "gender": gender_a,
                "device": device_a,
                "placement": placement_a,
            },
        )

        segment_b = apply_month_filter(trends_df, month_b, all_month_b)
        segment_b = filter_segment(
            segment_b,
            {
                geo_column: geo_b,
                "age_group": age_b,
                "gender": gender_b,
                "device": device_b,
                "placement": placement_b,
            },
        )

        summary_a = segment_summary(segment_a)
        summary_b = segment_summary(segment_b)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Szegmens A költség", f"{summary_a['spend']:,.0f} HUF")
        with col2:
            st.metric("Szegmens A ROAS", f"{summary_a['roas']:.2f}x")
        with col3:
            st.metric("Szegmens B költség", f"{summary_b['spend']:,.0f} HUF")
        with col4:
            st.metric("Szegmens B ROAS", f"{summary_b['roas']:.2f}x")

        trend_metric = st.selectbox(
            "Trend metrika",
            ["spend", "conversion_value", "conversions", "impressions", "clicks", "roas"],
            key="trend_metric",
        )

        if "month_period" in trends_df.columns:
            def metric_series(df_segment, label):
                if trend_metric in df_segment.columns:
                    agg_func = "mean" if trend_metric == "roas" else "sum"
                    grouped = (
                        df_segment.groupby("month_period")[trend_metric]
                        .agg(agg_func)
                        .rename(label)
                    )
                else:
                    grouped = pd.Series(dtype=float)
                return grouped

            series_a = metric_series(segment_a, "Szegmens A")
            series_b = metric_series(segment_b, "Szegmens B")
            trend_data = pd.concat([series_a, series_b], axis=1).fillna(0)
            if not trend_data.empty:
                st.line_chart(trend_data)
            else:
                st.info("Nincs elég adat a trend charthoz.")

        with st.expander("🔮 Forecasting (baseline, Prophet/SARIMAX előkészítés)", expanded=False):
            st.markdown(
                "Egyszerű baseline előrejelzés (utolsó 3 hónap átlaga), "
                "amíg a Prophet/SARIMAX réteg érkezik."
            )
            forecast_horizon = st.slider("Előrejelzés hónapok száma", 1, 6, 3, key="forecast_horizon")
            target_segment = st.selectbox("Szegmens", ["Szegmens A", "Szegmens B", "Mindkettő"], key="forecast_segment")

            def build_forecast(series):
                if series.empty:
                    return None
                series = series.copy()
                series.index = pd.PeriodIndex(series.index, freq="M")
                recent = series.tail(3)
                baseline = recent.mean() if not recent.empty else series.mean()
                future_periods = pd.period_range(start=series.index.max() + 1, periods=forecast_horizon, freq="M")
                forecast = pd.Series([baseline] * forecast_horizon, index=future_periods)
                combined = pd.DataFrame({
                    "Tény": series.astype(float),
                    "Előrejelzés": forecast.astype(float),
                })
                combined.index = combined.index.astype(str)
                return combined

            if target_segment in ["Szegmens A", "Mindkettő"]:
                forecast_a = build_forecast(series_a)
                if forecast_a is not None:
                    st.markdown("**Szegmens A előrejelzés**")
                    st.line_chart(forecast_a)
                else:
                    st.info("Szegmens A: nincs elég adat előrejelzéshez.")

            if target_segment in ["Szegmens B", "Mindkettő"]:
                forecast_b = build_forecast(series_b)
                if forecast_b is not None:
                    st.markdown("**Szegmens B előrejelzés**")
                    st.line_chart(forecast_b)
                else:
                    st.info("Szegmens B: nincs elég adat előrejelzéshez.")
    
    if st.session_state.scores_history:
        st.subheader("🖼️ Hirdetések Scoring Historia")
        
        scores_df = pd.DataFrame(st.session_state.scores_history)
        scores_df = scores_df[['timestamp', 'platform', 'emotion_score', 'attention_score', 'visual_contrast', 'personalization']]
        st.dataframe(scores_df, use_container_width=True)

st.divider()

with st.expander("ℹ️ Hogyan működik a modell?"):
    st.markdown("""
    ### Random Forest Algoritmus
    
    Ez a modell **100 döntési fát** használ szavazási rendszerben:
    
    - Mindegyik fa más szöget lát az adatokra
    - Szavazatot ad a ROAS-ra
    - A végeredmény az összes fa átlaga
    
    ### Neuromarketing Tényezők
    
    - **Emotion Score**: Érzelmi engagement (0-1) - Az agy döntéseit érzelmek hajtják
    - **Attention Score**: Figyelem (0-1) - Az első 3 másodperc kritikus
    - **Social Proof**: Vélemények (0-20) - Emberek másolatnak
    - **FOMO/Urgency**: Sietség - Csökkenti a döntési időt
    - **Visual Contrast**: Szín (0-1) - Magas kontraszt = figyelem
    - **Personalization**: Egyéniesítés (0-1) - Név, lokálitás = magasabb CTR
    - **Budget**: Költségvetés - Nagyobb adspend = több impresszió
    - **CPC**: Kattintás ára - Platform határozza meg
    - **CTR**: Kattintási arány - Jó ad = 2-5% CTR
    """)

st.markdown(
    "<p style='text-align: center; font-size: 12px;'><strong>HYPER App v9.2.5</strong> | Fázis 1-3 Integráció<br>✅ CSV Import • 🖼️ Hirdetés Analyzer • 🧠 Model Training • 📊 Dashboard</p>",
    unsafe_allow_html=True
)
