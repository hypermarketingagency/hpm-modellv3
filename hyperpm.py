import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 🎨 PAGE CONFIG & LOGO
# ============================================================================
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
    st.markdown(
        "<h1 style='margin-top: -10px'>HYPER - Marketing Campaign Analyzer</h1>",
        unsafe_allow_html=True
    )
    st.markdown("**Fázis 1-3** | Normalizálás • Analízis • Predikció")

# ============================================================================
# 📋 PLATFORM MAPPING DICTIONARIES
# ============================================================================
FACEBOOK_EXACT_MAPPING = {
    "Jelents kezdete": "date_start",
    "Kampny neve": "campaign_name",
    "Kampny teljestse": "campaign_status",
    "Elklttt sszeg (HUF)": "spend",
    "Megjelensek": "impressions",
    "Elrs": "reach",
    "CPC sszesen (HUF)": "cpc",
    "CTR (tkattintsi arny)": "ctr_percent",
    "Vsrlsi hirdetsmegtrls (ROAS)": "roas",
    "Gyakorisg": "frequency",
    "Eredmnyek": "results_count",
    "Eredmnyenknti kltsg": "conv_cost",
    "Vsrlsok": "conversions",
    "Vsrlsok konverzis rtke": "conversion_value",
    "Kosrba helyezsek": "add_to_cart",
    "Kosrba helyezs egysgnyi kltsge (HUF)": "add_to_cart_cost",
    "Kosrba helyezsek konverzis rtke": "add_to_cart_value",
    "Engagement": "engagement",
}

FACEBOOK_SKIP_COLUMNS = [
    "Jelents vge",
    "Hirdetssorozat kltsgkerete",
    "Hirdetssorozat kltsgkeretnek tpusa",
    "Vge",
    "Eredmny jelzse",
    "Hozzrendels belltsa",
]

GOOGLE_ADS_EXACT_MAPPING = {
    "Kampny": "campaign_name",
    "Kampny llapota": "campaign_status",
    "Kltsg": "spend",
    "Interakcik": "clicks",
    "Interakcis arny": "ctr_percent",
    "Konverzik": "conversions",
    "Konverzis rtk": "conversion_value",
    "Konverzis rtk/kltsg": "roas",
    "tl. CPC": "cpc",
    "Megjel.": "impressions",
    "Kltsg/konv.": "cpa",
    "tl. kltsg": "avg_cost",
    "Konv. arny": "conversion_rate",
}

GOOGLE_ADS_SKIP_COLUMNS = [
    "Kampny llapota",
    "Kltsgkeret",
    "Kltsgkeret neve",
    "Kltsgkeret tpusa azonostja",
    "Pnznem kd",
    "llpot",
    "llpot okai",
    "Optimalizlsi pontszm",
    "Kampny tpusa",
    "Ajnlattteli stratgia tpusa",
    "Keressi megj. arny",
    "Eredeti konv. rtk",
]

TIKTOK_EXACT_MAPPING = {
    "Campaign name": "campaign_name",
    "Primary status": "campaign_status",
    "Cost": "spend",
    "Impressions": "impressions",
    "Clicks (destination clicks)": "clicks",
    "CTR (destination)": "ctr_percent",
    "Purchases (website conversions)": "conversions",
    "Purchase value (website conversion value)": "conversion_value",
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

TIKTOK_SKIP_COLUMNS = [
    "Currency",
    "Total of...",
]

UNIFIED_SCHEMA = {
    "mandatory": [
        ("date_start", "date", "Jelents kezdete dtum"),
        ("campaign_name", "string", "Kampny neve"),
        ("platform", "string", "Platform (Facebook/Google Ads/TikTok)"),
        ("campaign_status", "string", "Kampny sttusza"),
        ("spend", "float", "Elklttt sszeg (HUF)"),
        ("conversions", "int", "Konverzik szma"),
        ("conversion_value", "float", "Konverzis rtk (HUF)"),
    ],
    "recommended": [
        ("impressions", "int", "Megjelensek"),
        ("clicks", "int", "Kattintsok"),
        ("ctr_percent", "percentage", "CTR %"),
        ("cpc", "float", "CPC (HUF)"),
        ("cpa", "float", "CPA (HUF)"),
        ("roas", "float", "ROAS (x)"),
    ]
}

# ============================================================================
# 🔧 HELPER FUNCTIONS
# ============================================================================

def clean_excel_structure(df):
    """Excel szerkezeti sorok eltvoltsa - Total of sorok szrse"""
    for idx, row in df.iterrows():
        if any(col in str(row.values) for col in ["Kampny", "Kltsg", "Campaign name", "Cost"]):
            df_clean = df.iloc[idx+1:].reset_index(drop=True)
            df_clean.columns = row.values
            df_clean = df_clean.loc[:, ~df_clean.columns.str.contains("Unnamed")]
            
            if len(df_clean) > 0 and len(df_clean.columns) > 0:
                first_col = df_clean.columns[0]
                df_col = df_clean[first_col].astype(str).str.strip()
                mask = (
                    df_col.str.lower().isin(["nan", "--", "0", "", "unnamed"]) |
                    df_col.str.lower().str.contains("sszesen|total", regex=True, na=False)
                )
                df_clean = df_clean[~mask]
            df_clean = df_clean.dropna(how='all')
            return df_clean.reset_index(drop=True)
    return df

def parse_numeric_value(val):
    """Szm rtelmezs - 1000,5 vagy 1.000,50 formtumbl"""
    if pd.isna(val) or val == "" or val == "-" or val == "--" or val == "folyamatban":
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    
    s = str(val).strip().replace('\u00a0', '').replace(' ', '').replace(',', '.')
    if ',' in s and '.' in s:
        s = s.replace('.', '').replace(',', '.')
    else:
        s = s.replace(',', '.')
    
    try:
        return float(s)
    except:
        return np.nan

def parse_percentage_value(val):
    """Szzalk rtelmezs"""
    if pd.isna(val) or val == "" or val == "-" or val == "10" or val == "--":
        return np.nan
    
    s = str(val).strip().replace(',', '.').replace(',', '.')
    try:
        num = float(s)
        if 0 <= num <= 1:
            return num * 100
        return num
    except:
        return np.nan

def parse_date(val):
    """Dtum rtelmezs"""
    if pd.isna(val):
        return None
    
    formats = ['%Y-%m-%d', '%Y.%m.%d', '%d.%m.%Y', '%d-%m-%Y', '%m%d%Y']
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
    """Platform automatikus detektlsa"""
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
    """Platform alap automata mapping"""
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
    """Adatok normalizlsa egysges formtumra"""
    df_filtered = df.copy()
    
    if len(df_filtered) > 0 and len(df_filtered.columns) > 0:
        first_col = df_filtered.columns[0]
        df_col = df_filtered[first_col].astype(str).str.strip()
        mask = (
            df_col.str.lower().isin(["nan", "--", "0", "", "unnamed"]) |
            df_col.str.lower().str.contains("sszesen|total|campaign name", regex=True, na=False)
        )
        df_filtered = df_filtered[~mask].reset_index(drop=True)
    
    normalized_df = pd.DataFrame()
    
    for csv_col, unified_col in mapping.items():
        if csv_col not in df_filtered.columns:
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
        
        field_name, field_type, field_desc = field_info
        raw_data = df[csv_col]
        
        if field_type == "percentage":
            normalized_df[field_name] = raw_data.apply(parse_percentage_value)
        elif field_type == "float":
            normalized_df[field_name] = raw_data.apply(parse_numeric_value)
        elif field_type == "int":
            normalized_df[field_name] = raw_data.apply(lambda x: int(parse_numeric_value(x)) if not pd.isna(parse_numeric_value(x)) else np.nan)
        elif field_type == "date":
            normalized_df[field_name] = raw_data.apply(parse_date)
        elif field_type == "string":
            normalized_df[field_name] = raw_data.astype(str)
        else:
            normalized_df[field_name] = raw_data
    
    if platform not in normalized_df.columns:
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
    """Streamlit megjelentsre formzs"""
    display_df = df.copy()
    percentage_cols = ["ctr_percent", "conversion_rate"]
    
    for col in percentage_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    
    return display_df

def analyze_text(text):
    """Szveg AI-alap analzise"""
    if not text:
        return 0.5, 0.5, 0, 0.5
    
    text_lower = text.lower()
    
    # Emotion Score
    emotion_words = ["boldogsg", "szeretet", "bizalom", "biztonsg", "kzssg", "csald", "mosolyog", "szp", 
                     "amazing", "fantastic", "love", "happy", "perfect"]
    emotion_count = sum(1 for word in emotion_words if word in text_lower)
    emotion_score = min(0.95, 0.3 + emotion_count * 0.1)
    
    # Attention Score
    attention_words = ["azonnal", "most", "els", "szenzcis", "j", "exkluzv", 
                       "revolutionary", "breakthrough", "incredible", "shocking"]
    attention_count = sum(1 for word in attention_words if word in text_lower)
    attention_score = min(0.95, 0.3 + attention_count * 0.08)
    
    # Urgency/FOMO
    urgency_words = ["most", "azonnal", "hamar", "korltozott", "csak ma", "utols", "le fog jrni",
                     "limited time", "hurry", "urgent"]
    urgency_fomo = 1 if any(word in text_lower for word in urgency_words) else 0
    
    # Personalization
    personal_words = ["te", "n", "neked", "nekem", "mi", "szemlyes", "custom", "your", "me", "personal"]
    personal_count = sum(1 for word in personal_words if word in text_lower)
    personalization = min(0.95, 0.2 + personal_count * 0.12)
    
    return emotion_score, attention_score, urgency_fomo, personalization

def analyze_image(image):
    """Kp analzise - vizulis kontraszt"""
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
        attention_from_image = size_score * 0.5 + color_pop * 0.5
        
        return visual_contrast, attention_from_image
    except Exception as e:
        return 0.6, 0.6

@st.cache_resource
def train_model(data):
    """Random Forest modell tantsa"""
    features = ["platform_encoded", "emotion_score", "attention_score", "social_proof", "urgency_fomo", "visual_contrast", "personalization", "budget", "cpc", "ctr"]
    
    X = data[features].fillna(0)
    y = data["roas"].fillna(0)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    return model, rmse, r2, features

@st.cache_resource
def load_demo_data():
    """Demo adatok generlsa"""
    np.random.seed(42)
    n_samples = 500
    
    data = {
        "platform_encoded": np.random.choice([0, 1, 2], n_samples),
        "emotion_score": np.random.uniform(0.1, 1.0, n_samples),
        "attention_score": np.random.uniform(0.2, 0.95, n_samples),
        "social_proof": np.random.choice([3, 5, 10, 20], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        "urgency_fomo": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        "visual_contrast": np.random.uniform(0.5, 1.0, n_samples),
        "personalization": np.random.uniform(0, 1, n_samples),
        "budget": np.random.uniform(100, 5000, n_samples),
        "cpc": np.random.uniform(0.5, 3.0, n_samples),
        "ctr": np.random.uniform(0.5, 5.0, n_samples) / 100,
    }
    
    neuromarketing_factor = (
        data["emotion_score"] * 0.3 +
        data["attention_score"] * 0.25 +
        np.log(data["social_proof"] + 1) * 0.15 +
        data["urgency_fomo"] * 0.1 +
        data["visual_contrast"] * 0.1 +
        data["personalization"] * 0.1
    )
    
    data["roas"] = np.clip(
        2 + neuromarketing_factor * 4 + np.log(data["budget"] + 0.1) * data["ctr"] * 20 + np.random.normal(0, 0.5, n_samples),
        1.0, 10.0
    )
    
    df = pd.DataFrame(data)
    df["platform"] = df["platform_encoded"].map({0: "Facebook", 1: "Google Ads", 2: "TikTok"})
    
    return df

# ============================================================================
# 🔄 SESSION STATE INITIALIZATION
# ============================================================================
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

# ============================================================================
# 📑 MAIN TAB INTERFACE
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["FÁZIS 1: CSV Import", "FÁZIS 2: Hirdetés Analyzer", "FÁZIS 3: Model Training", "Dashboard"])

# ============================================================================
# TAB 1: CSV IMPORT & NORMALIZATION
# ============================================================================
with tab1:
    st.markdown("## Fázis 1: Intelligens CSV/Excel Importer")
    
    st.subheader("1️⃣ CSV/Excel Feltöltés")
    uploaded_file = st.file_uploader(
        "Válassz CSV vagy Excel fájlt",
        type=["csv", "xlsx", "xls"],
        help="Facebook, Google Ads vagy TikTok export"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            else:
                raw_df = pd.read_excel(uploaded_file)
                if uploaded_file.name.endswith(('.xlsx', '.xls')):
                    raw_df = clean_excel_structure(raw_df)
            
            st.session_state.uploaded_data = raw_df
            st.success(f"✅ Betöltve: {uploaded_file.name}")
            st.info(f"Sorok: {len(raw_df)}, Oszlopok: {len(raw_df.columns)}")
            
            # Platform detection
            detected_platform = detect_platform(raw_df.columns)
            st.session_state.platform = detected_platform
            
            if detected_platform == "unknown":
                st.warning("⚠️ Nem sikerült felismerni a platform típusát. Válassz manuálisan:")
                selected_platform = st.selectbox("Platform", ["Facebook", "Google Ads", "TikTok"])
                platform_map = {"Facebook": "facebook", "Google Ads": "google_ads", "TikTok": "tiktok"}
                st.session_state.platform = platform_map[selected_platform]
            else:
                platform_names = {"facebook": "Facebook", "google_ads": "Google Ads", "tiktok": "TikTok"}
                platform_name = platform_names[detected_platform]
                st.success(f"✅ Felismert platform: **{platform_name}**")
            
            st.subheader("2️⃣ Automata Oszlop Felismerés")
            mapping, unmapped = create_mapping_from_platform(raw_df.columns, st.session_state.platform)
            st.session_state.mapping = mapping
            
            st.markdown("### Leképezett oszlopok:")
            mapping_display = [{"CSV Oszlop": csv_col, "Unified Field": unified_col} for csv_col, unified_col in sorted(mapping.items())]
            if mapping_display:
                st.dataframe(pd.DataFrame(mapping_display), use_container_width=True)
            
            if unmapped:
                st.markdown(f"### ⚠️ Felismeretlen oszlopok ({len(unmapped)}):")
                for col in unmapped[:5]:
                    st.text(col)
            
            st.subheader("3️⃣ Adatok Előnézete")
            st.dataframe(raw_df.head(3), use_container_width=True)
            
            if st.button("🔄 Normalizálás", type="primary"):
                try:
                    normalized_df = normalize_data(raw_df, mapping, st.session_state.platform)
                    st.session_state.normalized_data = normalized_df
                    st.success(f"✅ {len(normalized_df)} kampány sikeresen normalizálva!")
                except Exception as e:
                    st.error(f"❌ Hiba sikerült: {e}")
            
            # Display normalized data
            if st.session_state.normalized_data is not None:
                st.subheader("4️⃣ Normalizált Adatok")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if "spend" in st.session_state.normalized_data.columns:
                        st.metric("💰 Költség", f"{st.session_state.normalized_data['spend'].sum():,.0f} HUF")
                
                with col2:
                    if "conversion_value" in st.session_state.normalized_data.columns:
                        st.metric("📊 Érték", f"{st.session_state.normalized_data['conversion_value'].sum():,.0f} HUF")
                
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
                    st.download_button("📥 Excel", buffer.getvalue(), f"hyper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        except Exception as e:
            st.error(f"❌ Hiba: {e}")

# ============================================================================
# TAB 2: HIRDETÉS ANALYZER
# ============================================================================
with tab2:
    st.markdown("## Fázis 2: Hirdetés Neuromarketing Analízis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🖼️ Hirdetés Kép")
        uploaded_image = st.file_uploader("Válassz képet", type=["jpg", "jpeg", "png"], key="image_analyzer")
        
        if uploaded_image:
            image_data = Image.open(uploaded_image)
            st.image(image_data, use_column_width=True)
            visual_contrast, attention_img = analyze_image(uploaded_image)
        else:
            visual_contrast, attention_img = 0.6, 0.6
    
    with col2:
        st.markdown("### 📝 Hirdetés Szöveg")
        ad_text = st.text_area(
            "Másold ide a hirdetés szövegét",
            height=150,
            placeholder="Pl: Csodálatos módon jó megoldás! Csak ma 50% kedvezmény!",
            key="text_analyzer"
        )
        
        if ad_text:
            emotion_txt, attention_txt, urgency_txt, personal_txt = analyze_text(ad_text)
        else:
            emotion_txt, attention_txt, urgency_txt, personal_txt = 0.5, 0.5, 0, 0.5
    
    if uploaded_image or ad_text:
        st.markdown("---")
        st.subheader("✅ Automatikus AI Pontozás")
        
        emotion_score = min(0.95, emotion_txt * 0.7 + attention_img * 0.3)
        attention_score = min(0.95, attention_txt * 0.6 + visual_contrast * 0.4)
        urgency_fomo = urgency_txt
        personalization = personal_txt
        social_proof_auto = 5
        
        col1, col2 = st.columns(2)
        with col1:
            cola, colb = st.columns(2)
            with cola:
                st.metric("😊 Emotion Score", f"{emotion_score:.2f}/1.0")
            with colb:
                st.metric("👁️ Attention Score", f"{attention_score:.2f}/1.0")
        
        with col2:
            colc, cold = st.columns(2)
            with colc:
                st.metric("🎨 Visual Contrast", f"{visual_contrast:.2f}/1.0")
            with cold:
                st.metric("🎯 Personalization", f"{personalization:.2f}/1.0")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📱 Platform**")
            platform_auto = st.selectbox("Platform", ["Facebook", "Google Ads", "TikTok"], key="platform_analyzer")
        
        with col2:
            st.markdown("**💵 Hirdetési Költségvetés (HUF)**")
            budget_auto = st.number_input("Hirdetési Költségvetés", 10000, 5000000, 500000, 10000, key="budget_analyzer")
        
        with col3:
            st.markdown("**💰 Várható CPC (HUF)**")
            cpc_auto = st.number_input("Várható CPC", 10, 1000, 300, 10, key="cpc_analyzer")
        
        ctr_auto = 2.0 + (attention_score * 3)
        
        if st.button("💡 ROAS Kalkulus", type="primary", key="auto_prediction"):
            score_entry = {
                "timestamp": datetime.now(),
                "emotion_score": emotion_score,
                "attention_score": attention_score,
                "visual_contrast": visual_contrast,
                "personalization": personalization,
                "urgency_fomo": urgency_fomo,
                "social_proof": social_proof_auto,
                "platform": platform_auto,
                "budget": budget_auto,
                "cpc": cpc_auto,
                "ctr": ctr_auto,
            }
            st.session_state.scores_history.append(score_entry)
            st.success("✅ Scoring mentve! Használd a Fázis 3-ban a Model Training-hez.")

# ============================================================================
# TAB 3: MODEL TRAINING
# ============================================================================
with tab3:
    st.markdown("## Fázis 3: Model Training & Predikció")
    st.markdown("Válaszd ki az adatforrást:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_source = st.radio("Adatforrás", ["Demo Adatok", "Saját CSV"], key="data_source")
        
        if data_source == "Demo Adatok":
            st.info("Demo adatok használata - ideális teszteléshez")
            df = load_demo_data()
        else:
            st.info("Feltöltsd a saját CSV fájlodat")
            uploaded_train = st.file_uploader("CSV fájl feltöltése", type="csv", key="train_csv")
            
            if uploaded_train:
                df = pd.read_csv(uploaded_train)
                required_cols = ["emotion_score", "attention_score", "social_proof", "urgency_fomo", "visual_contrast", "personalization", "budget", "cpc", "ctr", "roas"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Hiányzó oszlopok: {', '.join(missing_cols)}")
                    st.stop()
            else:
                st.warning("Kérjük, tölts fel egy CSV fájlt!")
                st.stop()
    
    with col2:
        if "platform" in df.columns:
            df["platform_encoded"] = df["platform"].map({"Facebook": 0, "Google Ads": 1, "TikTok": 2}).fillna(0).astype(int)
        else:
            df["platform_encoded"] = 0
            df["platform"] = "Facebook"
    
    if st.button("🚀 Modell Tanítása", type="primary"):
        model, rmse, r2, features = train_model(df)
        st.session_state.trained_model = (model, features)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Model Teljesítmény")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
        with col2:
            st.metric("RMSE", f"{rmse:.3f}")
    
    st.markdown("---")
    st.subheader("🔮 Manuális ROAS Előrejelzés")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Platform**")
        platform_manual = st.selectbox("Platform", ["Facebook", "Google Ads", "TikTok"], key="platform_manual")
        
        st.markdown("**😊 Emotion Score**")
        emotion_manual = st.slider("Emotion Score", 0.0, 1.0, 0.7, 0.05, key="emotion_manual")
        
        st.markdown("**👁️ Attention Score**")
        attention_manual = st.slider("Attention Score", 0.0, 1.0, 0.8, 0.05, key="attention_manual")
    
    with col2:
        st.markdown("**👥 Social Proof**")
        social_proof_manual = st.slider("Social Proof", 0, 20, 5, key="social_proof_manual")
        
        st.markdown("**⏰ FOMO/Urgency Element**")
        urgency_manual = st.checkbox("FOMO/Urgency Element", key="urgency_manual")
        
        st.markdown("**🎨 Visual Contrast**")
        visual_manual = st.slider("Visual Contrast", 0.0, 1.0, 0.8, 0.05, key="visual_manual")
    
    st.markdown("**🎯 Personalización & Költségek**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        personal_manual = st.slider("Personalizáció", 0.0, 1.0, 0.6, 0.05, key="personal_manual")
    
    with col2:
        budget_manual = st.number_input("Hirdetési Költségvetés (HUF)", 10000, 5000000, 500000, 10000, key="budget_manual")
    
    with col3:
        cpc_manual = st.number_input("Várható CPC (HUF)", 10, 1000, 300, 10, key="cpc_manual")
    
    ctr_manual = st.number_input("Várható CTR %", 0.1, 15.0, 2.5, 0.1, key="ctr_manual")
    
    if st.button("🎯 ROAS Előrejelzés", type="primary", key="manual_prediction"):
        if st.session_state.trained_model:
            model, features = st.session_state.trained_model
            
            plat_enc = {"Facebook": 0, "Google Ads": 1, "TikTok": 2}[platform_manual]
            
            input_data = pd.DataFrame({
                "platform_encoded": [plat_enc],
                "emotion_score": [emotion_manual],
                "attention_score": [attention_manual],
                "social_proof": [social_proof_manual],
                "urgency_fomo": [int(urgency_manual)],
                "visual_contrast": [visual_manual],
                "personalization": [personal_manual],
                "budget": [budget_manual],
                "cpc": [cpc_manual],
                "ctr": [ctr_manual / 100],
            })
            
            roas_pred = model.predict(input_data)[0]
            revenue = budget_manual * roas_pred
            profit = revenue - budget_manual
            
            st.markdown("---")
            st.subheader("📈 Előrejelzés Eredménye")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Várható ROAS", f"{roas_pred:.2f}x", delta=f"{roas_pred-1:.2f}x profit")
            
            with col2:
                st.metric("💰 Bevétel", f"{revenue:,.0f} HUF", delta=f"{profit:,.0f} HUF")
            
            with col3:
                st.metric("🔗 CTR", f"{ctr_manual:.1f}%")
            
            with col4:
                st.metric("💵 CPC", f"{cpc_manual:.0f} HUF")
        else:
            st.warning("⚠️ Kérjük, tanítsd meg a modellt először!")

# ============================================================================
# TAB 4: DASHBOARD
# ============================================================================
with tab4:
    st.markdown("## 📊 Szintetikus Dashboard - Fázis 1-3 összefoglalás")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.normalized_data is not None:
            total_spend = st.session_state.normalized_data["spend"].sum()
            st.metric("💰 Összes Költség", f"{total_spend:,.0f} HUF")
    
    with col2:
        if st.session_state.scores_history:
            st.metric("📊 Elemzett Hirdetések", len(st.session_state.scores_history))
    
    with col3:
        if st.session_state.trained_model:
            st.metric("🤖 Modell Status", "✅ Aktív")
    
    st.markdown("---")
    
    if st.session_state.normalized_data is not None:
        st.subheader("📈 Kampányok Összestése")
        summary = st.session_state.normalized_data.groupby("platform").agg({
            "spend": "sum",
            "conversions": "sum",
            "conversion_value": "sum",
            "impressions": "sum",
        }).round(2)
        st.dataframe(summary, use_container_width=True)
    
    if st.session_state.scores_history:
        st.subheader("📋 Hirdetések Scoring Historia")
        scores_df = pd.DataFrame(st.session_state.scores_history)
        scores_df = scores_df[["timestamp", "platform", "emotion_score", "attention_score", "visual_contrast", "personalization"]]
        st.dataframe(scores_df, use_container_width=True)
    
    st.divider()
    
    with st.expander("❓ Hogyan működik a modell?"):
        st.markdown("""
        #### 🧠 Random Forest Regresszió
        - Ez a modell **100 döntési fa** használ szavazási rendszerben
        - Mindegyik fa más szöget lát az adatokra
        - Szavazatot ad a ROAS-ra
        - A végeredmény az összes fa átlaga
        
        #### 📊 Neuromarketing Metrikák
        - **Emotion Score** (0-1): Érzelmi engagement - Az agy döntéseit érzelmek hajtják
        - **Attention Score** (0-1): Figyelem - Az első 3 másodperc kritikus
        - **Social Proof** (0-20): Vélemények - Emberek másolatnak
        - **FOMO/Urgency**: Sietség - Csökkenti a döntési időt
        - **Visual Contrast** (0-1): Szín - Magas kontraszt = figyelem
        - **Personalization** (0-1): Egyniestés - Név, lokális = magasabb CTR
        - **Budget**: Költségvetés - Nagyobb adspend = több impresszió
        - **CPC**: Kattintás ára - Platform határozza meg
        - **CTR**: Kattintási arány - Jó ad = 2-5% CTR
        """)
    
    st.markdown(
        "<p style='text-align: center; font-size: 12px'><strong>HYPER App v9.2</strong> | Fázis 1-3 Integráció<br>CSV Import • 🖼️ Hirdetés Analyzer • 🧠 Model Training • 📊 Dashboard</p>",
        unsafe_allow_html=True
    )
