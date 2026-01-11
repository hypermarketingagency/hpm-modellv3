diff --git a/hyperpm.py b/hyperpm.py
index 243abd89844b071126a64c63505d85b484627122..d5dafc29fd0b2514facc924798a8a4c5dcd4b09f 100644
--- a/hyperpm.py
+++ b/hyperpm.py
@@ -1,149 +1,176 @@
 import streamlit as st
 import pandas as pd
 import numpy as np
 from datetime import datetime
 import io
 from PIL import Image
 from sklearn.ensemble import RandomForestRegressor
 from sklearn.metrics import mean_squared_error, r2_score
+from analytics.rollups import build_rollups, filter_segment, segment_summary
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
     st.markdown("<h1 style='margin-top: -10px;'>HYPER - Marketing Campaign Analyzer</h1>", unsafe_allow_html=True)
 
 st.markdown("**Fázis 1-3: Normalizálás → Analízis → Predikció**")
 
 # ============================================================================
 # 📊 HARDKÓDOLT MAPPINGEK (Platform-specifikus)
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
+    "Korosztály": "age_group",
+    "Nem": "gender",
+    "Város": "geo_city",
+    "Eszköz": "device",
+    "Elhelyezés": "placement",
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
+    "Korosztály": "age_group",
+    "Age range": "age_group",
+    "Gender": "gender",
+    "Nem": "gender",
+    "City": "geo_city",
+    "Város": "geo_city",
+    "Device": "device",
+    "Eszköz": "device",
+    "Placement": "placement",
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
+    "Age": "age_group",
+    "Age group": "age_group",
+    "Gender": "gender",
+    "City": "geo_city",
+    "Location": "geo_city",
+    "Device": "device",
+    "Placement": "placement",
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
         ("impressions", "int", "Megjelenések"),
         ("clicks", "int", "Kattintások"),
         ("ctr_percent", "percentage", "CTR (%)"),
         ("cpc", "float", "CPC (HUF)"),
         ("cpa", "float", "CPA (HUF)"),
         ("roas", "float", "ROAS (x)"),
+        ("age_group", "string", "Korcsoport"),
+        ("gender", "string", "Nem"),
+        ("geo_city", "string", "Város"),
+        ("device", "string", "Eszköz"),
+        ("placement", "string", "Elhelyezés"),
     ],
 }
 
 # ============================================================================
 # 🔧 HELPER FUNCTIONS
 # ============================================================================
 
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
@@ -888,60 +915,214 @@ with tab3:
 # ============================================================================
 # TAB 4: DASHBOARD
 # ============================================================================
 
 with tab4:
     st.markdown("### 📊 Szintetikus Dashboard - Fázis 1-3 Összefoglaló")
     
     col1, col2, col3 = st.columns(3)
     
     with col1:
         if st.session_state.normalized_data is not None:
             total_spend = st.session_state.normalized_data['spend'].sum()
             st.metric("💰 Összes Költség", f"{total_spend:,.0f} HUF")
     
     with col2:
         if st.session_state.scores_history:
             st.metric("📊 Elemzett Hirdetések", len(st.session_state.scores_history))
     
     with col3:
         if st.session_state.trained_model:
             st.metric("✅ Modell Status", "🟢 Aktív")
     
     st.markdown("---")
     
     if st.session_state.normalized_data is not None:
+        rollups = build_rollups(st.session_state.normalized_data)
         st.subheader("📈 Kampányok Összesítése")
         
         summary = st.session_state.normalized_data.groupby('platform').agg({
             'spend': 'sum',
             'conversions': 'sum',
             'conversion_value': 'sum',
             'impressions': 'sum',
         }).round(2)
         
         st.dataframe(summary, use_container_width=True)
+
+        with st.expander("📊 Rollup pivotok (havi / regionális / időszaki)"):
+            if rollups.get("monthly") is not None:
+                st.markdown("**Havi bontás**")
+                st.dataframe(rollups["monthly"], use_container_width=True)
+            if rollups.get("regional") is not None:
+                st.markdown("**Regionális bontás (város)**")
+                st.dataframe(rollups["regional"], use_container_width=True)
+            if rollups.get("weekly") is not None:
+                st.markdown("**Heti bontás**")
+                st.dataframe(rollups["weekly"], use_container_width=True)
+            if rollups.get("segment_pivot") is not None:
+                st.markdown("**Dimenzió pivot**")
+                st.dataframe(rollups["segment_pivot"], use_container_width=True)
+
+        st.markdown("---")
+        st.subheader("📊 Trendek")
+
+        trends_df = st.session_state.normalized_data.copy()
+        if "date_start" in trends_df.columns:
+            trends_df["date_start"] = pd.to_datetime(trends_df["date_start"], errors="coerce")
+            trends_df["month_period"] = trends_df["date_start"].dt.to_period("M").astype(str)
+            month_labels = {
+                1: "Január",
+                2: "Február",
+                3: "Március",
+                4: "Április",
+                5: "Május",
+                6: "Június",
+                7: "Július",
+                8: "Augusztus",
+                9: "Szeptember",
+                10: "Október",
+                11: "November",
+                12: "December",
+            }
+            trends_df["month_label"] = trends_df["date_start"].apply(
+                lambda x: f"{month_labels.get(x.month, x.month)} {x.year}" if pd.notna(x) else "Ismeretlen"
+            )
+        else:
+            trends_df["month_label"] = "Ismeretlen"
+            trends_df["month_period"] = "Ismeretlen"
+
+        def build_options(column_name):
+            if column_name not in trends_df.columns:
+                return ["Összes"]
+            values = (
+                trends_df[column_name]
+                .dropna()
+                .astype(str)
+                .replace("nan", pd.NA)
+                .dropna()
+                .unique()
+                .tolist()
+            )
+            return ["Összes"] + sorted(values)
+
+        month_options = ["Összes"] + trends_df[["month_label", "month_period"]].dropna().drop_duplicates().sort_values("month_period")["month_label"].tolist()
+        geo_options = build_options("geo_city")
+        age_options = build_options("age_group")
+        gender_options = build_options("gender")
+        device_options = build_options("device")
+        placement_options = build_options("placement")
+
+        col_a, col_b = st.columns(2)
+        with col_a:
+            st.markdown("**🔹 Szegmens A**")
+            month_a = st.selectbox("Hónap", month_options, key="trend_month_a")
+            geo_a = st.selectbox("Város", geo_options, key="trend_geo_a")
+            age_a = st.selectbox("Korcsoport", age_options, key="trend_age_a")
+            gender_a = st.selectbox("Nem", gender_options, key="trend_gender_a")
+            device_a = st.selectbox("Eszköz", device_options, key="trend_device_a")
+            placement_a = st.selectbox("Elhelyezés", placement_options, key="trend_placement_a")
+
+        with col_b:
+            st.markdown("**🔸 Szegmens B**")
+            month_b = st.selectbox("Hónap", month_options, key="trend_month_b", index=min(1, len(month_options) - 1))
+            geo_b = st.selectbox("Város", geo_options, key="trend_geo_b")
+            age_b = st.selectbox("Korcsoport", age_options, key="trend_age_b")
+            gender_b = st.selectbox("Nem", gender_options, key="trend_gender_b")
+            device_b = st.selectbox("Eszköz", device_options, key="trend_device_b")
+            placement_b = st.selectbox("Elhelyezés", placement_options, key="trend_placement_b")
+
+        def apply_month_filter(df, month_value):
+            if month_value == "Összes":
+                return df
+            return df[df["month_label"] == month_value]
+
+        segment_a = apply_month_filter(trends_df, month_a)
+        segment_a = filter_segment(
+            segment_a,
+            {
+                "geo_city": geo_a,
+                "age_group": age_a,
+                "gender": gender_a,
+                "device": device_a,
+                "placement": placement_a,
+            },
+        )
+
+        segment_b = apply_month_filter(trends_df, month_b)
+        segment_b = filter_segment(
+            segment_b,
+            {
+                "geo_city": geo_b,
+                "age_group": age_b,
+                "gender": gender_b,
+                "device": device_b,
+                "placement": placement_b,
+            },
+        )
+
+        summary_a = segment_summary(segment_a)
+        summary_b = segment_summary(segment_b)
+
+        col1, col2, col3, col4 = st.columns(4)
+        with col1:
+            st.metric("Szegmens A költség", f"{summary_a['spend']:,.0f} HUF")
+        with col2:
+            st.metric("Szegmens A ROAS", f"{summary_a['roas']:.2f}x")
+        with col3:
+            st.metric("Szegmens B költség", f"{summary_b['spend']:,.0f} HUF")
+        with col4:
+            st.metric("Szegmens B ROAS", f"{summary_b['roas']:.2f}x")
+
+        trend_metric = st.selectbox(
+            "Trend metrika",
+            ["spend", "conversion_value", "conversions", "impressions", "clicks", "roas"],
+            key="trend_metric",
+        )
+
+        if "month_period" in trends_df.columns:
+            def metric_series(df_segment, label):
+                if trend_metric in df_segment.columns:
+                    agg_func = "mean" if trend_metric == "roas" else "sum"
+                    grouped = (
+                        df_segment.groupby("month_period")[trend_metric]
+                        .agg(agg_func)
+                        .rename(label)
+                    )
+                else:
+                    grouped = pd.Series(dtype=float)
+                return grouped
+
+            series_a = metric_series(segment_a, "Szegmens A")
+            series_b = metric_series(segment_b, "Szegmens B")
+            trend_data = pd.concat([series_a, series_b], axis=1).fillna(0)
+            if not trend_data.empty:
+                st.line_chart(trend_data)
+            else:
+                st.info("Nincs elég adat a trend charthoz.")
+
+        st.info("🧪 Következő lépés: Prophet / SARIMAX forecasting réteg a szezonális trendek előrejelzésére.")
     
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
