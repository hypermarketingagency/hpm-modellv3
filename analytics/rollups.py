import pandas as pd


DIMENSION_COLUMNS = [
    "age_group",
    "gender",
    "geo_city",
    "geo_region",
    "device",
    "placement",
]


def add_time_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    if "date_start" not in df.columns:
        return df
    out = df.copy()
    out["date_start"] = pd.to_datetime(out["date_start"], errors="coerce")
    out["year"] = out["date_start"].dt.year
    out["month"] = out["date_start"].dt.to_period("M").astype(str)
    out["week"] = out["date_start"].dt.to_period("W").astype(str)
    return out


def build_rollups(df: pd.DataFrame) -> dict:
    enriched = add_time_dimensions(df)

    metrics = {
        "spend": "sum",
        "conversions": "sum",
        "conversion_value": "sum",
        "impressions": "sum",
        "clicks": "sum",
        "roas": "mean",
    }

    metrics = {k: v for k, v in metrics.items() if k in enriched.columns}

    rollups = {}

    if metrics:
        if "month" in enriched.columns:
            rollups["monthly"] = (
                enriched.groupby(["month", "platform"], dropna=False)
                .agg(metrics)
                .reset_index()
                .sort_values(["month", "platform"])
            )
        if "geo_region" in enriched.columns:
            rollups["regional"] = (
                enriched.groupby(["geo_region", "platform"], dropna=False)
                .agg(metrics)
                .reset_index()
                .sort_values(["geo_region", "platform"])
            )
        elif "geo_city" in enriched.columns:
            rollups["regional"] = (
                enriched.groupby(["geo_city", "platform"], dropna=False)
                .agg(metrics)
                .reset_index()
                .sort_values(["geo_city", "platform"])
            )
        if "week" in enriched.columns:
            rollups["weekly"] = (
                enriched.groupby(["week", "platform"], dropna=False)
                .agg(metrics)
                .reset_index()
                .sort_values(["week", "platform"])
            )

    pivot_dims = [col for col in DIMENSION_COLUMNS if col in enriched.columns]
    if pivot_dims and metrics:
        rollups["segment_pivot"] = (
            enriched.groupby(pivot_dims, dropna=False)
            .agg(metrics)
            .reset_index()
            .sort_values(pivot_dims)
        )

    return rollups


def filter_segment(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    filtered = df.copy()
    for key, value in filters.items():
        if value == "Összes" or key not in filtered.columns:
            continue
        filtered = filtered[filtered[key] == value]
    return filtered


def segment_summary(df: pd.DataFrame) -> pd.Series:
    summary = {
        "spend": df["spend"].sum() if "spend" in df.columns else 0,
        "conversions": df["conversions"].sum() if "conversions" in df.columns else 0,
        "conversion_value": df["conversion_value"].sum() if "conversion_value" in df.columns else 0,
    }
    if "roas" in df.columns and df["roas"].notna().any():
        summary["roas"] = df["roas"].mean()
    elif summary["spend"]:
        summary["roas"] = summary["conversion_value"] / summary["spend"]
    else:
        summary["roas"] = 0
    return pd.Series(summary)
