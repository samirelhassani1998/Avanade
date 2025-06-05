# streamlit_app.py
"""
Streamlit – Tableau de bord RPA Generali
=======================================
Fonctions principales :
• Upload CSV robuste (séparateur détecté)
• Nettoyage + typage automatique
• Filtres interactifs (zone fonctionnelle, plage volumétrie, % réussite)
• Indicateurs clés dynamiques
• Visualisations : histogrammes, box‑plots, heatmap corrélations, nuage de points, top‑N
• Insight tables : agrégation par zone, détection des outliers
• Export CSV nettoyé + CSV filtré

Samir El Hassani – Avanade · 2025‑06‑05
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata
from typing import List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

###############################################################################
# CONFIG & THEME
###############################################################################

AUTHOR = "Samir El Hassani"
REPO_URL = "https://github.com/samirelhassani1998/Avanade"

st.set_page_config(
    page_title="Inventaire RPA Generali", page_icon="🤖", layout="wide"
)

###############################################################################
# HELPERS
###############################################################################

def slugify(text: str) -> str:
    """ASCII‑fication + snake_case."""
    text = (
        unicodedata.normalize("NFKD", str(text))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
        .strip()
    )
    text = re.sub(r"[^0-9a-z]+", "_", text)
    return re.sub(r"_{2,}", "_", text).strip("_")


def sniff_sep(raw: bytes) -> str:
    sample = raw[:2048].decode("utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
        return dialect.delimiter
    except csv.Error:
        return ","


def read_csv_robust(upload) -> pd.DataFrame:
    raw = upload.read()
    sep = sniff_sep(raw)
    upload.seek(0)
    return pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str, keep_default_na=False)


def promote_first_row_header(df: pd.DataFrame) -> pd.DataFrame:
    if any(c.lower().startswith("unnamed") or c == "" for c in df.columns):
        if df.shape[0] > 0 and df.iloc[0].nunique() == df.shape[1]:
            df.columns = df.iloc[0]
            df = df.iloc[1:]
    return df.reset_index(drop=True)


def to_numeric(series: pd.Series, is_pct: bool = False) -> pd.Series:
    s = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("\u00a0", "", regex=False)  # nbsp
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    num = pd.to_numeric(s, errors="coerce")
    return num / 100 if is_pct else num


def clean_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = promote_first_row_header(df_raw).copy()
    df = df.dropna(how="all")
    df.columns = [slugify(c) if c else f"col_{i}" for i, c in enumerate(df.columns)]

    conversions = {
        "volumetrie_an": False,
        "gains_etp": False,
        "reussite": True,
    }
    for col, pct in conversions.items():
        if col in df.columns:
            df[col] = to_numeric(df[col], pct)

    # types finaux
    return df


###############################################################################
# CACHE FUNCTIONS
###############################################################################

@st.cache_data(show_spinner=False)
def get_profile(df: pd.DataFrame) -> pd.DataFrame:
    try:
        desc = df.describe(include="all", datetime_is_numeric=True)
    except TypeError:
        desc = df.describe(include="all")
    prof = (
        desc.T
        .assign(dtype=df.dtypes.astype(str))
        .reset_index()
        .rename(columns={"index": "column"})
    )
    total = len(df)
    prof["count"] = pd.to_numeric(prof["count"], errors="coerce").fillna(0)
    prof["missing"] = total - prof["count"]
    prof["missing_pct"] = (prof["missing"] / total * 100).round(1)
    return prof


###############################################################################
# SIDEBAR – UPLOAD & FILTERS
###############################################################################

st.sidebar.header("📂 Charger le CSV")
file = st.sidebar.file_uploader("Déposez le fichier exporté", type=["csv"])

if not file:
    st.sidebar.info("Importez un fichier pour commencer.")
    st.stop()

# chargement + nettoyage
raw = read_csv_robust(file)
base_df = clean_df(raw)

# filtres interactifs
zones = base_df["zone_fonctionnelle"].dropna().unique().tolist() if "zone_fonctionnelle" in base_df.columns else []
sel_zones = st.sidebar.multiselect("Zone fonctionnelle", options=zones, default=zones)

vol_min, vol_max = (base_df["volumetrie_an"].min(skipna=True), base_df["volumetrie_an"].max(skipna=True)) if "volumetrie_an" in base_df.columns else (0, 0)
if pd.notna(vol_min):
    v_range = st.sidebar.slider("Plage volumétrie/an", float(vol_min), float(vol_max), (float(vol_min), float(vol_max)))
else:
    v_range = (None, None)

# appliquer filtres
filt_df = base_df.copy()
if sel_zones:
    filt_df = filt_df[filt_df["zone_fonctionnelle"].isin(sel_zones) | filt_df["zone_fonctionnelle"].isna()]
if v_range[0] is not None and "volumetrie_an" in filt_df.columns:
    filt_df = filt_df[(filt_df["volumetrie_an"] >= v_range[0]) & (filt_df["volumetrie_an"] <= v_range[1])]

###############################################################################
# KPI SECTION
###############################################################################

st.title("🤖 Inventaire RPA Generali – Dashboard interactif")

kpi = st.columns(5)
with kpi[0]:
    st.metric("Robots (filtrés)", f"{filt_df.shape[0]:,}")
if "volumetrie_an" in filt_df.columns:
    with kpi[1]:
        st.metric("Volumétrie totale", f"{filt_df['volumetrie_an'].sum():,.0f}")
if "reussite" in filt_df.columns:
    with kpi[2]:
        st.metric("% Réussite médiane", f"{filt_df['reussite'].median()*100:,.1f}%")
    with kpi[3]:
        st.metric("% Réussite < 60%", f"{(filt_df['reussite']<0.60).sum()}")
if "gains_etp" in filt_df.columns:
    with kpi[4]:
        st.metric("Gains ETP cumulés", f"{filt_df['gains_etp'].sum():,.1f}")

###############################################################################
# TABS LAYOUT
###############################################################################

overview, dist_tab, rela_tab, data_tab = st.tabs(["Vue d'ensemble", "Distributions", "Relations", "Données"])

###############################################################################
# OVERVIEW TAB
###############################################################################

with overview:
    st.subheader("Top 15 volumétrie annuelle")
    if {"nom", "volumetrie_an"}.issubset(filt_df.columns):
        top = filt_df.nlargest(15, "volumetrie_an")[["nom", "volumetrie_an", "reussite", "gains_etp"]]
        st.dataframe(top, hide_index=True)

    st.subheader("Agrégation par zone fonctionnelle")
    if "zone_fonctionnelle" in filt_df.columns and "volumetrie_an" in filt_df.columns:
        agg = (
            filt_df.groupby("zone_fonctionnelle")
            .agg(
                robots=("identification", "count") if "identification" in filt_df.columns else ("nom", "count"),
                volumetrie_total=("volumetrie_an", "sum"),
                succes_moyen=("reussite", "mean") if "reussite" in filt_df.columns else ("volumetrie_an", "size"),
            )
            .sort_values("volumetrie_total", ascending=False)
        )
        st.dataframe(agg, use_container_width=True)

###############################################################################
# DISTRIBUTIONS TAB
###############################################################################

with dist_tab:
    st.subheader("Histogrammes & Box‑plots")
    if numeric_cols := filt_df.select_dtypes(include=[np.number]).columns.tolist():
        num = st.selectbox("Variable numérique", numeric_cols, key="dist_num")
        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(
                alt.Chart(filt_df).mark_bar().encode(
                    x=alt.X(num, bin=alt.Bin(maxbins=60)),
                    y="count()",
                ),
                use_container_width=True,
            )
        with col2:
            st.altair_chart(
                alt.Chart(filt_df).mark_boxplot(extent="min-max").encode(y=num),
                use_container_width=True,
            )
    else:
        st.info("Aucune colonne numérique disponible.")

###############################################################################
# RELATIONS TAB
###############################################################################

with rela_tab:
    st.subheader("Nuage de points Volumetrie vs % Réussite")
    if {"volumetrie_an", "reussite"}.issubset(filt_df.columns):
        scatter = alt.Chart(filt_df).mark_circle(size=60, opacity=0.7).encode(
            x="volumetrie_an",
            y=alt.Y("reussite", axis=alt.Axis(format=".0%")),
            color="zone_fonctionnelle" if "zone_fonctionnelle" in filt_df.columns else alt.value("steelblue"),
            tooltip=list(filt_df.columns),
        ).interactive()
        st.altair_chart(scatter, use_container_width=True)
    else:
        st.info("Variables nécessaires manquantes.")

    st.subheader("Heatmap des corrélations")
    num_cols = filt_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        corr = filt_df[num_cols].corr().stack().reset_index()
        corr.columns = ["x", "y", "corr"]
        st.altair_chart(
            alt.Chart(corr).mark_rect().encode(
                x="x:O",
                y="y:O",
                color=alt.Color("corr:Q", scale=alt.Scale(scheme="viridis")),
                tooltip=[alt.Tooltip("corr:Q", format=".2f")],
            ),
            use_container_width=True,
        )
    else:
        st.info("Pas assez de variables pour la corrélation.")

###############################################################################
# DATA TAB
###############################################################################

with data_tab:
    st.subheader("Profil statistique complet")
    st.dataframe(get_profile(filt_df), use_container_width=True)

    st.subheader("Jeu de données filtré")
    st.dataframe(filt_df, use_container_width=True)

    # téléchargement
    csv_bytes = filt_df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Télécharger CSV filtré", csv_bytes, "robots_filtered.csv", "text/csv")

###############################################################################
# FOOTER
###############################################################################

st.caption(
    f"Créé par {AUTHOR} – [Code]({REPO_URL}/blob/main/streamlit_app.py) – © 2025 Generali / Avanade"
)
