# streamlit_app.py
"""
Streamlit app – Analyse interactive du catalogue RPA Generali.
Inclut :
  • upload CSV robuste (détection séparateur)  
  • nettoyage (en‑têtes multiples, typage, NA)  
  • KPIs + visualisations (histogramme, heatmap corrélation, top volumétrie, répartition catégorielle)  
  • export du CSV nettoyé

Auteur : Samir El Hassani – Avanade · 2025‑06‑05
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from typing import List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

###############################################################################
# CONFIGURATION GLOBALE
###############################################################################

author = "Samir El Hassani"
repo_url = "https://github.com/samirelhassani1998/Avanade"

st.set_page_config(page_title="Inventaire RPA Generali", page_icon="🤖", layout="wide")

###############################################################################
# FONCTIONS UTILITAIRES
###############################################################################

def slugify(text: str) -> str:
    text = (
        unicodedata.normalize("NFKD", str(text))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
        .strip()
    )
    text = re.sub(r"[^0-9a-z]+", "_", text)
    text = re.sub(r"_{2,}", "_", text).strip("_")
    return text


def read_csv_robust(upload) -> pd.DataFrame:
    """Lecture avec détection automatique du séparateur (, ; ou tab)."""
    raw = upload.read()
    sample = raw[:2048].decode("utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
        sep = dialect.delimiter
    except csv.Error:
        sep = ","
    upload.seek(0)
    return pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str, keep_default_na=False)


def fix_multirow_header(df: pd.DataFrame) -> pd.DataFrame:
    if any(c.lower().startswith("unnamed") or c == "" for c in df.columns):
        if df.shape[0] > 0:
            new_cols = df.iloc[0].tolist()
            if len(set(new_cols)) == len(new_cols):
                df = df.iloc[1:].reset_index(drop=True)
                df.columns = new_cols
    return df


def to_numeric(series: pd.Series, percent: bool = False) -> pd.Series:
    s = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    num = pd.to_numeric(s, errors="coerce")
    return num / 100 if percent else num


def clean_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = fix_multirow_header(df_raw)
    df = df.dropna(how="all").copy()
    df.columns = [slugify(c) if c else f"col_{i}" for i, c in enumerate(df.columns)]

    mapping = {"volumetrie_an": False, "gains_etp": False, "reussite": True}
    for col, is_pct in mapping.items():
        if col in df.columns:
            df[col] = to_numeric(df[col], percent=is_pct)

    return df


@st.cache_data(show_spinner=False)
def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Profil statistique sécurisé (évite TypeError sur .round)."""
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

    total = float(len(df))
    prof["count"] = pd.to_numeric(prof["count"], errors="coerce").fillna(0)
    prof["missing"] = (total - prof["count"]).astype(float)
    prof["missing_pct"] = ((prof["missing"] / total) * 100).astype(float).round(1)

    return prof

###############################################################################
# BARRE LATÉRALE
###############################################################################

st.sidebar.header("📂 Import CSV")
uploader = st.sidebar.file_uploader("Glissez votre fichier exporté (UTF-8)", type=["csv"])

if not uploader:
    st.sidebar.info("L'app reste inactive tant qu'aucun fichier n'est fourni.")
    st.stop()

###############################################################################
# CHARGEMENT & NETTOYAGE
###############################################################################

raw_df = read_csv_robust(uploader)
df = clean_dataframe(raw_df)

st.success(f"{df.shape[0]:,} lignes × {df.shape[1]} colonnes après nettoyage.")

###############################################################################
# KPIs
###############################################################################

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Robots recensés", f"{df.shape[0]:,}")
if "volumetrie_an" in df.columns:
    with kpi_cols[1]:
        st.metric("Volumétrie annuelle", f"{df['volumetrie_an'].sum():,.0f}")
if "reussite" in df.columns:
    with kpi_cols[2]:
        st.metric("% réussite moyen", f"{df['reussite'].mean()*100:,.1f}%")
if "gains_etp" in df.columns:
    with kpi_cols[3]:
        st.metric("Gains ETP", f"{df['gains_etp'].sum():,.1f}")

###############################################################################
# PROFIL
###############################################################################

with st.expander("📑 Profil statistique", expanded=True):
    st.dataframe(profile_dataframe(df), use_container_width=True)

###############################################################################
# VISUALISATIONS
###############################################################################

numeric_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols: List[str] = [c for c in df.columns if df[c].nunique() <= 30 and c not in numeric_cols]

with st.expander("📊 Histogramme numérique"):
    if numeric_cols:
        num_var = st.selectbox("Variable", numeric_cols, key="hist")
        st.altair_chart(
            alt.Chart(df).mark_bar().encode(
                x=alt.X(num_var, bin=alt.Bin(maxbins=50)),
                y="count()",
                tooltip=["count()"],
            ).properties(height=300),
            use_container_width=True,
        )
    else:
        st.info("Aucune variable numérique détectée.")

with st.expander("🔗 Heatmap corrélation"):
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().stack().reset_index()
        corr.columns = ["x", "y", "corr"]
        st.altair_chart(
            alt.Chart(corr).mark_rect().encode(
                x="x:O",
                y="y:O",
                color=alt.Color("corr:Q", scale=alt.Scale(scheme="viridis")),
                tooltip=[alt.Tooltip("corr:Q", format=".2f")],
            ).properties(height=400),
            use_container_width=True,
        )
    else:
        st.info("Pas assez de variables numériques.")

with st.expander("🏅 Top 10 volumétrie"):
    if {"volumetrie_an", "nom"}.issubset(df.columns):
        top10 = df.sort_values("volumetrie_an", ascending=False).head(10)
        st.altair_chart(
            alt.Chart(top10).mark_bar().encode(
                y=alt.Y("nom", sort="-x"),
                x="volumetrie_an:Q",
                tooltip=["volumetrie_an"],
            ).properties(height=400),
            use_container_width=True,
        )
    else:
        st.info("Colonnes manquantes pour le top.")

with st.expander("📂 Répartition catégorielle"):
    if cat_cols:
        cat_var = st.selectbox("Catégorie", cat_cols, key="cat")
        st.altair_chart(
            alt.Chart(df).mark_bar().encode(
                y=alt.Y(cat_var, sort="-x"),
                x="count()",
                tooltip=["count()"],
            ).properties(height=400),
            use_container_width=True,
        )
    else:
        st.info("Aucune variable catégorielle <=30 modalités.")

###############################################################################
# EXPORT
###############################################################################

@st.cache_data
def get_csv_bytes(d: pd.DataFrame) -> bytes:
    return d.to_csv(index=False).encode("utf-8")

st.download_button("💾 Télécharger CSV nettoyé", get_csv_bytes(df), "robots_clean.csv", "text/csv")

###############################################################################
# FOOTER
###############################################################################

st.caption(
    f"Réalisé par {author} · [Code]({repo_url}/blob/main/streamlit_app.py) · © 2025 Generali / Avanade"
)
