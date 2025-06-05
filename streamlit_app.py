# streamlit_app.py
"""
Streamlit app – Générateur d'analyses pour le catalogue de robots RPA Generali
---------------------------------------------------------------------------------
• Upload : accepte le CSV exporté (encodage UTF-8 / séparateur virgule, point-virgule ou tabulation).
• Nettoyage automatique (en-têtes multiples, types numériques, NA, etc.).
• Tableaux de bord : KPIs, histogrammes, corrélations, distributions catégorie, tops.

Auteur : Samir El Hassani – Avanade · 2025-06-05
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
    """Simplifie un libellé : ascii, lower, espaces → _"""
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
    """Lecture robuste : détection du séparateur via csv.Sniffer."""
    raw = upload.read()
    sample = raw[:2048].decode("utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
        sep = dialect.delimiter
    except csv.Error:
        sep = ","  # défaut
    upload.seek(0)
    return pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str, keep_default_na=False)


def fix_multirow_header(df: pd.DataFrame) -> pd.DataFrame:
    """Si des colonnes Unnamed_x sont présentes, utilise la 1ʳᵉ ligne comme nouvel en-tête."""
    if any(c.lower().startswith("unnamed") or c == "" for c in df.columns):
        if df.shape[0] > 0:
            new_cols = df.iloc[0].tolist()
            if len(set(new_cols)) == len(new_cols):  # vraisemblablement un vrai header
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

    # normalisation des colonnes
    df.columns = [slugify(c) if c != "" else f"col_{i}" for i, c in enumerate(df.columns)]

    # typage des champs numériques connus
    mapping = {
        "volumetrie_an": False,
        "gains_etp": False,
        "reussite": True,
    }
    for col, is_percent in mapping.items():
        if col in df.columns:
            df[col] = to_numeric(df[col], percent=is_percent)

    return df


@st.cache_data(show_spinner=False)
def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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
    prof["missing"] = total - prof["count"]
    prof["missing_pct"] = (prof["missing"] / total * 100).round(1)
    return prof

###############################################################################
# BARRE LATÉRALE
###############################################################################

st.sidebar.header("📂 Import CSV")
uploader = st.sidebar.file_uploader("Glissez votre fichier exporté (UTF-8)", type=["csv"])

if not uploader:
    st.sidebar.info("L'app reste inactive tant qu'aucun CSV n'est fourni.")
    st.stop()

###############################################################################
# CHARGEMENT & NETTOYAGE
###############################################################################

raw_df = read_csv_robust(uploader)
df = clean_dataframe(raw_df)

st.success(f"{df.shape[0]:,} lignes × {df.shape[1]} colonnes après nettoyage.")

###############################################################################
# INDICATEURS CLÉS
###############################################################################

kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.metric("Robots recensés", f"{df.shape[0]:,}")

if "volumetrie_an" in df.columns:
    with kpi_cols[1]:
        st.metric("Volumétrie annuelle (totale)", f"{df['volumetrie_an'].sum():,.0f}")

if "reussite" in df.columns:
    with kpi_cols[2]:
        st.metric("% réussite moyen", f"{df['reussite'].mean()*100:,.1f} %")

if "gains_etp" in df.columns:
    with kpi_cols[3]:
        st.metric("Gains ETP cumulés", f"{df['gains_etp'].sum():,.1f}")

###############################################################################
# PROFIL DES DONNÉES
###############################################################################

with st.expander("📑 Profil statistique (pandas.describe)", expanded=True):
    st.dataframe(profile_dataframe(df), use_container_width=True)

###############################################################################
# VISUALISATIONS
###############################################################################

numeric_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols: List[str] = [c for c in df.columns if df[c].nunique() <= 30 and c not in numeric_cols]

## Distribution numérique
a1 = st.container()
with a1.expander("📊 Histogramme d'une variable numérique"):
    if numeric_cols:
        num_var = st.selectbox("Choisir la variable", numeric_cols)
        hist = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(num_var, bin=alt.Bin(maxbins=50), title=num_var),
                y="count()",
                tooltip=["count()"],
            ).properties(height=300)
        )
        st.altair_chart(hist, use_container_width=True)
    else:
        st.info("Aucune variable numérique détectée.")

## Corrélation
with st.expander("🔗 Corrélations (Pearson)"):
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().stack().reset_index()
        corr.columns = ["x", "y", "corr"]
        heat = (
            alt.Chart(corr)
            .mark_rect()
            .encode(
                x="x:O",
                y="y:O",
                color=alt.Color("corr:Q", scale=alt.Scale(scheme="viridis")),
                tooltip=[alt.Tooltip("corr:Q", format=".2f")],
            ).properties(height=400)
        )
        st.altair_chart(heat, use_container_width=True)
    else:
        st.info("Pas assez de variables numériques.")

## Top volumétrie
a2 = st.container()
with a2.expander("🏅 Top 10 par volumétrie annuelle"):
    if "volumetrie_an" in df.columns and "nom" in df.columns:
        top10 = df.sort_values("volumetrie_an", ascending=False).head(10)
        chart = (
            alt.Chart(top10)
            .mark_bar()
            .encode(
                y=alt.Y("nom", sort="-x"),
                x="volumetrie_an:Q",
                tooltip=["volumetrie_an"],
            ).properties(height=400)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Colonnes nom/volumetrie_an manquantes.")

## Répartition catégorielle
with st.expander("📂 Répartition d'une variable catégorielle (<=30 modalités)"):
    if cat_cols:
        cat_var = st.selectbox("Choisir la catégorie", cat_cols)
        bar = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                y=alt.Y(cat_var, sort="-x"),
                x="count()",
                tooltip=["count()"],
            ).properties(height=400)
        )
        st.altair_chart(bar, use_container_width=True)
    else:
        st.info("Aucune variable catégorielle courte.")

###############################################################################
# EXPORT DU CSV NETTOYÉ
###############################################################################

@st.cache_data
def to_csv_bytes(d: pd.DataFrame) -> bytes:
    return d.to_csv(index=False).encode("utf-8")

st.download_button("💾 Télécharger le CSV nettoyé", to_csv_bytes(df), "robots_clean.csv", "text/csv")

###############################################################################
# FOOTER
###############################################################################

st.caption(
    f"Réalisé par {author} · [Code]({repo_url}/blob/main/streamlit_app.py) · 2025 Generali / Avanade"
)
