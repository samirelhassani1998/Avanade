# streamlit_app.py
"""
Streamlit app for analysing Generali RPA robot inventory.
Upload a CSV exported from the master sheet (any future version).
The app cleans the data, infers column types, and offers interactive EDA.
Author: Samir <your-GitHub-username>
Created: 2025-06-05
"""

from __future__ import annotations

import io
import re
import unicodedata
from typing import Any, Dict

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Normalize column names: lowercase, ASCII, replace non-alphanum by underscore."""
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
        .lower()
    )
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"__+", "_", text).strip("_")
    return text


def detect_and_fix_header(df: pd.DataFrame) -> pd.DataFrame:
    """If first row duplicates column names, drop it (typical Generali export quirk)."""
    if df.shape[0] > 0 and (df.iloc[0].fillna("") == df.columns.to_series().fillna("")).all():
        df = df.iloc[1:].reset_index(drop=True)
    return df


def clean_generali_robot_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply Generali-specific cleaning & typing rules."""
    raw = detect_and_fix_header(raw)

    # remove fully empty rows
    raw = raw.dropna(how="all").copy()

    # normalise column names
    raw.columns = [slugify(c) for c in raw.columns]

    # type coercion for known numeric fields, if present
    def to_float(series: pd.Series, percent: bool = False) -> pd.Series:
        cleaned = series.str.replace(",", "", regex=False).str.replace("%", "", regex=False)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if percent:
            numeric /= 100.0
        return numeric

    numeric_map = {
        "volumetrie_an": dict(percent=False),
        "reussite": dict(percent=True),
        "gains_etp": dict(percent=False),
    }
    for col, opts in numeric_map.items():
        if col in raw.columns:
            raw[col] = to_float(raw[col].astype(str), percent=opts["percent"])

    return raw


@st.cache_data(show_spinner=False)
def analyse_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a profile summary DataFrame (count, unique, missing etc.)."""
    summary = (
        df.describe(include="all", datetime_is_numeric=True)
        .T
        .assign(
            missing=lambda d: df.shape[0] - d["count"],
            missing_pct=lambda d: 100 * (df.shape[0] - d["count"]) / df.shape[0],
            dtype=df.dtypes.astype(str),
        )
        .reset_index()
        .rename(columns={"index": "column"})
    )
    return summary


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Generali robots analyser", layout="wide")

st.title("📊 Generali RPA robots – Analyse exploratoire interactive")
st.write(
    "Uploadez la dernière exportation CSV de l'inventaire des robots. "
    "Le script se charge automatiquement du nettoyage minimal et propose "
    "plusieurs visualisations et indicateurs clés."
)

uploaded = st.sidebar.file_uploader("Déposez un fichier CSV", type="csv")

if uploaded is None:
    st.info("➡️ Chargez un fichier CSV via le menu latéral pour démarrer.")
    st.stop()

# charge & nettoie
try:
    raw_text = uploaded.getvalue().decode("utf-8")  # sécurité pour différents encodages
except UnicodeDecodeError:
    raw_text = uploaded.getvalue().decode("latin-1")

raw_df = pd.read_csv(io.StringIO(raw_text), dtype=str, skip_blank_lines=True)
clean_df = clean_generali_robot_df(raw_df)

st.success(f"✅ Données chargées : {clean_df.shape[0]:,} lignes × {clean_df.shape[1]} colonnes")

# -- options d'affichage -----------------------------------------------------
if st.sidebar.checkbox("Afficher les données brutes", value=False):
    st.subheader("Jeu de données (nettoyé)")
    st.dataframe(clean_df, use_container_width=True)

# -- résumé -----------------------------------------------------------------
st.header("⚙️ Profil des colonnes")
summary_df = analyse_df(clean_df)
st.dataframe(summary_df, use_container_width=True)

# -- analyses interactives ---------------------------------------------------

st.header("📈 Visualisations rapides")

numeric_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in clean_df.columns if clean_df[c].dtype == "object" and clean_df[c].nunique() < 30]

col1, col2 = st.columns(2)

with col1:
    if numeric_cols:
        chosen_num = st.selectbox("Variable numérique", numeric_cols, key="num")
        hist = (
            alt.Chart(clean_df)
            .mark_bar()
            .encode(
                x=alt.X(chosen_num, bin=alt.Bin(maxbins=40), title=chosen_num),
                y=alt.Y("count()", title="Effectif"),
                tooltip=["count()"],
            )
            .properties(height=300)
        )
        st.altair_chart(hist, use_container_width=True)
    else:
        st.info("Aucune variable numérique trouvée.")

with col2:
    if len(numeric_cols) >= 2:
        corr = clean_df[numeric_cols].corr().stack().reset_index()
        corr.columns = ["var1", "var2", "corr"]
        heatmap = (
            alt.Chart(corr)
            .mark_rect()
            .encode(
                x="var1:O",
                y="var2:O",
                color=alt.Color("corr:Q", scale=alt.Scale(scheme="viridis")),
                tooltip=["var1", "var2", alt.Tooltip("corr:Q", format=".2f")],
            )
            .properties(height=300)
        )
        st.altair_chart(heatmap, use_container_width=True)
    else:
        st.info("Pas assez de variables numériques pour afficher la matrice de corrélation.")

# petites distributions categos
if cat_cols:
    st.subheader("Répartition des variables catégorielles")
    chosen_cat = st.selectbox("Variable catégorielle (< 30 modalités)", cat_cols, key="cat")
    bar = (
        alt.Chart(clean_df)
        .mark_bar()
        .encode(
            y=alt.Y(chosen_cat, sort="-x"),
            x=alt.X("count()", title="Effectif"),
            tooltip=["count()"],
        )
        .properties(height=400)
    )
    st.altair_chart(bar, use_container_width=True)

# -- téléchargement du csv nettoyé ------------------------------------------
@st.cache_data
def get_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

st.download_button(
    "💾 Télécharger le CSV nettoyé",
    get_csv_bytes(clean_df),
    file_name="robots_clean.csv",
    mime="text/csv",
)

st.caption("© 2025 - Généré avec Streamlit • Samir • Avanade")
