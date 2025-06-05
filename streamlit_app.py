# streamlit_app.py
"""
Streamlit app for analysing Generali RPA robot inventory.
Upload a CSV exported from the master sheet (any future version).
If aucun fichier n'est uploadé, l'app affiche simplement l'interface d'accueil.

Samir – Avanade (2025‑06‑05)
"""

from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

###############################################################################
# CONFIGURATION
###############################################################################

author = "Samir El Hassani"
repository_url = "https://github.com/samirelhassani1998/Avanade"

st.set_page_config(
    page_title="Generali RPA inventory", page_icon="🤖", layout="wide"
)

###############################################################################
# HELPERS
###############################################################################

def slugify(text: str) -> str:
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
        .strip()
    )
    text = re.sub(r"[^0-9a-z_]+", "_", text)
    text = re.sub(r"_{2,}", "_", text).strip("_")
    return text


def detect_header_duplication(df: pd.DataFrame) -> pd.DataFrame:
    """If first row equals current column names, drop this first row."""
    if df.empty:
        return df
    col_equal = (df.iloc[0].astype(str).str.strip() == df.columns.astype(str).str.strip()).all()
    return df.iloc[1:].reset_index(drop=True) if col_equal else df


def to_numeric(series: pd.Series, percent: bool = False) -> pd.Series:
    if series.dtype.kind in "biufc":
        return series
    s = series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
    out = pd.to_numeric(s, errors="coerce")
    return out / 100 if percent else out


def clean_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply rules specific to Generali RPA inventory."""
    df = detect_header_duplication(df_raw)
    df = df.dropna(how="all")

    # Normalise column names once
    df.columns = [slugify(c) for c in df.columns]

    # Typed columns of interest – optional if not present
    mappings = {
        "volumetrie_an": dict(percent=False),
        "gains_etp": dict(percent=False),
        "reussite": dict(percent=True),  # ex: "71%"
    }
    for col, opts in mappings.items():
        if col in df.columns:
            df[col] = to_numeric(df[col], percent=opts["percent"])

    return df


@st.cache_data(show_spinner=False)
def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.describe(include="all", datetime_is_numeric=True)
        .T
        .assign(dtype=df.dtypes.astype(str))
        .reset_index()
        .rename(columns={"index": "column"})
    )
    total_rows = len(df)
    out["missing"] = total_rows - out["count"]
    out["missing_pct"] = 100 * out["missing"] / total_rows
    return out


###############################################################################
# SIDEBAR
###############################################################################

st.sidebar.header("📂 Charger un CSV")
upload_file = st.sidebar.file_uploader("Déposez le fichier exporté (CSV)", type=["csv"])

if upload_file is None:
    st.sidebar.info("L'app reste vide tant qu'aucun fichier n'est chargé.")
else:
    st.sidebar.success("Fichier prêt à être analysé ✅")

###############################################################################
# MAIN PAGE
###############################################################################

st.title("🤖 Inventaire RPA Generali – Analyse interactive")

st.markdown(
    "Cette application vous permet de charger l'export CSV du catalogue de robots, "
    "d'obtenir un nettoyage de base et d'explorer rapidement le jeu de données.\n"
    "Les visualisations se mettent à jour automatiquement si la structure évolue."
)

if upload_file is None:
    st.info("\n**➡️ Pour commencer :** importez un fichier CSV via la barre latérale.\n")
    st.stop()

###########################
# 01 — Chargement / nettoyage
###########################

raw_text = upload_file.getvalue().decode("utf-8", errors="replace")
df_raw = pd.read_csv(io.StringIO(raw_text), dtype=str)

df = clean_dataframe(df_raw)

st.success(f"{df.shape[0]:,} lignes × {df.shape[1]} colonnes après nettoyage.")

###########################
# 02 — Profil des données
###########################

st.subheader("📑 Profil statistique")
profile = profile_dataframe(df)
st.dataframe(profile, use_container_width=True)

###########################
# 03 — Visualisations interactives
###########################

numeric_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols: List[str] = [c for c in df.columns if df[c].nunique() < 30 and c not in numeric_cols]

with st.expander("Histogrammes des variables numériques"):
    if numeric_cols:
        col_choice = st.selectbox("Choisir une variable numérique", numeric_cols)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(col_choice, bin=alt.Bin(maxbins=40)),
                y="count()",
                tooltip=["count()"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Aucune colonne numérique détectée.")

with st.expander("⚖️ Corrélations (Pearson)"):
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
            )
            .properties(height=400)
        )
        st.altair_chart(heat, use_container_width=True)
    else:
        st.info("Pas assez de variables numériques pour la corrélation.")

with st.expander("📊 Répartition des variables catégorielles"):
    if cat_cols:
        cat_choice = st.selectbox("Variable catégorielle (<30 modalités)", cat_cols)
        bar = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                y=alt.Y(cat_choice, sort="-x"),
                x="count()",
                tooltip=["count()"],
            )
            .properties(height=400)
        )
        st.altair_chart(bar, use_container_width=True)
    else:
        st.info("Aucune variable catégorielle avec <30 modalités.")

###########################
# 04 — Export DataFrame nettoyé
###########################

@st.cache_data
def get_clean_csv_bytes(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8")

st.download_button(
    "💾 Télécharger le CSV nettoyé", get_clean_csv_bytes(df), "robots_clean.csv", "text/csv"
)

###############################################################################
# FOOTER
###############################################################################

st.caption(
    f"App Streamlit par {author}.  \u2022  "
    f"[Voir le code]({repository_url}/blob/main/streamlit_app.py)  \u2022  "
    "© 2025 Generali / Avanade"
)
