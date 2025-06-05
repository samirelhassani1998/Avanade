# -*- coding: utf-8 -*-
"""
Streamlit – Inventaire RPA Generali (v2.1)
Auteur : Samir El Hassani – Avanade · 2025-06-05
"""

from __future__ import annotations

###############################################################################
# Imports
###############################################################################
import csv
import io
import itertools
import re
import unicodedata
import warnings
from typing import List

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import streamlit as st
from sklearn.cluster import KMeans

###############################################################################
# Page & Altair
###############################################################################
st.set_page_config(page_title="Inventaire RPA Generali",
                   page_icon="🤖",
                   layout="wide")
pd.options.display.float_format = "{:,.2f}".format
alt.data_transformers.disable_max_rows()

AUTHOR = "Samir El Hassani"
REPO   = "https://github.com/samirelhassani1998/Avanade"

###############################################################################
# Utilitaires : nettoyage et typage
###############################################################################
def slugify(txt: str) -> str:
    txt = (
        unicodedata.normalize("NFKD", str(txt))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
        .strip()
    )
    txt = re.sub(r"[^0-9a-z]+", "_", txt)
    return re.sub(r"_{2,}", "_", txt).strip("_")


def read_csv_robust(buf) -> pd.DataFrame:
    """Lecture avec détection automatique du séparateur (virgule, « ; » ou tab)."""
    raw = buf.read()
    sample = raw[:2048].decode("utf-8", "replace")
    try:
        sep = csv.Sniffer().sniff(sample, [",", ";", "\t"]).delimiter
    except csv.Error:
        sep = ","
    buf.seek(0)                                 # IMPORTANT : repositionner le curseur
    return pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str, keep_default_na=False)


def promote_header(df: pd.DataFrame) -> pd.DataFrame:
    """Si la première ligne contient les vrais noms de colonnes → la promouvoir."""
    if any(c.lower().startswith("unnamed") for c in df.columns):
        first = df.iloc[0].tolist()
        if len(set(first)) == len(first):       # pas de doublon = vraie en-tête
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = first
    return df


def to_num(s: pd.Series, pct: bool = False) -> pd.Series:
    s = (
        s.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)      # gère les espaces insécables
    )
    n = pd.to_numeric(s, errors="coerce")
    return n / 100 if pct else n


def clean(df0: pd.DataFrame) -> pd.DataFrame:
    df = promote_header(df0)
    df.dropna(how="all", inplace=True)
    df.columns = [slugify(c) or f"col_{i}" for i, c in enumerate(df.columns)]

    mapping = {"volumetrie_an": False, "gains_etp": False, "reussite": True}
    for col, pct in mapping.items():
        if col in df.columns:
            df[col] = to_num(df[col], pct)
    return df

###############################################################################
# Profil + corrélation avec p-value
###############################################################################
@st.cache_data(show_spinner=False)
def profile(df: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        desc = df.describe(include="all")
    prof = (
        desc.T.assign(dtype=df.dtypes.astype(str))
        .reset_index()
        .rename(columns={"index": "col"})
    )
    total = len(df)
    prof["count"]       = pd.to_numeric(prof["count"], errors="coerce").fillna(0)
    prof["missing"]     = (total - prof["count"]).astype(float)
    prof["missing_pct"] = (prof["missing"] / total * 100).round(1)
    return prof


@st.cache_data(show_spinner=False)
def corr_with_p(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(np.number)
    combos = list(itertools.combinations(num.columns, 2))
    data = []
    for a, b in combos:
        r, p = stats.pearsonr(num[a].dropna(), num[b].dropna())
        data.append({"var_x": a, "var_y": b, "corr": r, "pval": p})
    return pd.DataFrame(data)

###############################################################################
# SIDEBAR : upload + filtres
###############################################################################
st.sidebar.header("📂 Charger le CSV")
upload = st.sidebar.file_uploader("Fichier (UTF-8)", type="csv")

if upload is None:
    st.sidebar.info("➡️ Glissez un fichier pour démarrer.")
    st.stop()

df_raw = read_csv_robust(upload)
df     = clean(df_raw)

# ---- Filtres dynamiques
with st.sidebar:
    st.markdown("### 🎛️ Filtres")
    if "zone_fonctionnelle" in df.columns:
        zones = st.multiselect(
            "Zone fonctionnelle",
            sorted(df["zone_fonctionnelle"].unique()),
            default=sorted(df["zone_fonctionnelle"].unique()),
        )
        df = df[df["zone_fonctionnelle"].isin(zones)]

    if "volumetrie_an" in df.columns:
        vmin, vmax = float(df["volumetrie_an"].min()), float(df["volumetrie_an"].max())
        rng = st.slider("Plage volumétrie/an", vmin, vmax, (vmin, vmax))
        df = df[df["volumetrie_an"].between(*rng)]

    if "reussite" in df.columns:
        rs = st.slider("% réussite", 0.0, 1.0, (0.0, 1.0))
        df = df[df["reussite"].between(*rs)]

###############################################################################
# KPIs
###############################################################################
st.success(f"{len(df):,} lignes × {df.shape[1]} colonnes (après filtres).")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Robots", f"{len(df):,}")

if "volumetrie_an" in df.columns:
    k2.metric("Volumétrie totale", f"{df['volumetrie_an'].sum():,.0f}")

if "reussite" in df.columns:
    k3.metric("Médiane % réussite", f"{df['reussite'].median()*100:,.1f}%")
    k4.metric("Robots < 60 % réussite", int((df["reussite"] < 0.60).sum()))

if "gains_etp" in df.columns:
    k5.metric("Gains ETP cumulés", f"{df['gains_etp'].sum():,.1f}")

###############################################################################
# TABS
###############################################################################
tab_over, tab_dist, tab_rel, tab_clust, tab_data = st.tabs(
    ["Vue d’ensemble", "Distributions", "Relations", "Clustering", "Données"]
)

# ───────────────── Vue d’ensemble ────────────────────────────────────────────
with tab_over:
    st.subheader("📈 Pareto 80/20 volumétrie")
    if {"volumetrie_an", "nom"}.issubset(df.columns):
        pareto = df.sort_values("volumetrie_an", ascending=False).reset_index(drop=True)
        pareto["cum_pct"] = pareto["volumetrie_an"].cumsum() / pareto["volumetrie_an"].sum()
        chart = alt.layer(
            alt.Chart(pareto.head(20)).mark_bar().encode(
                x=alt.X("nom:N", sort="-y", axis=alt.Axis(title=None, labels=False)),
                y="volumetrie_an:Q",
                tooltip=["nom", "volumetrie_an"],
            ),
            alt.Chart(pareto.head(20)).mark_line(point=True, color="#F39C12").encode(
                x="nom:N",
                y=alt.Y("cum_pct:Q", axis=alt.Axis(format="%")),
                tooltip=alt.Tooltip("cum_pct:Q", format=".1%"),
            ),
        ).resolve_scale(y="independent").properties(height=400)
        st.altair_chart(chart, use_container_width=True)

    if "zone_fonctionnelle" in df.columns:
        agg = (
            df.groupby("zone_fonctionnelle")
            .agg(
                robots=("id", "count") if "id" in df.columns else ("nom", "count"),
                volumetrie=("volumetrie_an", "sum"),
                reussite_moy=("reussite", "mean"),
            )
            .reset_index()
        )
        st.dataframe(agg, hide_index=True, use_container_width=True)

# ───────────────── Distributions ─────────────────────────────────────────────
with tab_dist:
    st.subheader("📊 Histogramme & box-plot")
    numcols: List[str] = df.select_dtypes(np.number).columns.tolist()
    if numcols:
        var = st.selectbox("Variable numérique", numcols)
        c1, c2 = st.columns(2)

        hist = alt.Chart(df).mark_bar().encode(
            x=alt.X(var, bin=alt.Bin(maxbins=50)),
            y="count()",
            tooltip=["count()"],
        ).properties(height=350)
        c1.altair_chart(hist, use_container_width=True)

        box = alt.Chart(df).mark_boxplot(extent="min-max").encode(
            y=var
        ).properties(height=350)
        c2.altair_chart(box, use_container_width=True)
    else:
        st.info("Aucune variable numérique.")

# ───────────────── Relations ────────────────────────────────────────────────
with tab_rel:
    st.subheader("🔗 Corrélations & scatter")
    num = df.select_dtypes(np.number).columns
    if len(num) >= 2:
        corr_df = corr_with_p(df)
        heat = alt.Chart(corr_df).mark_rect().encode(
            x="var_x:O",
            y="var_y:O",
            color=alt.Color("corr:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=[
                alt.Tooltip("corr:Q", format=".2f", title="r"),
                alt.Tooltip("pval:Q", format=".3e", title="p-value"),
            ],
        ).properties(height=400)
        st.altair_chart(heat, use_container_width=True)

        if {"volumetrie_an", "reussite"}.issubset(df.columns):
            scatter = alt.Chart(df).mark_circle(size=80, opacity=0.7).encode(
                x="volumetrie_an:Q",
                y=alt.Y("reussite:Q", axis=alt.Axis(format="%")),
                color=(
                    alt.Color("zone_fonctionnelle:N", legend=None)
                    if "zone_fonctionnelle" in df.columns
                    else alt.value("#1f77b4")
                ),
                tooltip=list(df.columns),
            ).interactive()

            trend = (
                alt.Chart(df)
                .transform_regression("volumetrie_an", "reussite")
                .mark_line(color="red")
            )
            st.altair_chart(scatter + trend, use_container_width=True)
    else:
        st.info("Pas assez de variables numériques.")

# ───────────────── Clustering ────────────────────────────────────────────────
with tab_clust:
    st.subheader("🎯 Clustering K-means")
    num_clean = df.select_dtypes(np.number).dropna()
    if num_clean.shape[0] >= 5 and num_clean.shape[1] >= 2:
        k = st.slider("Nombre de clusters", 2, 6, 3)
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(num_clean)
        df["cluster"] = km.labels_
        st.write(f"Inertie (distortion) : **{km.inertia_:,.0f}**")

        if {"volumetrie_an", "reussite"}.issubset(df.columns):
            fig = px.scatter(
                df,
                x="volumetrie_an",
                y="reussite",
                color="cluster",
                hover_data=df.columns,
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas assez de lignes ou de variables numériques pour clusteriser.")

# ───────────────── Données ───────────────────────────────────────────────────
with tab_data:
    st.subheader("📑 Profil statistique")
    st.dataframe(profile(df), use_container_width=True)

    st.subheader("🗃️ Tableau filtré")
    st.dataframe(df, use_container_width=True, height=450)

    @st.cache_data
    def csv_bytes(d: pd.DataFrame) -> bytes:
        return d.to_csv(index=False).encode("utf-8")

    st.download_button(
        "💾 Télécharger le CSV filtré",
        csv_bytes(df),
        file_name="robots_filtered.csv",
        mime="text/csv",
    )

###############################################################################
# Footer
###############################################################################
st.caption(f"Réalisé par {AUTHOR} • "
           f"[Code]({REPO}/blob/main/streamlit_app.py) • "
           "© 2025 Generali / Avanade")
