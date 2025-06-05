# -*- coding: utf-8 -*-
"""
Streamlit – Inventaire RPA Generali (v2)
Inclut :
  • upload robuste + nettoyage
  • filtres dynamiques (zone, volumétrie, réussite)
  • KPIs : volume, % réussite, gains, Pareto 80/20
  • onglets : Vue d’ensemble | Distributions | Relations | Clustering | Data
  • visualisations Altair + Plotly prêtes pour slides
  • export CSV nettoyé / filtré
Auteur : Samir El Hassani – Avanade · 2025-06-05
"""
from __future__ import annotations
import csv, io, re, unicodedata, itertools, warnings
from typing import List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import streamlit as st
from sklearn.cluster import KMeans

###############################################################################
# CONFIG GLOBALE
###############################################################################
st.set_page_config(page_title="Inventaire RPA Generali", page_icon="🤖", layout="wide")
pd.options.display.float_format = "{:,.2f}".format
alt.data_transformers.disable_max_rows()

AUTHOR = "Samir El Hassani"
REPO   = "https://github.com/samirelhassani1998/Avanade"

###############################################################################
# UTILITAIRES NETTOYAGE
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
   raw = buf.read()
    sample = raw[:2048].decode("utf-8", "replace")
    try:
        sep = csv.Sniffer().sniff(sample, [",", ";", "\t"]).delimiter
    except csv.Error:
        sep = ","
    buf.seek(0)
    return pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str, keep_default_na=False)

def promote_header(df: pd.DataFrame) -> pd.DataFrame:
    if any(c.lower().startswith("unnamed") for c in df.columns):
        first = df.iloc[0].tolist()
        if len(set(first)) == len(first):
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = first
    return df

def to_num(s: pd.Series, pct=False):
    s = (
        s.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)  # espaces insécables
    )
    n = pd.to_numeric(s, errors="coerce")
    return n / 100 if pct else n

def clean(df0: pd.DataFrame) -> pd.DataFrame:
    df = promote_header(df0)
    df.dropna(how="all", inplace=True)
    df.columns = [slugify(c) or f"col_{i}" for i, c in enumerate(df.columns)]

    mapping = {"volumetrie_an": False, "gains_etp": False, "reussite": True}
    for c, pct in mapping.items():
        if c in df.columns:
            df[c] = to_num(df[c], pct)
    return df

###############################################################################
# PROFIL + CORRÉLATION AVEC P-VALUE
###############################################################################
@st.cache_data(show_spinner=False)
def profile(df: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        desc = df.describe(include="all")
    p = (
        desc.T.assign(dtype=df.dtypes.astype(str))
        .reset_index()
        .rename(columns={"index": "col"})
    )
    total = len(df)
    p["missing"]      = total - p["count"].fillna(0)
    p["missing_pct"]  = (p["missing"] / total * 100).round(1)
    return p

def corr_with_p(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(np.number)
    combos = list(itertools.combinations(num.columns, 2))
    data = []
    for a, b in combos:
        r, p = stats.pearsonr(num[a].dropna(), num[b].dropna())
        data.append({"var_x": a, "var_y": b, "corr": r, "pval": p})
    return pd.DataFrame(data)

###############################################################################
# SIDEBAR – UPLOAD & FILTRES
###############################################################################
st.sidebar.header("📂 Charger le CSV")
file = st.sidebar.file_uploader("Fichier UTF-8", type="csv")

if not file:
    st.sidebar.info("➡️ Chargez un CSV pour démarrer.")
    st.stop()

df0  = read_csv_robust(file)
df   = clean(df0)

# Filtres
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
        vmin, vmax = df["volumetrie_an"].min(), df["volumetrie_an"].max()
        sel = st.slider("Plage volumétrie/an", float(vmin), float(vmax), (float(vmin), float(vmax)))
        df = df[df["volumetrie_an"].between(sel[0], sel[1])]

    if "reussite" in df.columns:
        rmin, rmax = 0.0, 1.0
        rs = st.slider("% Réussite", 0.0, 1.0, (0.0, 1.0))
        df = df[df["reussite"].between(rs[0], rs[1])]

###############################################################################
# KPIs
###############################################################################
st.success(f"{len(df):,} lignes × {df.shape[1]} colonnes après filtrage.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Robots", f"{len(df):,}")
if "volumetrie_an" in df.columns:
    k2.metric("Volumétrie totale", f"{df['volumetrie_an'].sum():,.0f}")
if "reussite" in df.columns:
    k3.metric("Médiane % réussite", f"{df['reussite'].median()*100:,.1f}%")
    k4.metric("Robots < 60 % réussite", int((df['reussite'] < 0.60).sum()))
if "gains_etp" in df.columns:
    k5.metric("Gains ETP cumulés", f"{df['gains_etp'].sum():,.1f}")

###############################################################################
# ONGLETS
###############################################################################
tab_over, tab_dist, tab_rel, tab_clust, tab_data = st.tabs(
    ["Vue d’ensemble", "Distributions", "Relations", "Clustering", "Données"]
)

# ----- Vue d'ensemble --------------------------------------------------------
with tab_over:
    st.subheader("📈 Pareto 80/20 volumétrie")
    if {"volumetrie_an", "nom"}.issubset(df.columns):
        pareto = df.sort_values("volumetrie_an", ascending=False).reset_index(drop=True)
        pareto["cum_pct"] = pareto["volumetrie_an"].cumsum() / pareto["volumetrie_an"].sum()
        chart = alt.layer(
            alt.Chart(pareto.head(20)).mark_bar().encode(
                x=alt.X("nom:N", sort="-y", axis=alt.Axis(title=None, labels=False)),
                y="volumetrie_an:Q",
                tooltip=["nom", "volumetrie_an"]
            ),
            alt.Chart(pareto.head(20)).mark_line(point=True, color="#F39C12").encode(
                x="nom:N",
                y=alt.Y("cum_pct:Q", axis=alt.Axis(format="%")),
                tooltip=alt.Tooltip("cum_pct:Q", format=".1%")
            )
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

# ----- Distributions ---------------------------------------------------------
with tab_dist:
    st.subheader("📊 Histogramme & box-plot")
    numcols = df.select_dtypes(np.number).columns
    if numcols.any():
        var = st.selectbox("Variable numérique", numcols)
        c1, c2 = st.columns(2)
        # Histogramme
        hist = alt.Chart(df).mark_bar().encode(
            x=alt.X(var, bin=alt.Bin(maxbins=50)),
            y="count()",
            tooltip=["count()"]
        ).properties(height=350)
        c1.altair_chart(hist, use_container_width=True)
        # Box-plot
        box = alt.Chart(df).mark_boxplot(extent="min-max").encode(y=var).properties(height=350)
        c2.altair_chart(box, use_container_width=True)
    else:
        st.info("Aucune variable numérique.")

# ----- Relations -------------------------------------------------------------
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

        # Scatter volumétrie vs réussite
        if {"volumetrie_an", "reussite"}.issubset(df.columns):
            scatter = alt.Chart(df).mark_circle(size=80, opacity=0.7).encode(
                x="volumetrie_an:Q",
                y=alt.Y("reussite:Q", axis=alt.Axis(format="%")),
                color=alt.Color("zone_fonctionnelle:N", legend=None) if "zone_fonctionnelle" in df.columns else alt.value("#1f77b4"),
                tooltip=list(df.columns),
            ).interactive()
            # Trendline (régression linéaire)
            trend = (
                alt.Chart(df)
                .transform_regression("volumetrie_an", "reussite")
                .mark_line(color="red")
            )
            st.altair_chart(scatter + trend, use_container_width=True)
    else:
        st.info("Pas assez de numériques.")

# ----- Clustering ------------------------------------------------------------
with tab_clust:
    st.subheader("🎯 Clustering K-means")
    num = df.select_dtypes(np.number).dropna()
    if num.shape[0] >= 5:
        k = st.slider("Nombre de clusters", 2, 6, 3)
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(num)
        df["cluster"] = km.labels_
        st.write(f"Inertie : {km.inertia_:,.0f}")
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

# ----- Data & export ---------------------------------------------------------
with tab_data:
    st.subheader("📑 Profil statistique")
    st.dataframe(profile(df), use_container_width=True)
    st.subheader("🗃️ Tableau complet")
    st.dataframe(df, use_container_width=True, height=450)

    @st.cache_data
    def csv_bytes(d: pd.DataFrame) -> bytes:
        return d.to_csv(index=False).encode("utf-8")

    st.download_button("💾 Télécharger le CSV filtré", csv_bytes(df), "robots_filtered.csv", "text/csv")

###############################################################################
# FOOTER
###############################################################################
st.caption(f"Réalisé par {AUTHOR} · [Code]({REPO}/blob/main/streamlit_app.py)")

