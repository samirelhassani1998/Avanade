# -*- coding: utf-8 -*-
"""
Streamlit – Inventaire RPA Generali (v2.3)
Auteur : Samir El Hassani · Avanade – 2025-06-05
"""

from __future__ import annotations

###############################################################################
# Imports
###############################################################################
import csv, io, itertools, re, unicodedata, warnings
from typing import List

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import streamlit as st
from sklearn.cluster import KMeans

###############################################################################
# Config Streamlit & Altair
###############################################################################
st.set_page_config("Inventaire RPA Generali", "🤖", layout="wide")
pd.options.display.float_format = "{:,.2f}".format
alt.data_transformers.disable_max_rows()

AUTHOR = "Samir El Hassani"
REPO   = "https://github.com/samirelhassani1998/Avanade"

###############################################################################
# Fonctions utilitaires (slugify, read_csv_robust, promote_header, to_num, clean)
###############################################################################
def slugify(txt: str) -> str:
    txt = (unicodedata.normalize("NFKD", str(txt))
           .encode("ascii", "ignore").decode("ascii").lower().strip())
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
    s = (s.astype(str).str.replace("%", "").str.replace(",", "").str.replace(" ", ""))
    n = pd.to_numeric(s, errors="coerce")
    return n / 100 if pct else n


def clean(df0: pd.DataFrame) -> pd.DataFrame:
    df = promote_header(df0)
    df.dropna(how="all", inplace=True)
    df.columns = [slugify(c) or f"col_{i}" for i, c in enumerate(df.columns)]
    for col, pct in {"volumetrie_an": False, "gains_etp": False, "reussite": True}.items():
        if col in df.columns:
            df[col] = to_num(df[col], pct)
    return df

###############################################################################
# Profil + corrélation robuste
###############################################################################
@st.cache_data(show_spinner=False)
def profile(df: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        desc = df.describe(include="all")
    total = len(df)
    prof = (desc.T.assign(dtype=df.dtypes.astype(str))
                  .reset_index().rename(columns={"index": "col"}))
    prof["count"]       = pd.to_numeric(prof["count"], errors="coerce").fillna(0)
    prof["missing"]     = (total - prof["count"]).astype(float)
    prof["missing_pct"] = (prof["missing"] / total * 100).round(1)
    return prof


@st.cache_data(show_spinner=False)
def corr_with_p(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(np.number)
    pairs, data = itertools.combinations(num.columns, 2), []
    for a, b in pairs:
        valid = num[[a, b]].dropna()
        if len(valid) < 3:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
            r, p = stats.pearsonr(valid[a], valid[b])
        data.append({"var_x": a, "var_y": b, "corr": r, "pval": p})
    return pd.DataFrame(data)

###############################################################################
# Sidebar : upload + filtres
###############################################################################
st.sidebar.header("📂 Charger le CSV")
up = st.sidebar.file_uploader("Fichier (UTF-8)", ["csv"])
if up is None:
    st.sidebar.info("Chargez un fichier pour commencer.")
    st.stop()

df = clean(read_csv_robust(up))

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
st.success(f"{len(df):,} lignes · {df.shape[1]} colonnes (après filtres).")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Robots", f"{len(df):,}")
if "volumetrie_an" in df.columns:
    k2.metric("Volumétrie totale", f"{df['volumetrie_an'].sum():,.0f}")
if "reussite" in df.columns:
    k3.metric("Médiane % réussite", f"{df['reussite'].median()*100:,.1f}%")
    k4.metric("Robots < 60 % réussite", int((df["reussite"] < .6).sum()))
if "gains_etp" in df.columns:
    k5.metric("Gains ETP cumulés", f"{df['gains_etp'].sum():,.1f}")

###############################################################################
# Tabs
###############################################################################
tab_over, tab_dist, tab_rel, tab_clust, tab_data = st.tabs(
    ["Vue d’ensemble", "Distributions", "Relations", "Clustering", "Données"]
)

# ─ Vue d’ensemble ───────────────────────────────────────────────────────────
with tab_over:
    st.subheader("📈 Pareto 80/20 volumétrie")
    if {"volumetrie_an", "nom"}.issubset(df.columns):
        pareto = df.sort_values("volumetrie_an", False).reset_index(drop=True)
        pareto["cum_pct"] = pareto["volumetrie_an"].cumsum() / pareto["volumetrie_an"].sum()
        st.altair_chart(
            alt.layer(
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
            ).resolve_scale(y="independent").properties(height=400),
            use_container_width=True,
        )
    if "zone_fonctionnelle" in df.columns:
        agg = (df.groupby("zone_fonctionnelle")
                 .agg(robots=("id" if "id" in df.columns else "nom", "count"),
                      volumetrie=("volumetrie_an", "sum"),
                      reussite_moy=("reussite", "mean"))
                 .reset_index())
        st.dataframe(agg, use_container_width=True, hide_index=True)

# ─ Distributions ────────────────────────────────────────────────────────────
with tab_dist:
    st.subheader("📊 Histogramme + box-plot")
    nums: List[str] = df.select_dtypes(np.number).columns.tolist()
    if nums:
        var = st.selectbox("Variable numérique", nums)
        c1, c2 = st.columns(2)
        c1.altair_chart(
            alt.Chart(df).mark_bar().encode(
                x=alt.X(var, bin=alt.Bin(maxbins=50)),
                y="count()",
                tooltip=["count()"],
            ).properties(height=350),
            use_container_width=True,
        )
        c2.altair_chart(
            alt.Chart(df).mark_boxplot(extent="min-max").encode(y=var).properties(height=350),
            use_container_width=True,
        )
    else:
        st.info("Aucune variable numérique détectée.")

# ─ Relations ────────────────────────────────────────────────────────────────
with tab_rel:
    st.subheader("🔗 Corrélations & scatter")
    if len(nums) >= 2:
        st.altair_chart(
            alt.Chart(corr_with_p(df)).mark_rect().encode(
                x="var_x:O",
                y="var_y:O",
                color=alt.Color("corr:Q", scale=alt.Scale(scheme="viridis")),
                tooltip=[
                    alt.Tooltip("corr:Q", format=".2f", title="r"),
                    alt.Tooltip("pval:Q", format=".3e", title="p-value"),
                ],
            ).properties(height=400),
            use_container_width=True,
        )
        if {"volumetrie_an", "reussite"}.issubset(df.columns):
            scatter = alt.Chart(df).mark_circle(size=80, opacity=0.7).encode(
                x="volumetrie_an:Q",
                y=alt.Y("reussite:Q", axis=alt.Axis(format="%")),
                color=alt.Color("zone_fonctionnelle:N", legend=None)
                      if "zone_fonctionnelle" in df.columns else alt.value("#1f77b4"),
                tooltip=list(df.columns),
            ).interactive()
            trend = (alt.Chart(df)
                       .transform_regression("volumetrie_an", "reussite")
                       .mark_line(color="red"))
            st.altair_chart(scatter + trend, use_container_width=True)
    else:
        st.info("Pas assez de numériques pour corrélations.")

# ─ Clustering ───────────────────────────────────────────────────────────────
with tab_clust:
    st.subheader("🎯 Clustering K‐means")
    num_dat = df.select_dtypes(np.number).dropna()
    if num_dat.shape[0] >= 5 and num_dat.shape[1] >= 2:
        k = st.slider("Nombre de clusters", 2, 6, 3)
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(num_dat)
        # Crée une colonne cluster alignée à l'index global
        df["cluster"] = np.nan
        df.loc[num_dat.index, "cluster"] = km.labels_.astype(int)

        st.write(f"Inertie (distortion) : **{km.inertia_:,.0f}**")

        if {"volumetrie_an", "reussite"}.issubset(df.columns):
            st.plotly_chart(
                px.scatter(
                    df.dropna(subset=["cluster"]),
                    x="volumetrie_an",
                    y="reussite",
                    color="cluster",
                    hover_data=df.columns,
                    height=500,
                ),
                use_container_width=True,
            )
    else:
        st.info("Pas assez de données numériques pour clusteriser.")

# ─ Données ──────────────────────────────────────────────────────────────────
with tab_data:
    st.subheader("📑 Profil statistique")
    st.dataframe(profile(df), use_container_width=True)
    st.subheader("🗃️ Tableau filtré")
    st.dataframe(df, use_container_width=True, height=450)

    @st.cache_data
    def csv_bytes(d: pd.DataFrame) -> bytes:
        return d.to_csv(index=False).encode("utf-8")

    st.download_button("💾 Télécharger le CSV filtré",
                       csv_bytes(df),
                       "robots_filtered.csv",
                       "text/csv")

###############################################################################
# Footer
###############################################################################
st.caption(f"Réalisé par {AUTHOR} • "
           f"[Code]({REPO}/blob/main/streamlit_app.py) • "
           "© 2025 Generali / Avanade")
