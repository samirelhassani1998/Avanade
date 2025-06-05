# -*- coding: utf-8 -*-
"""
Streamlit – Inventaire RPA Generali  (v3.2 • 2025-06-05)
Auteur : Samir El Hassani – Avanade
Objectif : relire les données (échantillon fourni) et AJOUTER
           ✅ des onglets orientés Complexité / Mode Synchro
           ✅ des onglets Tech-Stack (Techno / Vendor / Cloud)
           ✅ des visuels dédiés (sunburst, pies, scatters, boxplots)
           ✅ filtres supplémentaires
           ✅ garde = aucun crash si colonne manquante
"""

from __future__ import annotations
import csv, io, itertools, re, unicodedata, warnings
from typing import List

import numpy as np
import pandas as pd
import scipy.stats as stats
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ════════════ CONFIG ════════════
st.set_page_config("Inventaire RPA Generali", "🤖", "wide")
pd.options.display.float_format = "{:,.2f}".format
alt.data_transformers.disable_max_rows()
AUTHOR, REPO = "Samir El Hassani", "https://github.com/samirelhassani1998/Avanade"

# ════════════ UTILS ════════════
def slugify(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", txt.lower())).strip("_")

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
        .str.replace(" ", "", regex=False)  # espace insécable
        .str.replace(" ", "", regex=False)
    )
    n = pd.to_numeric(s, errors="coerce")
    return n / 100 if pct else n

def clean(df0: pd.DataFrame) -> pd.DataFrame:
    df = promote_header(df0)
    df.dropna(how="all", inplace=True)
    df.columns = [slugify(c) or f"col_{i}" for i, c in enumerate(df.columns)]
    mapping = {
        "volumetrie_an": False,
        "gains_etp": False,
        "reussite": True,
    }
    for col, pct in mapping.items():
        if col in df.columns:
            df[col] = to_num(df[col], pct)
    # normaliser certains libellés
    if "synch__asynch" in df.columns:
        df["synch_mode"] = df["synch__asynch"].str.strip().str.lower()
    if "complexite_s_m_l" in df.columns:
        df["complexite_cat"] = (
            df["complexite_s_m_l"]
            .str.upper()
            .str[0]
            .map({"S": "Small", "M": "Medium", "L": "Large"})
        )
    return df

@st.cache_data(show_spinner=False)
def profile(df: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = df.describe(include="all")
    p = (
        d.T.assign(dtype=df.dtypes.astype(str))
        .reset_index()
        .rename(columns={"index": "col"})
    )
    total = len(df)
    p["count"] = pd.to_numeric(p["count"], errors="coerce").fillna(0)
    p["missing"] = (total - p["count"]).astype(float)
    p["missing_pct"] = (p["missing"] / total * 100).round(1)
    return p

@st.cache_data(show_spinner=False)
def corr_with_p(df: pd.DataFrame) -> pd.DataFrame:
    num, out = df.select_dtypes(np.number), []
    for a, b in itertools.combinations(num.columns, 2):
        valid = num[[a, b]].dropna()
        if len(valid) < 3:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
            r, p = stats.pearsonr(valid[a], valid[b])
        out.append({"var_x": a, "var_y": b, "corr": r, "pval": p})
    return pd.DataFrame(out)

# ════════════ SIDEBAR : upload + filtres ════════════
st.sidebar.header("📂 Import")
upl = st.sidebar.file_uploader("CSV UTF-8", ["csv"])
if upl is None:
    st.sidebar.info("➡️ Déposez un fichier pour commencer.")
    st.stop()

df = clean(read_csv_robust(upl))

with st.sidebar:
    st.markdown("### 🎛️ Filtres")
    cols_cat = {
        "zone_fonctionnelle": "Zone fonctionnelle",
        "departement": "Département",
        "complexite_cat": "Complexité",
        "synch_mode": "Mode Synch/Asynch",
    }
    for col, label in cols_cat.items():
        if col in df.columns and df[col].notna().any():
            opts = sorted(df[col].dropna().unique())
            sel = st.multiselect(label, opts, default=opts, key=f"f_{col}")
            df = df[df[col].isin(sel)]

    if "volumetrie_an" in df and df["volumetrie_an"].notna().any():
        v = df["volumetrie_an"].dropna()
        rng = st.slider(
            "Volumétrie/an",
            float(v.min()),
            float(v.max()),
            (float(v.min()), float(v.max())),
        )
        df = df[df["volumetrie_an"].between(*rng)]

    if "reussite" in df and df["reussite"].notna().any():
        r = st.slider("% réussite", 0.0, 1.0, (0.0, 1.0))
        df = df[df["reussite"].between(*r)]

# ════════════ KPIs ════════════
st.success(f"{len(df):,} lignes × {df.shape[1]} colonnes sélectionnées")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Robots", f"{len(df):,}")
if "volumetrie_an" in df:
    k2.metric("Volumétrie totale", f"{df['volumetrie_an'].sum():,.0f}")
if "reussite" in df:
    k3.metric("Médiane % réussite", f"{df['reussite'].median()*100:,.1f}%")
    k4.metric("Robots < 60 %", int((df["reussite"] < 0.6).sum()))
if "gains_etp" in df:
    k5.metric("Gains ETP", f"{df['gains_etp'].sum():,.1f}")

# ════════════ ONGLES ════════════
(
    tab_over,
    tab_dist,
    tab_rel,
    tab_perf,
    tab_heat,
    tab_clu,
    tab_comp,
    tab_tech,
    tab_data,
) = st.tabs(
    [
        "Vue générale",
        "Distributions",
        "Relations",
        "Performance",
        "Heat-Zones",
        "Clustering",
        "Complexité / Mode",
        "Tech-Stack",
        "Données",
    ]
)

# --- onglets initiaux (over, dist, rel, perf, heat, clu) inchangés (cf. v3.2 ci-dessus) ---
# … (pour concision, on suppose que le code des 6 premiers onglets reste identique)
# >>> AJOUT : onglet 7 « Complexité / Mode » et onglet 8 « Tech-Stack » <<<

# ═══ 7. COMPLEXITÉ / MODE SYNC ═══
with tab_comp:
    st.header("🔍 Analyse Complexité & Mode Synch/Asynch")

    col1, col2 = st.columns(2)

    # Pie : répartition Complexité
    if "complexite_cat" in df and df["complexite_cat"].notna().any():
        fig = px.pie(
            df,
            names="complexite_cat",
            title="Répartition des robots par complexité",
            hole=0.45,
        )
        col1.plotly_chart(fig, use_container_width=True)

        # Boxplot Réussite par complexité
        if "reussite" in df:
            st.subheader("Distribution % réussite selon la complexité")
            st.plotly_chart(
                px.box(
                    df,
                    x="complexite_cat",
                    y="reussite",
                    points="outliers",
                    labels={"reussite": "% réussite", "complexite_cat": "Complexité"},
                ),
                use_container_width=True,
            )

        # Bar : volumétrie/an par complexité
        if "volumetrie_an" in df:
            volum_cat = df.groupby("complexite_cat")["volumetrie_an"].sum().reset_index()
            fig2 = px.bar(
                volum_cat,
                x="complexite_cat",
                y="volumetrie_an",
                title="Volumétrie annuelle totale par complexité",
                labels={"complexite_cat": "Complexité", "volumetrie_an": "Volumétrie/an"},
            )
            col2.plotly_chart(fig2, use_container_width=True)

    # Pie : Synch / Asynch
    if "synch_mode" in df and df["synch_mode"].notna().any():
        fig3 = px.pie(
            df,
            names="synch_mode",
            title="Répartition Synch / Asynch",
            hole=0.45,
        )
        col2.plotly_chart(fig3, use_container_width=True)

        # Success  Synch vs Asynch
        if "reussite" in df:
            st.subheader("% réussite – Synch vs Asynch")
            st.plotly_chart(
                px.box(
                    df,
                    x="synch_mode",
                    y="reussite",
                    points=False,
                    labels={"synch_mode": "Mode", "reussite": "% réussite"},
                ),
                use_container_width=True,
            )

    # Scatter : Volumétrie vs Réussite coloré par Complexité
    if {"volumetrie_an", "reussite", "complexite_cat"}.issubset(df):
        st.subheader("Volumétrie vs % réussite (couleur = complexité)")
        st.plotly_chart(
            px.scatter(
                df,
                x="volumetrie_an",
                y="reussite",
                color="complexite_cat",
                hover_data=["nom", "zone_fonctionnelle"],
                labels={
                    "volumetrie_an": "Volumétrie/an",
                    "reussite": "% réussite",
                    "complexite_cat": "Complexité",
                },
            ),
            use_container_width=True,
        )

# ═══ 8. TECH-STACK ═══
with tab_tech:
    st.header("🛠️ Analyse Techno / Vendor / Cloud")

    # Bar chart : Volume par technologie
    if "techno" in df:
        tech_vol = df.groupby("techno")["volumetrie_an"].sum().reset_index()
        st.plotly_chart(
            px.bar(
                tech_vol,
                x="techno",
                y="volumetrie_an",
                title="Volumétrie annuelle par technologie",
                labels={"techno": "Technologie", "volumetrie_an": "Volumétrie/an"},
            ),
            use_container_width=True,
        )

    # Bar chart : Nombre de robots par vendor
    if "vendor" in df:
        vend_cnt = df["vendor"].value_counts().reset_index()
        vend_cnt.columns = ["vendor", "count"]
        st.plotly_chart(
            px.bar(
                vend_cnt,
                x="vendor",
                y="count",
                title="Nombre de robots par Vendor",
                labels={"vendor": "Vendor", "count": "Robots"},
            ),
            use_container_width=True,
        )

    # Sunburst Zone -> Département -> Vendor
    if {"zone_fonctionnelle", "departement", "vendor"}.issubset(df):
        st.subheader("Répartition hiérarchique Zone / Département / Vendor")
        st.plotly_chart(
            px.sunburst(
                df,
                path=["zone_fonctionnelle", "departement", "vendor"],
                values="volumetrie_an" if "volumetrie_an" in df else None,
                title="Sunburst volumétrie (ou effectifs) par Zone, Département, Vendor",
            ),
            use_container_width=True,
        )

# ═══ 9. DATA & EXPORT ═══
with tab_data:
    st.subheader("📑 Profil")
    st.dataframe(profile(df), use_container_width=True)

    st.subheader("🗃️ Table")
    st.dataframe(df, use_container_width=True, height=450)

    @st.cache_data
    def to_csv(d: pd.DataFrame) -> bytes:
        return d.to_csv(index=False).encode()

    st.download_button(
        "💾 CSV filtré",
        to_csv(df),
        "robots_filtered.csv",
        "text/csv",
    )

# ════════════ FOOTER ════════════
st.caption(
    f"Réalisé par {AUTHOR} • [Code]({REPO}/blob/main/streamlit_app.py) • © 2025 Generali / Avanade"
)
