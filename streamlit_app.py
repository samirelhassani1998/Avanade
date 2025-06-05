# -*- coding: utf-8 -*-
"""
Streamlit – Inventaire RPA Generali  (v3.5 • 2025-06-05)
Auteur : Samir El Hassani – Avanade

✅ Lit indifféremment un CSV (séparateur auto) *ou* la feuille 1 d’un Excel  
✅ Repère dynamiquement la VRAIE ligne d’en-têtes (celle qui contient « Id » & « Nom »)  
✅ Nettoie / renomme les colonnes (slugify) →  zone_fonctionnelle, volumetrie_an, reussite, …  
✅ Toutes les visualisations (9 onglets) fonctionnent même si certaines colonnes manquent  
"""

from __future__ import annotations
import csv, io, itertools, re, unicodedata, warnings, pathlib

import numpy as np
import pandas as pd
import scipy.stats as stats
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─────────────────── CONFIG
st.set_page_config("Inventaire RPA Generali", "🤖", "wide")
pd.options.display.float_format = "{:,.2f}".format
alt.data_transformers.disable_max_rows()
AUTHOR = "Samir El Hassani"

# ─────────────────── UTILS
def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", text.lower())).strip("_")

def read_any(buf) -> pd.DataFrame:
    """Lit Excel (1ʳᵉ feuille) ou CSV (séparateur auto, engine='python')."""
    ext = pathlib.Path(buf.name).suffix.lower()
    if ext in (".xls", ".xlsx", ".xlsm"):
        return pd.read_excel(buf, sheet_name=0, dtype=str, keep_default_na=False)
    # CSV : détection robuste du séparateur avec pandas engine python
    buf.seek(0)
    return pd.read_csv(buf, dtype=str, keep_default_na=False,
                       sep=None, engine="python")

def promote_header(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cherche la 1ʳᵉ ligne qui contient à la fois « Id » et « Nom ».
    La transforme en en-têtes et supprime les lignes précédentes.
    """
    for i in range(min(10, len(df))):
        row = df.iloc[i].str.strip().str.lower()
        if {"id", "nom"}.issubset(set(row)):
            df = df.iloc[i + 1:].reset_index(drop=True)
            df.columns = row
            break
    return df

def to_num(s: pd.Series, pct=False):
    s = (s.astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.strip())
    n = pd.to_numeric(s, errors="coerce")
    return n / 100 if pct else n

def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = promote_header(raw)
    df.dropna(axis=1, how="all", inplace=True)
    df.columns = [slugify(c) or f"col_{i}" for i, c in enumerate(df.columns)]

    # cast numériques
    mapping = {"volumetrie_an": False, "gains_etp": False, "reussite": True}
    for base, pct in mapping.items():
        col = next((c for c in df.columns if c.startswith(base)), None)
        if col:
            df[col] = to_num(df[col], pct)
            df.rename(columns={col: base}, inplace=True)

    # libellés dérivés
    if "synch_asynch" in df:
        df["synch_mode"] = df["synch_asynch"].str.lower().str.strip()

    if "complexite_s_m_l" in df:
        df["complexite_cat"] = (df["complexite_s_m_l"].str.upper().str[0]
                                .map({"S": "Small", "M": "Medium", "L": "Large"}))
    return df

@st.cache_data(show_spinner=False)
def profile(df: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        desc = df.describe(include="all")
    total = len(df)
    p = (desc.T.assign(dtype=df.dtypes.astype(str))
                 .reset_index().rename(columns={"index": "col"}))
    p["missing"] = total - pd.to_numeric(p["count"].fillna(0))
    p["missing_pct"] = (p["missing"] / total * 100).round(1)
    return p

@st.cache_data(show_spinner=False)
def corr_with_p(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(np.number)
    out = []
    for a, b in itertools.combinations(num.columns, 2):
        valid = num[[a, b]].dropna()
        if len(valid) > 2:
            r, p = stats.pearsonr(valid[a], valid[b])
            out.append({"var_x": a, "var_y": b, "corr": r, "pval": p})
    return pd.DataFrame(out)

# ─────────────────── SIDEBAR : import
st.sidebar.header("📂 Import")
file = st.sidebar.file_uploader("Feuille 1 (Excel) ou CSV", ["csv", "xls", "xlsx", "xlsm"])
if not file:
    st.sidebar.info("➡️ Déposez un fichier pour commencer"); st.stop()

df = clean(read_any(file))

if st.sidebar.checkbox("Voir toutes les colonnes 🌐"):
    st.sidebar.write(df.columns.tolist())

# ─────────────────── FILTRES
with st.sidebar:
    st.markdown("### 🎛️ Filtres")
    cat_map = {"zone_fonctionnelle": "Zone", "departement": "Département",
               "complexite_cat": "Complexité", "synch_mode": "Synch/Asynch"}
    for col, lab in cat_map.items():
        if col in df and df[col].notna().any():
            opts = sorted(df[col].dropna().unique())
            df = df[df[col].isin(st.multiselect(lab, opts, default=opts))]

    if "volumetrie_an" in df:
        vmin, vmax = float(df["volumetrie_an"].min()), float(df["volumetrie_an"].max())
        rng = st.slider("Volumétrie/an", vmin, vmax, (vmin, vmax))
        df = df[df["volumetrie_an"].between(*rng)]

    if "reussite" in df:
        pct_low, pct_hi = st.slider("% réussite", 0.0, 1.0, (0.0, 1.0))
        df = df[df["reussite"].between(pct_low, pct_hi)]

# ─────────────────── KPI
st.success(f"{len(df):,} lignes – {df.shape[1]} colonnes")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Robots", f"{len(df):,}")
if "volumetrie_an" in df:
    k2.metric("Volumétrie totale", f"{df['volumetrie_an'].sum():,.0f}")
if "reussite" in df:
    k3.metric("Médiane réussite", f"{df['reussite'].median()*100:,.1f}%")
    k4.metric("Réussite < 60 %", (df["reussite"] < .6).sum())
if "gains_etp" in df:
    k5.metric("Gains ETP totaux", f"{df['gains_etp'].sum():,.1f}")

# ─────────────────── ONGLETs
tabs = st.tabs(["Vue générale", "Distributions", "Relations", "Performance",
                "Heat-Zones", "Clustering", "Complexité/Mode",
                "Tech-Stack", "Données"])
(tab_over, tab_dist, tab_rel, tab_perf,
 tab_heat, tab_clu, tab_comp, tab_tech, tab_data) = tabs

# ═════ 1. Vue générale
with tab_over:
    st.subheader("📈 Pareto volumétrie (Top 20)")
    if {"volumetrie_an", "nom"}.issubset(df):
        pareto = (df.dropna(subset=["volumetrie_an"])
                    .sort_values("volumetrie_an", ascending=False)
                    .reset_index(drop=True))
        pareto["cum"] = pareto["volumetrie_an"].cumsum()/pareto["volumetrie_an"].sum()
        st.altair_chart(
            alt.layer(
                alt.Chart(pareto.head(20)).mark_bar().encode(
                    x=alt.X("nom:N", sort="-y", axis=None),
                    y="volumetrie_an:Q", tooltip=["nom", "volumetrie_an"]),
                alt.Chart(pareto.head(20)).mark_line(point=True,
                                                     color="#E67E22").encode(
                    x="nom:N", y=alt.Y("cum:Q", axis=alt.Axis(format="%")),
                    tooltip=alt.Tooltip("cum:Q", format=".1%"))
            ).resolve_scale(y="independent"),
            use_container_width=True)

    if "zone_fonctionnelle" in df:
        st.dataframe(df.groupby("zone_fonctionnelle")
                       .agg(robots=("nom", "count"),
                            volumetrie=("volumetrie_an", "sum"),
                            reussite_moy=("reussite", "mean"))
                       .reset_index(),
                     hide_index=True, use_container_width=True)

# ═════ 2. Distributions
with tab_dist:
    st.subheader("📊 Histogrammes & box-plots")
    numcols = df.select_dtypes(np.number).columns
    if numcols.any():
        var = st.selectbox("Variable numérique", numcols)
        c1, c2 = st.columns(2)
        c1.altair_chart(alt.Chart(df.dropna(subset=[var]))
                          .mark_bar().encode(x=alt.X(var, bin=True), y='count()'),
                          use_container_width=True)
        c2.altair_chart(alt.Chart(df.dropna(subset=[var]))
                          .mark_boxplot(extent="min-max").encode(y=var),
                          use_container_width=True)
    else:
        st.info("Pas de numérique.")

# ═════ 3. Relations
with tab_rel:
    st.subheader("🔗 Corrélations")
    if df.select_dtypes(np.number).shape[1] >= 2:
        heat = corr_with_p(df)
        if not heat.empty:
            st.altair_chart(
                alt.Chart(heat).mark_rect().encode(
                    x="var_x:O", y="var_y:O",
                    color=alt.Color("corr:Q", scale=alt.Scale(scheme="viridis")),
                    tooltip=[alt.Tooltip("corr:Q", format=".2f"),
                             alt.Tooltip("pval:Q", format=".1e")]),
                use_container_width=True)

    if {"volumetrie_an", "reussite"}.issubset(df):
        st.subheader("Scatter Volume vs % réussite")
        st.altair_chart(
            (alt.Chart(df.dropna(subset=["volumetrie_an", "reussite"]))
                 .mark_circle(size=70, opacity=.6)
                 .encode(x="volumetrie_an", y=alt.Y("reussite",
                                                    axis=alt.Axis(format="%")),
                         tooltip=["nom"]))
            + (alt.Chart(df).transform_regression("volumetrie_an", "reussite")
                 .mark_line(color="red")),
            use_container_width=True)

# ═════ 4. Performance
with tab_perf:
    st.subheader("📐 Ajustement normal")
    numcols = df.select_dtypes(np.number).columns
    if numcols.any():
        choice = st.selectbox("Variable", numcols)
        series = df[choice].dropna()
        if len(series) > 3:
            mu, sd = series.mean(), series.std()
            xs = np.linspace(series.min(), series.max(), 200)
            fig = go.Figure()
            fig.add_histogram(x=series, nbinsx=40, histnorm="probability density")
            fig.add_scatter(x=xs, y=stats.norm.pdf(xs, mu, sd), mode="lines")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas assez de valeurs.")

# ═════ 5. Heat-Zones
with tab_heat:
    st.subheader("🌡️ Zones vs classes de réussite")
    required = {"zone_fonctionnelle", "reussite", "volumetrie_an"}
    if required.issubset(df):
        df["classe"] = pd.cut(df["reussite"], [0, .6, .8, 1.01],
                              labels=["<60 %", "60-80 %", "≥80 %"],
                              include_lowest=True)
        mode = st.radio("Couleur par …", ["Nombre", "Volumétrie"], horizontal=True)
        if mode == "Nombre":
            pivot = (df.groupby(["zone_fonctionnelle", "classe"])
                       .size().reset_index(name="val"))
            titre = "Robots"
        else:
            pivot = (df.groupby(["zone_fonctionnelle", "classe"])
                       ["volumetrie_an"].sum().reset_index(name="val"))
            titre = "Volumétrie"
        st.altair_chart(alt.Chart(pivot).mark_rect().encode(
            x="classe:N", y="zone_fonctionnelle:N",
            color=alt.Color("val:Q", scale=alt.Scale(scheme="blues"), title=titre),
            tooltip=["zone_fonctionnelle:N", "classe:N",
                     alt.Tooltip("val:Q", format=",.0f")]),
            use_container_width=True)
    else:
        st.info("Colonnes indispensables absentes.")

# ═════ 6. Clustering
with tab_clu:
    st.subheader("🎯 K-means")
    sel = [c for c in ("volumetrie_an", "reussite", "gains_etp") if c in df]
    data = df[sel].dropna()
    if data.shape[0] >= 5 and data.shape[1] >= 2:
        k = st.slider("k", 2, 6, 3)
        km = KMeans(n_clusters=k, n_init="auto", random_state=0)
        df.loc[data.index, "cluster"] = km.fit_predict(StandardScaler().fit_transform(data))
        st.metric("Inertie", f"{km.inertia_:,.0f}")
        x, y = st.selectbox("X", sel, 0), st.selectbox("Y", sel, 1)
        st.plotly_chart(px.scatter(df, x=x, y=y, color="cluster",
                                   hover_data=["nom"]), use_container_width=True)
    else:
        st.info("≥ 5 lignes & 2 variables requises.")

# ═════ 7. Complexité / Mode
with tab_comp:
    st.header("🔍 Complexité & Synch/Asynch")
    if "complexite_cat" in df:
        st.plotly_chart(px.pie(df, names="complexite_cat", hole=.45,
                               title="Répartition par complexité"),
                        use_container_width=True)
    if "synch_mode" in df:
        st.plotly_chart(px.pie(df, names="synch_mode", hole=.45,
                               title="Synch / Asynch"),
                        use_container_width=True)
    if {"volumetrie_an", "reussite", "complexite_cat"}.issubset(df):
        st.plotly_chart(px.scatter(df, x="volumetrie_an", y="reussite",
                                   color="complexite_cat"), use_container_width=True)

# ═════ 8. Tech-Stack
with tab_tech:
    st.header("🛠️ Techno / Vendor / Cloud")
    if "techno" in df:
        st.plotly_chart(px.bar(df["techno"].value_counts().reset_index(),
                               x="index", y="techno",
                               labels={"index": "Technologie", "techno": "Robots"}),
                        use_container_width=True)
    if "vendor" in df:
        st.plotly_chart(px.bar(df["vendor"].value_counts().reset_index(),
                               x="index", y="vendor",
                               labels={"index": "Vendor", "vendor": "Robots"}),
                        use_container_width=True)
    if {"zone_fonctionnelle", "departement", "vendor"}.issubset(df):
        st.plotly_chart(px.sunburst(df, path=["zone_fonctionnelle",
                                              "departement", "vendor"],
                                    values="volumetrie_an"
                                     if "volumetrie_an" in df else None),
                        use_container_width=True)

# ═════ 9. Données
with tab_data:
    st.subheader("Profil")
    st.dataframe(profile(df), use_container_width=True)

    st.subheader("Table filtrée")
    st.dataframe(df, use_container_width=True, height=450)

    @st.cache_data
    def to_csv(d: pd.DataFrame): return d.to_csv(index=False).encode()

    st.download_button("💾 Exporter CSV", to_csv(df), "robots_filtered.csv",
                       "text/csv")

# ─────────────────── FOOTER
st.caption(f"Réalisé par {AUTHOR} • © 2025 Generali / Avanade")
