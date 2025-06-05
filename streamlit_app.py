# -*- coding: utf-8 -*-
"""
Streamlit – Inventaire RPA Generali (v3.4 • 2025-06-05)
Auteur : Samir El Hassani – Avanade
Correctifs & complétude :
 • lecture XLS/XLSX ou CSV (Sheet 1)
 • détection automatique de la bonne ligne d’en-têtes
 • mappage flexible des noms de colonnes
 • toutes les visualisations initiales + onglets Complexité / Tech-Stack
 • aucun crash si colonne manquante
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

# ════════════ CONFIG ════════════
st.set_page_config("Inventaire RPA Generali", "🤖", "wide")
pd.options.display.float_format = "{:,.2f}".format
alt.data_transformers.disable_max_rows()
AUTHOR, REPO = "Samir El Hassani", "https://github.com/samirelhassani1998/Avanade"

# ════════════ UTILS ════════════
def slugify(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", txt.lower())).strip("_")

def read_any(buf) -> pd.DataFrame:
    """Lit CSV ou Excel (1ʳᵉ feuille)."""
    name = pathlib.Path(getattr(buf, "name", "file")).suffix.lower()
    if name in (".xls", ".xlsx", ".xlsm"):
        return pd.read_excel(buf, sheet_name=0, dtype=str, keep_default_na=False)
    raw = buf.read()
    sample = raw[:2048].decode("utf-8", "replace")
    try:
        sep = csv.Sniffer().sniff(sample, [",", ";", "\t"]).delimiter
    except csv.Error:
        sep = ","
    buf.seek(0)
    return pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str, keep_default_na=False)

def promote_header_auto(df: pd.DataFrame) -> pd.DataFrame:
    """Si la 1ʳᵉ ligne contient « Id » & « Nom », on la prend comme header."""
    if len(df) and {"id", "nom"}.issubset(df.iloc[0].str.strip().str.lower()):
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = df.iloc[0].str.strip().str.lower()
    return df

def to_num(s: pd.Series, pct=False):
    s = (s.astype(str)
           .str.replace("%", "", regex=False)
           .str.replace(",", "", regex=False)
           .str.replace(" ", "", regex=False)
           .str.strip())
    n = pd.to_numeric(s, errors="coerce")
    return n/100 if pct else n

def clean(df0: pd.DataFrame) -> pd.DataFrame:
    df = promote_header_auto(df0)
    df.dropna(axis=1, how="all", inplace=True)
    df.columns = [slugify(c) or f"col_{i}" for i, c in enumerate(df.columns)]

    mapping = {"volumetrie_an": False, "gains_etp": False,
               "reussite": True, "%reussite": True}
    for raw, pct in mapping.items():
        col = next((c for c in df.columns if c.startswith(raw)), None)
        if col:
            df[col] = to_num(df[col], pct)
            df.rename(columns={col: raw}, inplace=True)

    if "synch_asynch" in df:            # mode synchrone / asynchrone
        df["synch_mode"] = df["synch_asynch"].str.lower().str.strip()
    if "complexite_s_m_l" in df:        # complexité S/M/L
        df["complexite_cat"] = (df["complexite_s_m_l"].str.upper().str[0]
                                .map({"S": "Small", "M": "Medium", "L": "Large"}))
    return df

@st.cache_data(show_spinner=False)
def profile(df: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = df.describe(include="all")
    total = len(df)
    p = (d.T.assign(dtype=df.dtypes.astype(str))
            .reset_index().rename(columns={"index": "col"}))
    p["count"] = pd.to_numeric(p["count"], errors="coerce").fillna(0)
    p["missing"] = total - p["count"]
    p["missing_pct"] = (p["missing"] / total * 100).round(1)
    return p

@st.cache_data(show_spinner=False)
def corr_with_p(df):
    num = df.select_dtypes(np.number)
    out = []
    for a, b in itertools.combinations(num.columns, 2):
        valid = num[[a, b]].dropna()
        if len(valid) < 3:
            continue
        r, p = stats.pearsonr(valid[a], valid[b])
        out.append({"var_x": a, "var_y": b, "corr": r, "pval": p})
    return pd.DataFrame(out)

# ════════════ SIDE BAR ════════════
st.sidebar.header("📂 Import")
upl = st.sidebar.file_uploader("Excel ou CSV", ["csv", "xls", "xlsx", "xlsm"])
if upl is None:
    st.sidebar.info("➡️ Déposez un fichier pour commencer."); st.stop()

df = clean(read_any(upl))

if st.sidebar.checkbox("👀 Colonnes", False):
    st.sidebar.write(df.columns.tolist())

with st.sidebar:
    st.markdown("### 🎛️ Filtres")
    filt_cat = {"zone_fonctionnelle": "Zone",
                "departement": "Département",
                "complexite_cat": "Complexité",
                "synch_mode": "Synch/Asynch"}
    for col, lab in filt_cat.items():
        if col in df and df[col].notna().any():
            sel = st.multiselect(lab, sorted(df[col].dropna().unique()),
                                 default=list(sorted(df[col].dropna().unique())))
            df = df[df[col].isin(sel)]

    if "volumetrie_an" in df:
        v = df["volumetrie_an"].dropna()
        r = st.slider("Volumétrie/an", float(v.min()), float(v.max()),
                      (float(v.min()), float(v.max())))
        df = df[df["volumetrie_an"].between(*r)]

    if "reussite" in df:
        pct = st.slider("% réussite", 0.0, 1.0, (0.0, 1.0))
        df = df[df["reussite"].between(*pct)]

# ════════════ KPI ════════════
st.success(f"{len(df):,} lignes × {df.shape[1]} colonnes sélectionnées")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Robots", f"{len(df):,}")
if "volumetrie_an" in df:
    k2.metric("Volumétrie totale", f"{df['volumetrie_an'].sum():,.0f}")
if "reussite" in df:
    k3.metric("Médiane % réussite", f"{df['reussite'].median()*100:,.1f}%")
    k4.metric("Robots < 60 %", int((df["reussite"] < .6).sum()))
if "gains_etp" in df:
    k5.metric("Gains ETP", f"{df['gains_etp'].sum():,.1f}")

# ════════════ TABS ════════════
tabs = st.tabs(["Vue générale", "Distributions", "Relations", "Performance",
                "Heat-Zones", "Clustering", "Complexité / Mode",
                "Tech-Stack", "Données"])
(tab_over, tab_dist, tab_rel, tab_perf,
 tab_heat, tab_clu, tab_comp, tab_tech, tab_data) = tabs

# ═══ 1. Vue générale (Pareto + agrégats) ═══
with tab_over:
    st.subheader("📈 Pareto 80/20 volumétrie")
    if {"volumetrie_an", "nom"}.issubset(df):
        pareto = (df.dropna(subset=["volumetrie_an"])
                    .sort_values("volumetrie_an", ascending=False)
                    .reset_index(drop=True))
        pareto["cum_pct"] = (pareto["volumetrie_an"].cumsum()
                             / pareto["volumetrie_an"].sum())
        st.altair_chart(
            alt.layer(
                alt.Chart(pareto.head(20)).mark_bar().encode(
                    x=alt.X("nom:N", sort="-y", axis=alt.Axis(title=None,
                                                              labels=False)),
                    y="volumetrie_an:Q",
                    tooltip=["nom", "volumetrie_an"]),
                alt.Chart(pareto.head(20)).mark_line(point=True,
                                                     color="#F39C12").encode(
                    x="nom:N",
                    y=alt.Y("cum_pct:Q", axis=alt.Axis(format="%")),
                    tooltip=alt.Tooltip("cum_pct:Q", format=".1%"))
            ).resolve_scale(y="independent").properties(height=400),
            use_container_width=True)

    if "zone_fonctionnelle" in df:
        st.dataframe(
            df.groupby("zone_fonctionnelle")
              .agg(robots=("nom", "count"),
                   volumetrie=("volumetrie_an", "sum"),
                   reussite_moy=("reussite", "mean"))
              .reset_index(),
            hide_index=True, use_container_width=True)

# ═══ 2. Distributions ═══
with tab_dist:
    st.subheader("📊 Histogramme & box-plot")
    nums = df.select_dtypes(np.number).columns
    if len(nums):
        var = st.selectbox("Variable numérique", nums)
        c1, c2 = st.columns(2)
        c1.altair_chart(
            alt.Chart(df.dropna(subset=[var]))
               .mark_bar().encode(x=alt.X(var, bin=alt.Bin(maxbins=50)),
                                  y='count()'),
            use_container_width=True)
        c2.altair_chart(
            alt.Chart(df.dropna(subset=[var])).mark_boxplot(extent="min-max")
               .encode(y=var),
            use_container_width=True)
    else:
        st.info("Aucune variable numérique détectée.")

# ═══ 3. Relations ═══
with tab_rel:
    st.subheader("🔗 Corrélations & nuage")
    nums = df.select_dtypes(np.number).columns
    if len(nums) >= 2:
        cd = corr_with_p(df)
        if not cd.empty:
            st.altair_chart(
                alt.Chart(cd).mark_rect().encode(
                    x="var_x:O", y="var_y:O",
                    color=alt.Color("corr:Q", scale=alt.Scale(scheme="viridis")),
                    tooltip=[alt.Tooltip("corr:Q", format=".2f"),
                             alt.Tooltip("pval:Q", format=".2e")]),
                use_container_width=True)
    if {"volumetrie_an", "reussite"}.issubset(df):
        base = df.dropna(subset=["volumetrie_an", "reussite"])
        st.altair_chart(
            (alt.Chart(base).mark_circle(size=70, opacity=.6)
                 .encode(x="volumetrie_an", y=alt.Y("reussite",
                                                    axis=alt.Axis(format="%")),
                         tooltip=list(df.columns)))
            + (alt.Chart(base).transform_regression("volumetrie_an",
                                                    "reussite")
                 .mark_line(color="red")),
            use_container_width=True)

# ═══ 4. Performance (distribution gaussienne) ═══
with tab_perf:
    st.subheader("📐 Ajustement gaussien")
    numcols = df.select_dtypes(np.number).columns
    if numcols.any():
        var = st.selectbox("Variable", numcols)
        data = df[var].dropna()
        if len(data) > 3:
            mu, sig = data.mean(), data.std()
            x = np.linspace(data.min(), data.max(), 200)
            fig = go.Figure()
            fig.add_histogram(x=data, nbinsx=40, histnorm='probability density',
                              name="Empirique")
            fig.add_scatter(x=x, y=stats.norm.pdf(x, mu, sig),
                            mode='lines', name=f"N({mu:.1f},{sig:.1f}²)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas assez de valeurs.")

# ═══ 5. Heat-Zones ═══
with tab_heat:
    st.subheader("🌡️ Heatmap Zone × classe de % réussite")
    needed = {"zone_fonctionnelle", "reussite", "volumetrie_an"}
    if needed.issubset(df):
        bins = [0, .6, .8, 1.01]
        labels = ["<60 %", "60‒80 %", "≥80 %"]
        df["classe_reussite"] = pd.cut(df["reussite"], bins=bins,
                                       labels=labels, include_lowest=True)
        metric = st.radio("Couleur basée sur :", ["Nombre de robots",
                                                  "Volumétrie totale"],
                          horizontal=True)
        if metric == "Nombre de robots":
            pv = (df.groupby(["zone_fonctionnelle", "classe_reussite"])
                    .size().reset_index(name="val"))
            titre = "Robots"
        else:
            pv = (df.groupby(["zone_fonctionnelle", "classe_reussite"])
                    ["volumetrie_an"].sum().reset_index(name="val"))
            titre = "Volumétrie"
        st.altair_chart(
            alt.Chart(pv).mark_rect().encode(
                x="classe_reussite:N", y="zone_fonctionnelle:N",
                color=alt.Color("val:Q", scale=alt.Scale(scheme="blues"),
                                title=titre),
                tooltip=["zone_fonctionnelle:N", "classe_reussite:N",
                         alt.Tooltip("val:Q", format=",.0f", title=titre)]),
            use_container_width=True)
    else:
        st.info("Colonnes manquantes.")

# ═══ 6. Clustering K-means ═══
with tab_clu:
    st.subheader("🎯 Clustering K-means")
    feats = ["volumetrie_an", "reussite", "gains_etp"]
    feats = [c for c in feats if c in df]
    numc = df[feats].dropna()
    if numc.shape[0] >= 5 and numc.shape[1] >= 2:
        ks = st.slider("k clusters", 2, 6, 3)
        scaled = StandardScaler().fit_transform(numc)
        km = KMeans(n_clusters=ks, random_state=0, n_init="auto").fit(scaled)
        df.loc[numc.index, "cluster"] = km.labels_
        st.metric("Inertie", f"{km.inertia_:,.0f}")
        x, y = st.selectbox("Axe X", feats, 0), st.selectbox("Axe Y", feats, 1)
        st.plotly_chart(
            px.scatter(df, x=x, y=y, color="cluster",
                       hover_data=["nom"]+feats), use_container_width=True)
    else:
        st.info("Minimum : 5 lignes & 2 variables numériques.")

# ═══ 7. Complexité / Synch ═══
with tab_comp:
    st.header("🔍 Complexité & Synch/Asynch")
    c1, c2 = st.columns(2)

    if "complexite_cat" in df:
        c1.plotly_chart(px.pie(df, names="complexite_cat", hole=.45,
                               title="Répartition par complexité"),
                        use_container_width=True)
        if "volumetrie_an" in df:
            vol = df.groupby("complexite_cat")["volumetrie_an"].sum().reset_index()
            c2.plotly_chart(px.bar(vol, x="complexite_cat", y="volumetrie_an",
                                   title="Volumétrie par complexité"),
                            use_container_width=True)

    if "synch_mode" in df:
        c2.plotly_chart(px.pie(df, names="synch_mode", hole=.45,
                               title="Synch vs Asynch"),
                        use_container_width=True)

    if {"volumetrie_an", "reussite", "complexite_cat"}.issubset(df):
        st.plotly_chart(px.scatter(df, x="volumetrie_an", y="reussite",
                                   color="complexite_cat",
                                   hover_data=["nom"],
                                   labels={"reussite": "% réussite"}),
                        use_container_width=True)

# ═══ 8. Tech-Stack ═══
with tab_tech:
    st.header("🛠️ Techno / Vendor / Cloud")
    if "techno" in df:
        st.plotly_chart(px.bar(df["techno"].value_counts().reset_index(),
                               x="index", y="techno",
                               labels={"index": "Technologie",
                                       "techno": "Robots"},
                               title="Nombre de robots par techno"),
                        use_container_width=True)
    if "vendor" in df:
        st.plotly_chart(px.bar(df["vendor"].value_counts().reset_index(),
                               x="index", y="vendor",
                               labels={"index": "Vendor", "vendor": "Robots"},
                               title="Robots par vendor"),
                        use_container_width=True)
    if {"zone_fonctionnelle", "departement", "vendor"}.issubset(df):
        st.plotly_chart(px.sunburst(df, path=["zone_fonctionnelle",
                                              "departement", "vendor"],
                                    values="volumetrie_an"
                                            if "volumetrie_an" in df else None,
                                    title="Sunburst Zone / Département / Vendor"),
                        use_container_width=True)

# ═══ 9. Données ═══
with tab_data:
    st.subheader("📑 Profil de table")
    st.dataframe(profile(df), use_container_width=True)

    st.subheader("🗃️ Données filtrées")
    st.dataframe(df, use_container_width=True, height=450)

    @st.cache_data
    def to_csv(d: pd.DataFrame): return d.to_csv(index=False).encode()

    st.download_button("💾 Télécharger CSV filtré",
                       to_csv(df), "robots_filtered.csv",
                       "text/csv")

# ═════════ FOOTER ═════════
st.caption(f"Réalisé par {AUTHOR} • [Code]({REPO}/blob/main/streamlit_app.py) • © 2025 Generali / Avanade")
