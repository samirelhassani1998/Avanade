# -*- coding: utf-8 -*-
"""
Streamlit – Inventaire RPA Generali  (v3.1 • 2025-06-05)
Auteur : Samir El Hassani – Avanade
Correctif : robustifier la section PCA/Clustering (StandardScaler)
"""

from __future__ import annotations
import csv, io, itertools, re, unicodedata, warnings
from typing import List

# ────────── Data / maths ──────────
import numpy as np
import pandas as pd
import scipy.stats as stats

# ────────── Viz ──────────
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

# ────────── Streamlit ──────────
import streamlit as st

# ────────── ML / traitement ─────
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler        # ← NOUVEAU

# (Imports OPTIONNELS : ne crashent pas si absents)
try:
    import seaborn as sns
    HAS_SEABORN = True
except ModuleNotFoundError:
    HAS_SEABORN = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ModuleNotFoundError:
    HAS_WORDCLOUD = False


# ───────────── Config générale ─────────────
st.set_page_config("Inventaire RPA Generali", "🤖", "wide")
pd.options.display.float_format = "{:,.2f}".format
alt.data_transformers.disable_max_rows()
AUTHOR, REPO = "Samir El Hassani", "https://github.com/samirelhassani1998/Avanade"

# ───────────── Fonctions utilitaires ───────
def slugify(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", str(txt)).encode("ascii","ignore").decode()
    return re.sub(r"_+","_",re.sub(r"[^0-9a-z]+","_",txt.lower())).strip("_")

def read_csv_robust(buf) -> pd.DataFrame:
    raw = buf.read()
    sample = raw[:2048].decode("utf-8","replace")
    try:  sep = csv.Sniffer().sniff(sample,[",",";","\t"]).delimiter
    except csv.Error: sep = ","
    buf.seek(0)
    return pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str, keep_default_na=False)

def promote_header(df: pd.DataFrame) -> pd.DataFrame:
    if any(c.lower().startswith("unnamed") for c in df.columns):
        first = df.iloc[0].tolist()
        if len(set(first)) == len(first):
            df = df.iloc[1:].reset_index(drop=True); df.columns = first
    return df

def to_num(s: pd.Series, pct=False):
    s = (s.astype(str)
           .str.replace("%","",regex=False)
           .str.replace(",","",regex=False)
           .str.replace(" ","",regex=False)   # espaces fines / insécables
           .str.replace(" ","",regex=False))
    n = pd.to_numeric(s, errors="coerce")
    return n/100 if pct else n

def clean(df0: pd.DataFrame) -> pd.DataFrame:
    df = promote_header(df0); df.dropna(how="all", inplace=True)
    df.columns = [slugify(c) or f"col_{i}" for i,c in enumerate(df.columns)]
    for col,pct in {"volumetrie_an":False,"gains_etp":False,"reussite":True}.items():
        if col in df.columns: df[col] = to_num(df[col], pct)
    return df

@st.cache_data(show_spinner=False)
def profile(df: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings(): warnings.simplefilter("ignore"); d = df.describe(include="all")
    p = (d.T.assign(dtype=df.dtypes.astype(str)).reset_index().rename(columns={"index":"col"}))
    total = len(df); p["count"] = pd.to_numeric(p["count"], errors="coerce").fillna(0)
    p["missing"] = (total-p["count"]).astype(float); p["missing_pct"] = (p["missing"]/total*100).round(1)
    return p

@st.cache_data(show_spinner=False)
def corr_with_p(df: pd.DataFrame) -> pd.DataFrame:
    num, out = df.select_dtypes(np.number), []
    for a,b in itertools.combinations(num.columns,2):
        valid = num[[a,b]].dropna()
        if len(valid)<3: continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
            r,p = stats.pearsonr(valid[a], valid[b])
        out.append({"var_x":a,"var_y":b,"corr":r,"pval":p})
    return pd.DataFrame(out)

# ───────────── Sidebar : Upload + Filtres ─────
st.sidebar.header("📂 Import")
upl = st.sidebar.file_uploader("CSV UTF-8", ["csv"])
if upl is None:
    st.sidebar.info("➡️ Déposez un fichier pour commencer."); st.stop()

df = clean(read_csv_robust(upl))

with st.sidebar:
    st.markdown("### 🎛️ Filtres")
    if "zone_fonctionnelle" in df and df["zone_fonctionnelle"].notna().any():
        zone_sel = st.multiselect("Zone fonctionnelle",
                    sorted(df["zone_fonctionnelle"].dropna().unique()),
                    default=list(sorted(df["zone_fonctionnelle"].dropna().unique())))
        df = df[df["zone_fonctionnelle"].isin(zone_sel)]
    if "volumetrie_an" in df and df["volumetrie_an"].notna().any():
        v = df["volumetrie_an"].dropna()
        rng = st.slider("Volumétrie/an", float(v.min()), float(v.max()),
                        (float(v.min()), float(v.max())))
        df = df[df["volumetrie_an"].between(*rng)]
    if "reussite" in df and df["reussite"].notna().any():
        r = st.slider("% réussite", 0.0, 1.0, (0.0, 1.0))
        df = df[df["reussite"].between(*r)]

# ───────────── KPIs ──────────────────────────
st.success(f"{len(df):,} lignes × {df.shape[1]} colonnes sélectionnées")
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Robots", f"{len(df):,}")
if "volumetrie_an" in df: k2.metric("Volumétrie totale", f"{df['volumetrie_an'].sum():,.0f}")
if "reussite" in df:
    k3.metric("Médiane % réussite", f"{df['reussite'].median()*100:,.1f}%")
    k4.metric("Robots < 60 %", int((df["reussite"]<0.6).sum()))
if "gains_etp" in df: k5.metric("Gains ETP", f"{df['gains_etp'].sum():,.1f}")

# ───────────── Onglets (5 → 7) ───────────────
tab_over, tab_dist, tab_rel, tab_perf, tab_heat, tab_clu, tab_data = st.tabs(
    ["Vue d’ensemble","Distributions","Relations",
     "Performance","Heat-Zones","Clustering","Données"])

# ═══ 1. Vue d’ensemble – Pareto, agrégats ═══
with tab_over:
    st.subheader("📈 Pareto 80/20 volumétrie")
    if {"volumetrie_an","nom"}.issubset(df) and df["volumetrie_an"].notna().any():
        pareto = (df.dropna(subset=["volumetrie_an"])
                    .sort_values("volumetrie_an", ascending=False)
                    .reset_index(drop=True))
        pareto["cum_pct"] = pareto["volumetrie_an"].cumsum()/pareto["volumetrie_an"].sum()
        st.altair_chart(
            alt.layer(
                alt.Chart(pareto.head(20)).mark_bar().encode(
                    x=alt.X("nom:N", sort="-y", axis=alt.Axis(title=None, labels=False)),
                    y="volumetrie_an:Q", tooltip=["nom","volumetrie_an"]),
                alt.Chart(pareto.head(20)).mark_line(point=True,color="#F39C12").encode(
                    x="nom:N", y=alt.Y("cum_pct:Q", axis=alt.Axis(format="%")),
                    tooltip=alt.Tooltip("cum_pct:Q",format=".1%"))
            ).resolve_scale(y="independent").properties(height=400),
            use_container_width=True)

    if "zone_fonctionnelle" in df:
        st.dataframe(
            df.groupby("zone_fonctionnelle")
              .agg(robots=("nom","count"),
                   volumetrie=("volumetrie_an","sum"),
                   reussite_moy=("reussite","mean"))
              .reset_index(),
            use_container_width=True, hide_index=True)

# ═══ 2. Distributions – histogrammes & box-plots ═══
with tab_dist:
    st.subheader("📊 Histogramme & box-plot")
    nums = df.select_dtypes(np.number).columns.tolist()
    if nums:
        var = st.selectbox("Variable numérique", nums, key="dist_var")
        col1,col2 = st.columns(2)
        # Histogramme
        if HAS_SEABORN:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            sns.histplot(df[var].dropna(), kde=True, ax=ax)
            col1.pyplot(fig, use_container_width=True)
        else:
            col1.altair_chart(alt.Chart(df.dropna(subset=[var])).mark_bar().encode(
                x=alt.X(var, bin=alt.Bin(maxbins=50)), y='count()'), use_container_width=True)
        # Box-plot
        col2.altair_chart(alt.Chart(df.dropna(subset=[var])).mark_boxplot(extent="min-max")
                          .encode(y=var), use_container_width=True)
    else: st.info("Aucune variable numérique")

# ═══ 3. Relations – corrélations & scatter ═══
with tab_rel:
    st.subheader("🔗 Corrélations (r + p-value) & Scatter")
    nums = df.select_dtypes(np.number).columns
    if len(nums)>=2:
        cd = corr_with_p(df)
        if not cd.empty:
            st.altair_chart(
                alt.Chart(cd).mark_rect().encode(
                    x="var_x:O", y="var_y:O",
                    color=alt.Color("corr:Q", scale=alt.Scale(scheme="viridis")),
                    tooltip=[alt.Tooltip("corr:Q",format=".2f"),
                             alt.Tooltip("pval:Q",format=".3e")]),
                use_container_width=True)
        if {"volumetrie_an","reussite"}.issubset(df):
            base = df.dropna(subset=["volumetrie_an","reussite"])
            st.altair_chart(
                (alt.Chart(base).mark_circle(size=70,opacity=0.6)
                    .encode(x="volumetrie_an", y=alt.Y("reussite",axis=alt.Axis(format="%")),
                            tooltip=list(df.columns)))
                + (alt.Chart(base).transform_regression("volumetrie_an","reussite")
                    .mark_line(color="red")), use_container_width=True)
    else: st.info("Pas assez de numériques.")

# ═══ 4. Performance – distribution Gaussienne ═══
with tab_perf:
    st.subheader("📐 Ajustement Gaussien (KDE + courbe normale)")
    numcols = df.select_dtypes(np.number).columns
    if numcols.any():
        var = st.selectbox("Variable à analyser", numcols, key="perf_var")
        data = df[var].dropna()
        if len(data) > 3:
            mu, sigma = data.mean(), data.std()
            x = np.linspace(data.min(), data.max(), 200)
            pdf = stats.norm.pdf(x, mu, sigma)

            fig = go.Figure()
            fig.add_histogram(x=data, nbinsx=40, histnorm='probability density',
                              name="Empirique", opacity=0.6)
            fig.add_scatter(x=x, y=pdf, mode='lines', name=f"N({mu:.1f}, {sigma:.1f}²)")
            fig.update_layout(height=450, bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Pas assez de valeurs")
    else: st.info("Aucune numérique")

# ═══ 5. Heat-Zones – Heatmap Zone × Classe de % réussite ═══
with tab_heat:
    st.subheader("🌡️ Zones fonctionnelles vs classes de % réussite")

    needed = {"zone_fonctionnelle", "reussite", "volumetrie_an"}
    if needed.issubset(df.columns) and df["zone_fonctionnelle"].notna().any():

        # 1) Construction de la classe de % réussite
        bins   = [0.0, 0.60, 0.80, 1.01]              # 1.01 pour inclure 100 %
        labels = ["< 60 %", "60 – 80 %", "≥ 80 %"]
        df["classe_reussite"] = pd.cut(
            df["reussite"], bins=bins, labels=labels, include_lowest=True
        )

        # 2) Choix de la métrique à agréger
        mode = st.radio("Couleur basée sur …",
                        ["Nombre de robots", "Volumétrie annuelle totale"],
                        horizontal=True)

        if mode == "Nombre de robots":
            pivot = (df.groupby(["zone_fonctionnelle", "classe_reussite"])
                       .size().reset_index(name="valeur"))
            titre_couleur = "Robots"
        else:
            pivot = (df.groupby(["zone_fonctionnelle", "classe_reussite"])["volumetrie_an"]
                       .sum().reset_index(name="valeur"))
            titre_couleur = "Volumétrie"

        # 3) Heat-map Altair
        chart = (
            alt.Chart(pivot)
               .mark_rect()
               .encode(
                   x=alt.X("classe_reussite:N", title="Classe de % réussite"),
                   y=alt.Y("zone_fonctionnelle:N", sort="-x", title="Zone fonctionnelle"),
                   color=alt.Color("valeur:Q", scale=alt.Scale(scheme="blues"), title=titre_couleur),
                   tooltip=["zone_fonctionnelle:N","classe_reussite:N",
                            alt.Tooltip("valeur:Q", format=",.0f", title=titre_couleur)]
               )
               .properties(height=max(300, 22*pivot["zone_fonctionnelle"].nunique()))
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Colonnes manquantes ou vides pour construire la heat-map.")

# ═══ 6. Clustering – K-means (avec garde StandardScaler) ═══
with tab_clu:
    st.subheader("🎯 K-means & Analyse PCA (simplifiée)")

    # a) Sélection des variables numériques exploitables
    potentielles = ['volumetrie_an','reussite','gains_etp']
    num_presentes = [c for c in potentielles if c in df.columns]
    num_clean = df[num_presentes].dropna()

    if num_clean.shape[0] >= 5 and num_clean.shape[1] >= 2:
        # b) Standardisation protégée
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(num_clean.values)

        # c) K-means
        k = st.slider("k clusters", 2, 6, 3)
        km = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(data_scaled)
        df["cluster"] = pd.Series(km.labels_, index=num_clean.index).reindex(df.index)
        st.write(f"Inertie : **{km.inertia_:,.0f}**")

        # d) Visu scatter 2D (on prend d’office les 2 premières variables)
        x_var, y_var = num_presentes[:2]
        st.plotly_chart(
            px.scatter(df, x=x_var, y=y_var, color="cluster", hover_data=df.columns,
                       height=500), use_container_width=True)
    else:
        st.info("Pas assez de données (≥ 5 lignes et ≥ 2 variables numériques) pour exécuter le clustering.")

# ═══ 7. Données – profil & export ═══
with tab_data:
    st.subheader("📑 Profil")
    st.dataframe(profile(df), use_container_width=True)

    st.subheader("🗃️ Table")
    st.dataframe(df, use_container_width=True, height=450)

    @st.cache_data
    def to_csv(d: pd.DataFrame) -> bytes: return d.to_csv(index=False).encode()
    st.download_button("💾 CSV filtré", to_csv(df), "robots_filtered.csv", "text/csv")

# ───────────── Footer ──────────────────────
st.caption(f"Réalisé par {AUTHOR} • [Code]({REPO}/blob/main/streamlit_app.py) • © 2025 Generali / Avanade")
