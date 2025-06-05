# -*- coding: utf-8 -*-
"""
Streamlit • Inventaire RPA Generali  (v4.0 • 2025-06-05)
Auteur : Samir El Hassani – Avanade
"""

from __future__ import annotations
import csv, io, itertools, pathlib, re, unicodedata, warnings

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─────────────────────────── CONFIG ────────────────────────────
st.set_page_config("Inventaire RPA Generali", page_icon="🤖", layout="wide")
pd.options.display.float_format = "{:,.2f}".format
alt.data_transformers.disable_max_rows()

# ─────────────────────────── UTILS ─────────────────────────────
def slugify(t: str) -> str:
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode()
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", t.lower())).strip("_")

def read_any(buf) -> pd.DataFrame:
    ext = pathlib.Path(buf.name).suffix.lower()
    if ext in (".xls", ".xlsx", ".xlsm"):
        return pd.read_excel(buf, 0, dtype=str, keep_default_na=False)
    buf.seek(0)
    return pd.read_csv(buf, sep=None, engine="python", dtype=str, keep_default_na=False)

def promote_header(df: pd.DataFrame) -> pd.DataFrame:
    # recherche jusqu’à 10 lignes si "Id" et "Nom" sont présents
    for i in range(min(10, len(df))):
        row = df.iloc[i].str.strip().str.lower()
        if {"id", "nom"}.issubset(set(row)):
            df = df.iloc[i + 1 :].reset_index(drop=True)
            df.columns = row
            break
    return df

def to_num(s: pd.Series, pct=False):
    s = (
        s.astype(str)
        .str.replace("%", "")
        .str.replace(",", "")
        .str.replace(" ", "")  # espace insécable
        .str.replace(" ", "")
        .str.strip()
    )
    n = pd.to_numeric(s, errors="coerce")
    return n / 100 if pct else n

def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = promote_header(raw)
    df.dropna(axis=1, how="all", inplace=True)
    df.columns = [slugify(c) or f"col_{i}" for i, c in enumerate(df.columns)]

    for base, pct in {
        "volumetrie_an": False,
        "gains_etp": False,
        "reussite": True,
    }.items():
        col = next((c for c in df.columns if c.startswith(base)), None)
        if col:
            df.rename(columns={col: base}, inplace=True)
            df[base] = to_num(df[base], pct)

    if "synch_asynch" in df:
        df["synch_mode"] = df["synch_asynch"].str.lower().str.strip()

    if "complexite_s_m_l" in df:
        df["complexite_cat"] = (
            df["complexite_s_m_l"]
            .str.upper()
            .str[0]
            .map({"S": "Small", "M": "Medium", "L": "Large"})
        )
    return df

@st.cache_data(show_spinner=False)
def profile(df: pd.DataFrame) -> pd.DataFrame:
    desc = df.describe(include="all", datetime_is_numeric=True)
    tot = len(df)
    p = (
        desc.T.assign(dtype=df.dtypes.astype(str))
        .reset_index()
        .rename(columns={"index": "col"})
    )
    p["count"] = pd.to_numeric(p["count"], errors="coerce").fillna(0)
    p["missing"] = tot - p["count"]
    p["missing_pct"] = (p["missing"] / tot * 100).round(1)
    return p

@st.cache_data(show_spinner=False)
def corr_with_p(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(np.number)
    rec = []
    for a, b in itertools.combinations(num.columns, 2):
        valid = num[[a, b]].dropna()
        if len(valid) > 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
                r, p = stats.pearsonr(valid[a], valid[b])
            rec.append({"var_x": a, "var_y": b, "corr": r, "pval": p})
    return pd.DataFrame(rec)

# ─────────────────────────── IMPORT ───────────────────────────
st.sidebar.header("📂 Import")
file = st.sidebar.file_uploader("Feuille 1 Excel / CSV", ["csv", "xls", "xlsx", "xlsm"])
if not file:
    st.sidebar.info("➡️ Déposez un fichier pour commencer.")
    st.stop()

df = clean(read_any(file))
if st.sidebar.checkbox("Afficher colonnes"): st.sidebar.write(df.columns.tolist())

# ─────────────────────────── FILTRES ──────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filtres")
    c_map = {
        "zone_fonctionnelle": "Zone",
        "departement": "Département",
        "complexite_cat": "Complexité",
        "synch_mode": "Synch/Asynch",
    }
    for c, label in c_map.items():
        if c in df and df[c].notna().any():
            vals = sorted(df[c].dropna().unique())
            df = df[df[c].isin(st.multiselect(label, vals, vals))]

    if "volumetrie_an" in df:
        v = df["volumetrie_an"].dropna()
        rng = st.slider(
            "Volumétrie/an", float(v.min()), float(v.max()), (float(v.min()), float(v.max()))
        )
        df = df[df["volumetrie_an"].between(*rng)]

    if "reussite" in df:
        pct = st.slider("% réussite", 0.0, 1.0, (0.0, 1.0))
        df = df[df["reussite"].between(*pct)]

# ─────────────────────────── KPIs ─────────────────────────────
st.success(f"{len(df):,} lignes – {df.shape[1]} colonnes sélectionnées")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Robots", f"{len(df):,}")
if "volumetrie_an" in df:
    k2.metric("Volumétrie", f"{df['volumetrie_an'].sum():,.0f}")
if "reussite" in df:
    k3.metric("Médiane %", f"{df['reussite'].median()*100:,.1f}%")
    k4.metric("Robots <60 %", int((df["reussite"] < 0.6).sum()))
if "gains_etp" in df:
    k5.metric("Gains ETP", f"{df['gains_etp'].sum():,.1f}")

# ─────────────────────────── TABS ─────────────────────────────
tab_names = [
    "Vue générale",
    "Distributions",
    "Relations",
    "Performance",
    "Heat-Zones",
    "Quick-Wins",
    "Clustering",
    "Complexité/Mode",
    "Tech-Stack",
    "Gains ETP",
    "Données",
]
(
    tab_over,
    tab_dist,
    tab_rel,
    tab_perf,
    tab_heat,
    tab_qw,
    tab_clu,
    tab_comp,
    tab_tech,
    tab_gain,
    tab_data,
) = st.tabs(tab_names)

# ══════════ 1. Vue générale ═══════════════════════════════════
with tab_over:
    st.subheader("📈 Pareto volumétrie (Top 20)")
    if {"volumetrie_an", "nom"}.issubset(df):
        top = (
            df.dropna(subset=["volumetrie_an"])
            .sort_values("volumetrie_an", ascending=False)
            .reset_index(drop=True)
        )
        top["cum"] = top["volumetrie_an"].cumsum() / top["volumetrie_an"].sum()
        st.altair_chart(
            alt.layer(
                alt.Chart(top.head(20))
                .mark_bar()
                .encode(
                    x=alt.X("nom:N", sort="-y", axis=None),
                    y="volumetrie_an:Q",
                    tooltip=["nom", "volumetrie_an"],
                ),
                alt.Chart(top.head(20))
                .mark_line(point=True, color="#E67E22")
                .encode(
                    x="nom:N",
                    y=alt.Y("cum:Q", axis=alt.Axis(format="%")),
                    tooltip=alt.Tooltip("cum:Q", format=".1%"),
                ),
            ).resolve_scale(y="independent"),
            use_container_width=True,
        )

    if "zone_fonctionnelle" in df:
        st.dataframe(
            df.groupby("zone_fonctionnelle")
            .agg(
                robots=("nom", "count"),
                volumetrie=("volumetrie_an", "sum"),
                reussite_moy=("reussite", "mean"),
            )
            .reset_index(),
            use_container_width=True,
            hide_index=True,
        )

# ══════════ 2. Distributions ══════════════════════════════════
with tab_dist:
    st.subheader("📊 Histogrammes & box-plots")
    nums = df.select_dtypes(np.number).columns
    if nums.any():
        var = st.selectbox("Variable numérique", nums, key="dist_var")
        c1, c2 = st.columns(2)
        c1.altair_chart(
            alt.Chart(df.dropna(subset=[var]))
            .mark_bar()
            .encode(x=alt.X(var, bin=True), y="count()"),
            use_container_width=True,
        )
        c2.altair_chart(
            alt.Chart(df.dropna(subset=[var]))
            .mark_boxplot(extent="min-max")
            .encode(y=f"{var}:Q"),
            use_container_width=True,
        )
    else:
        st.info("Aucune variable numérique.")

# ══════════ 3. Relations (corrélation) ════════════════════════
with tab_rel:
    st.subheader("🔗 Matrice de corrélation")
    heat = corr_with_p(df)
    if not heat.empty:
        st.altair_chart(
            alt.Chart(heat)
            .mark_rect()
            .encode(
                x="var_x:O",
                y="var_y:O",
                color=alt.Color("corr:Q", scale=alt.Scale(scheme="viridis")),
                tooltip=[
                    alt.Tooltip("corr:Q", format=".2f"),
                    alt.Tooltip("pval:Q", format=".3e"),
                ],
            ),
            use_container_width=True,
        )
    else:
        st.info("Pas assez de données numériques pour la corrélation.")

# ══════════ 4. Performance (KDE, Q-Q, Bubble) ════════════════
with tab_perf:
    st.subheader("📐 Ajustement Gaussien & analyses")
    numcols = df.select_dtypes(np.number).columns
    if numcols.any():
        v = st.selectbox("Variable", numcols, key="gauss")
        ser = df[v].dropna()
        if len(ser) > 3:
            mu, sig = ser.mean(), ser.std()
            xs = np.linspace(ser.min(), ser.max(), 200)
            fig = go.Figure()
            fig.add_histogram(
                x=ser, nbinsx=40, histnorm="probability density", name="Empirique"
            )
            fig.add_scatter(
                x=xs,
                y=stats.norm.pdf(xs, mu, sig),
                mode="lines",
                name=f"Normale N({mu:.1f}, {sig:.1f}²)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Q-Q plot
            if len(ser) > 20:
                qq_x = np.sort((ser - mu) / sig)
                qq_y = stats.norm.ppf((np.arange(len(qq_x)) + 0.5) / len(qq_x))
                qq = px.scatter(
                    x=qq_x,
                    y=qq_y,
                    labels={"x": f"{v} normalisé (observé)", "y": "Quantiles théoriques"},
                    title="Q-Q Plot",
                    height=400,
                )
                qq.add_shape(
                    type="line",
                    x0=qq_x.min(),
                    y0=qq_x.min(),
                    x1=qq_x.max(),
                    y1=qq_x.max(),
                    line=dict(dash="dash"),
                )
                st.plotly_chart(qq, use_container_width=True)

    # Bubble gains / volumétrie
    st.markdown("#### 💰 Bubble Gains vs Volumétrie")
    if {"volumetrie_an", "gains_etp", "reussite"}.issubset(df):
        bub = px.scatter(
            df.dropna(subset=["volumetrie_an", "gains_etp"]),
            x="volumetrie_an",
            y="gains_etp",
            size="gains_etp",
            color="reussite",
            color_continuous_scale="RdYlGn",
            hover_name="nom",
            height=450,
            log_x=True,
        )
        st.plotly_chart(bub, use_container_width=True)

# ══════════ 5. Heat-Zones (zone × classe % réussite) ══════════
with tab_heat:
    st.subheader("🌡️ Zones × classes de % réussite")
    needed = {"zone_fonctionnelle", "reussite"}
    if needed.issubset(df.columns) and df["zone_fonctionnelle"].notna().any():
        df["classe"] = pd.cut(
            df["reussite"], [0, 0.6, 0.8, 1.01], labels=["<60 %", "60-80 %", "≥80 %"]
        )
        mode = st.radio(
            "Couleur basée sur …",
            ["Nombre de robots", "Volumétrie annuelle totale"],
            horizontal=True,
        )
        if mode == "Nombre de robots":
            pt = (
                df.groupby(["zone_fonctionnelle", "classe"])
                .size()
                .reset_index(name="val")
            )
            title_c = "Robots"
        else:
            pt = (
                df.groupby(["zone_fonctionnelle", "classe"])["volumetrie_an"]
                .sum()
                .reset_index(name="val")
            )
            title_c = "Volumétrie"

        st.altair_chart(
            alt.Chart(pt)
            .mark_rect()
            .encode(
                x="classe:N",
                y=alt.Y("zone_fonctionnelle:N", sort="-x"),
                color=alt.Color("val:Q", scale=alt.Scale(scheme="blues"), title=title_c),
                tooltip=["zone_fonctionnelle", "classe", "val"],
            )
            .properties(
                height=max(300, 22 * pt["zone_fonctionnelle"].nunique())
            ),
            use_container_width=True,
        )
    else:
        st.info("Colonnes requises manquantes.")

# ══════════ 6. Quick-Wins (quadrant) ══════════════════════════
with tab_qw:
    st.subheader("✨ Quadrant Volumétrie × % Réussite")
    if {"volumetrie_an", "reussite", "nom"}.issubset(df):
        base = df.dropna(subset=["volumetrie_an", "reussite"])
        med_vol, med_pct = base["volumetrie_an"].median(), base["reussite"].median()
        quad = alt.Chart(base).mark_circle(size=80, opacity=0.7).encode(
            x=alt.X("volumetrie_an", scale=alt.Scale(type="log")),
            y=alt.Y("reussite", axis=alt.Axis(format="%")),
            color=alt.condition(
                f"datum.reussite < {med_pct} && datum.volumetrie_an > {med_vol}",
                alt.value("#E74C3C"),
                alt.value("#2ECC71"),
            ),
            tooltip=["nom", "volumetrie_an", alt.Tooltip("reussite:Q", format=".1%")],
        )
        hline = (
            alt.Chart(pd.DataFrame({"y": [med_pct]}))
            .mark_rule(strokeDash=[4, 4])
            .encode(y="y:Q")
        )
        vline = (
            alt.Chart(pd.DataFrame({"x": [med_vol]}))
            .mark_rule(strokeDash=[4, 4])
            .encode(x="x:Q")
        )
        st.altair_chart(
            (quad + hline + vline).properties(height=500), use_container_width=True
        )
    else:
        st.info("Colonnes manquantes pour ce quadrant.")

# ══════════ 7. Clustering (K-means) ═══════════════════════════
with tab_clu:
    st.subheader("🎯 Clustering K-means")
    feats = [c for c in ("volumetrie_an", "reussite", "gains_etp") if c in df]
    base = df[feats].dropna()
    if base.shape[0] >= 5 and base.shape[1] >= 2:
        k = st.slider("k", 2, 6, 3)
        lab = KMeans(k, random_state=0, n_init="auto").fit_predict(
            StandardScaler().fit_transform(base)
        )
        df.loc[base.index, "cluster"] = lab
        x_axis = st.selectbox("Axe X", feats, index=0)
        y_axis = st.selectbox("Axe Y", feats, index=1)
        st.plotly_chart(
            px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                color="cluster",
                hover_data=["nom"],
                height=500,
            ),
            use_container_width=True,
        )
    else:
        st.info("≥ 5 lignes et ≥ 2 variables numériques requises.")

# ══════════ 8. Complexité / Mode (pies) ═══════════════════════
with tab_comp:
    st.header("🔍 Complexité & Synch")
    if "complexite_cat" in df and df["complexite_cat"].notna().any():
        st.plotly_chart(
            px.pie(df, names="complexite_cat", hole=0.45, title="Répartition Complexité"),
            use_container_width=True,
        )
    if "synch_mode" in df and df["synch_mode"].notna().any():
        st.plotly_chart(
            px.pie(df, names="synch_mode", hole=0.45, title="Répartition Synch/Asynch"),
            use_container_width=True,
        )

# ══════════ 9. Tech-Stack ═════════════════════════════════════
with tab_tech:
    st.header("🛠️ Tech-Stack")
    # Techno
    if "techno" in df and df["techno"].notna().any():
        st.plotly_chart(
            px.bar(
                df["techno"].value_counts().reset_index().rename(columns={"index": "Technologie", "techno": "Robots"}),
                x="Technologie",
                y="Robots",
                title="Robots par technologie",
            ),
            use_container_width=True,
        )
    # Vendor
    if "vendor" in df and df["vendor"].notna().any():
        st.plotly_chart(
            px.bar(
                df["vendor"].value_counts().reset_index().rename(columns={"index": "Vendor", "vendor": "Robots"}),
                x="Vendor",
                y="Robots",
                title="Robots par Vendor",
            ),
            use_container_width=True,
        )
    # Sunburst Zone → Complexité
    if {"zone_fonctionnelle", "complexite_cat", "volumetrie_an"}.issubset(df):
        st.plotly_chart(
            px.sunburst(
                df,
                path=["zone_fonctionnelle", "complexite_cat"],
                values="volumetrie_an",
                height=500,
                title="Sunburst Volumétrie — Zone → Complexité",
            ),
            use_container_width=True,
        )

# ══════════ 10. Gains ETP (waterfall) ═════════════════════════
with tab_gain:
    st.subheader("🏆 Gains ETP par zone")
    if {"zone_fonctionnelle", "gains_etp"}.issubset(df):
        wf = (
            df.groupby("zone_fonctionnelle")["gains_etp"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig = go.Figure(
            go.Waterfall(
                x=wf["zone_fonctionnelle"],
                y=wf["gains_etp"],
                measure=["relative"] * len(wf),
                decreasing={"marker": {"color": "#E67E22"}},
                totals={"marker": {"color": "#2ECC71"}},
            )
        )
        fig.update_layout(height=450, yaxis_title="Gains ETP")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Gains ETP indisponibles.")

# ══════════ 11. Données (profil + table) ══════════════════════
with tab_data:
    st.subheader("📑 Profil statistique")
    st.dataframe(profile(df), use_container_width=True)

    st.subheader("🗃️ Données filtrées")
    st.dataframe(df, use_container_width=True, height=450)

    @st.cache_data
    def to_csv(d: pd.DataFrame) -> bytes:
        return d.to_csv(index=False).encode()

    st.download_button("💾 Télécharger CSV filtré", to_csv(df), "robots_filtered.csv", "text/csv")

# ─────────────────────────── FOOTER ──────────────────────────
st.caption("© 2025 Generali / Avanade • Samir El Hassani")
