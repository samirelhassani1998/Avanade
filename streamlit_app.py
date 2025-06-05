# -*- coding: utf-8 -*-
"""
Streamlit – Inventaire RPA Generali  (v4.1 • 2025-06-05)
Auteur : Samir El Hassani – Avanade
"""

from __future__ import annotations
import csv, io, itertools, re, unicodedata, warnings, pathlib
from typing import List

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
st.set_page_config("Inventaire RPA Generali", "🤖", "wide")
pd.options.display.float_format = "{:,.2f}".format
alt.data_transformers.disable_max_rows()

# ─────────────────────────── UTILS ─────────────────────────────
def slugify(t: str) -> str:
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode()
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", t.lower())).strip("_")

def read_any(buf) -> pd.DataFrame:
    ext = pathlib.Path(buf.name).suffix.lower()
    if ext in {".xls", ".xlsx", ".xlsm"}:
        return pd.read_excel(buf, 0, dtype=str, keep_default_na=False)
    buf.seek(0)
    return pd.read_csv(buf, sep=None, engine="python", dtype=str, keep_default_na=False)

def promote_header(df: pd.DataFrame) -> pd.DataFrame:
    for i in range(min(10, len(df))):
        row = df.iloc[i].str.strip().str.lower()
        if {"id", "nom"}.issubset(set(row)):
            df = df.iloc[i + 1 :].reset_index(drop=True)
            df.columns = row
            break
    return df

def to_num(s: pd.Series, pct=False):
    clean = (
        s.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip()
    )
    n = pd.to_numeric(clean, errors="coerce")
    return n / 100 if pct else n

def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = promote_header(raw)
    df.dropna(axis=1, how="all", inplace=True)
    df.columns = [slugify(c) or f"col_{i}" for i, c in enumerate(df.columns)]
    mapping = {"volumetrie_an": False, "gains_etp": False, "reussite": True}
    for base, pct in mapping.items():
        col = next((c for c in df.columns if c.startswith(base)), None)
        if col:
            df.rename(columns={col: base}, inplace=True)
            df[base] = to_num(df[base], pct)
    if "synch_asynch" in df:
        df["synch_mode"] = df["synch_asynch"].str.lower().str.strip()
    if "complexite_s_m_l" in df:
        df["complexite_cat"] = (
            df["complexite_s_m_l"].str.upper().str[0].map({"S": "Small", "M": "Medium", "L": "Large"})
        )
    return df

@st.cache_data(show_spinner=False)
def profile(df: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = df.describe(include="all")
    p = (
        d.T.assign(dtype=df.dtypes.astype(str))
        .reset_index()
        .rename(columns={"index": "col"})
    )
    tot = len(df)
    p["missing"] = tot - p["count"].fillna(0)
    p["missing_pct"] = (p["missing"] / tot * 100).round(1)
    return p

@st.cache_data(show_spinner=False)
def corr_with_p(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(np.number)
    out = []
    for a, b in itertools.combinations(num, 2):
        v = num[[a, b]].dropna()
        if len(v) > 2:
            r, p = stats.pearsonr(v[a], v[b])
            out.append({"var_x": a, "var_y": b, "corr": r, "pval": p})
    return pd.DataFrame(out)

# ─────────────────────────── IMPORT ────────────────────────────
st.sidebar.header("📂 Import")
file = st.sidebar.file_uploader("Feuille 1 Excel / CSV", ["csv", "xls", "xlsx", "xlsm"])
if not file:
    st.sidebar.info("➡️ Déposez un fichier")
    st.stop()

df = clean(read_any(file))
if st.sidebar.checkbox("Afficher colonnes"):
    st.sidebar.write(df.columns.tolist())

# ─────────────────────────── FILTRES ───────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filtres")
    c_map = {
        "zone_fonctionnelle": "Zone",
        "departement": "Département",
        "complexite_cat": "Complexité",
        "synch_mode": "Synch/Asynch",
    }
    for c, lab in c_map.items():
        if c in df and df[c].notna().any():
            vals = sorted(df[c].dropna().unique())
            df = df[df[c].isin(st.multiselect(lab, vals, vals))]
    if "volumetrie_an" in df:
        v = df["volumetrie_an"].dropna()
        rng = st.slider("Volumétrie/an", float(v.min()), float(v.max()), (float(v.min()), float(v.max())))
        df = df[df["volumetrie_an"].between(*rng)]
    if "reussite" in df:
        pct = st.slider("% réussite", 0.0, 1.0, (0.0, 1.0))
        df = df[df["reussite"].between(*pct)]

# ─────────────────────────── KPI ───────────────────────────────
st.success(f"{len(df):,} lignes – {df.shape[1]} colonnes")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Robots", f"{len(df):,}")
if "volumetrie_an" in df:
    k2.metric("Volumétrie", f"{df['volumetrie_an'].sum():,.0f}")
if "reussite" in df:
    k3.metric("Médiane %", f"{df['reussite'].median()*100:,.1f}%")
    k4.metric("<60 %", int((df["reussite"] < 0.6).sum()))
if "gains_etp" in df:
    k5.metric("Gains ETP", f"{df['gains_etp'].sum():,.1f}")

# ─────────────────────────── TABS ──────────────────────────────
(
    tab_over,
    tab_dist,
    tab_rel,
    tab_perf,
    tab_heat,
    tab_qw,
    tab_gain,
    tab_clu,
    tab_comp,
    tab_tech,
    tab_data,
) = st.tabs(
    [
        "Vue générale",
        "Distributions",
        "Corrélations",
        "Performance",
        "Heat-Zones",
        "Quick-Wins",
        "Gains ETP",
        "Clustering",
        "Complexité/Mode",
        "Tech-Stack",
        "Données",
    ]
)

# ═══ 1. Vue générale ═══
with tab_over:
    st.subheader("📈 Pareto volumétrie (Top 20)")
    if {"volumetrie_an", "nom"}.issubset(df):
        top = df.dropna(subset=["volumetrie_an"]).sort_values("volumetrie_an", ascending=False)
        top["cum"] = top["volumetrie_an"].cumsum() / top["volumetrie_an"].sum()
        st.altair_chart(
            alt.layer(
                alt.Chart(top.head(20))
                .mark_bar()
                .encode(x=alt.X("nom:N", sort="-y", axis=None), y="volumetrie_an:Q"),
                alt.Chart(top.head(20))
                .mark_line(point=True, color="#E67E22")
                .encode(x="nom:N", y=alt.Y("cum:Q", axis=alt.Axis(format="%"))),
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
st.markdown("#### 🔵 Bubble Zones")
if {"zone_fonctionnelle","volumetrie_an","reussite"}.issubset(df):
    agg = (df.groupby("zone_fonctionnelle")
             .agg(robots=("nom","count"),
                  volumetrie=("volumetrie_an","sum"),
                  reussite_moy=("reussite","mean"))
             .reset_index())
    fig = px.scatter(agg,
        x="volumetrie", y="reussite_moy", size="robots", text="zone_fonctionnelle",
        labels={"volumetrie":"Volumétrie","reussite_moy":"% Réussite"},
        height=500)
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)
st.markdown("#### 🌳 Treemap volumétrie par zone")
st.plotly_chart(
    px.treemap(agg, path=["zone_fonctionnelle"], values="volumetrie",
               color="reussite_moy", color_continuous_scale="RdYlGn",
               hover_data={"robots":True, "reussite_moy":":.1%"}),
    use_container_width=True)

# ═══ 2. Distributions ═══
with tab_dist:
    st.subheader("📊 Histogramme & box-plot")
    nums = df.select_dtypes(np.number).columns
    if nums.any():
        var = st.selectbox("Variable", nums)
        c1, c2 = st.columns(2)
        c1.altair_chart(
            alt.Chart(df).transform_filter(f"datum.{var}!=null").mark_bar().encode(
                x=alt.X(f"{var}:Q", bin=True), y="count()"
            ),
            use_container_width=True,
        )
        c2.altair_chart(
            alt.Chart(df).transform_filter(f"datum.{var}!=null").mark_boxplot(extent="min-max").encode(
                y=f"{var}:Q"
            ),
            use_container_width=True,
        )
    else:
        st.info("Aucune variable numérique.")

# ═══ 3. Corrélations ═══
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
                tooltip=["corr:Q", "pval:Q"],
            ),
            use_container_width=True,
        )

# ═══ 4. Performance (KDE + Q-Q + Bubble) ═══
with tab_perf:
    st.subheader("📐 Ajustement Gaussien")
    nums = df.select_dtypes(np.number).columns
    if nums.any():
        v = st.selectbox("Variable", nums, key="gauss")
        ser = df[v].dropna()
        if len(ser) > 3:
            mu, sig = ser.mean(), ser.std()
            xs = np.linspace(ser.min(), ser.max(), 200)
            fig = go.Figure()
            fig.add_histogram(x=ser, nbinsx=40, histnorm="probability density", name="Empirique")
            fig.add_scatter(x=xs, y=stats.norm.pdf(xs, mu, sig), mode="lines", name=f"N({mu:.1f},{sig:.1f})")
            st.plotly_chart(fig, use_container_width=True)

            # Q-Q plot
            st.markdown("#### 🆚 Q-Q plot")
            qq_x = np.sort(ser)
            qq_y = stats.norm.ppf((np.arange(len(qq_x)) + 0.5) / len(qq_x))
            st.plotly_chart(
                px.scatter(x=qq_x, y=qq_y, labels={"x": f"{v} observé", "y": "Quantiles théoriques"}),
                use_container_width=True,
            )

    # Bubble gains
    st.markdown("#### 💰 Bubble Gains / Volumétrie")
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

# ═══ 5. Heat-Zones (Zone × classe de % réussite) ═══
with tab_heat:
    st.subheader("🌡️ Zones vs classes de % réussite")
    needed = {"zone_fonctionnelle", "reussite", "volumetrie_an"}
    if needed.issubset(df.columns) and df["zone_fonctionnelle"].notna().any():
        df["classe"] = pd.cut(df["reussite"], [0, 0.6, 0.8, 1.01], labels=["<60", "60-80", "≥ 80"])
        mode = st.radio("Couleur par …", ["Nombre", "Volumétrie"], horizontal=True)
        if mode == "Nombre":
            pt = df.groupby(["zone_fonctionnelle", "classe"]).size().reset_index(name="val")
            title = "Robots"
        else:
            pt = (
                df.groupby(["zone_fonctionnelle", "classe"])["volumetrie_an"]
                .sum()
                .reset_index(name="val")
            )
            title = "Volumétrie"
        st.altair_chart(
            alt.Chart(pt)
            .mark_rect()
            .encode(
                x="classe:N",
                y=alt.Y("zone_fonctionnelle:N", sort="-x"),
                color=alt.Color("val:Q", scale=alt.Scale(scheme="blues"), title=title),
                tooltip=["zone_fonctionnelle", "classe", alt.Tooltip("val:Q", format=",.0f")],
            )
            .properties(height=max(300, 20 * pt["zone_fonctionnelle"].nunique())),
            use_container_width=True,
        )

# ═══ 6. Quick-Wins (Quadrant) ═══
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
        hline = alt.Chart(pd.DataFrame({"y": [med_pct]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
        vline = alt.Chart(pd.DataFrame({"x": [med_vol]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart((quad + hline + vline).properties(height=500), use_container_width=True)

# ═══ 7. Gains ETP – Waterfall ═══
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

# ═══ 8. Clustering ═══
with tab_clu:
    st.subheader("🎯 K-means")
    feats = [c for c in ("volumetrie_an", "reussite", "gains_etp") if c in df]
    base = df[feats].dropna()
    if base.shape[0] >= 5 and base.shape[1] >= 2:
        k = st.slider("k", 2, 6, 3)
        lab = KMeans(k, n_init="auto", random_state=0).fit_predict(StandardScaler().fit_transform(base))
        df.loc[base.index, "cluster"] = lab
        x_axis, y_axis = st.selectbox("Axe X", feats, 0), st.selectbox("Axe Y", feats, 1)
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
        st.info("≥ 5 lignes et ≥ 2 variables requises.")

# ═══ 9. Complexité / Mode ═══
with tab_comp:
    st.header("🔍 Complexité & Synch")
    if "complexite_cat" in df and df["complexite_cat"].notna().any():
        st.plotly_chart(px.pie(df, names="complexite_cat", hole=0.45), use_container_width=True)
    if "synch_mode" in df and df["synch_mode"].notna().any():
        st.plotly_chart(px.pie(df, names="synch_mode", hole=0.45), use_container_width=True)

# ═══ 10. Tech-Stack ═══
with tab_tech:
    st.header("🛠️ Tech-Stack")

# ── helper sûr pour bar-charts Tech ────────────────────────────
def tech_summary(col: str, label: str):
    """Affiche un graphique ou un compteur, selon la richesse de la colonne."""
    if col not in df or df[col].dropna().empty:
        st.info(f"Aucune donnée « {label} » après filtres."); return

    vc = df[col].dropna().value_counts()
    if len(vc) <= 1:
        st.metric(f"{label} unique", vc.index[0] if not vc.empty else "—")
        return

    tmp = vc.reset_index().rename(columns={"index": label, col: "Robots"})
    fig = px.bar(tmp, x=label, y="Robots",
                 title=f"Robots par {label.lower()}", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ----- Dans l’onglet 🛠️ Tech-Stack -------
with tab_tech:
    st.header("🛠️ Tech-Stack")
    tech_summary("techno",  "Technologie")
    tech_summary("vendor",  "Vendor")


# ----- SUNBURST Zone → Complexité ------------------------------------------
st.markdown("#### 🌞 Sunburst Zone → Complexité")

needed_cols = {"zone_fonctionnelle", "complexite_cat", "volumetrie_an"}
if needed_cols.issubset(df.columns):
    sb_df = (
        df.dropna(subset=["zone_fonctionnelle", "complexite_cat", "volumetrie_an"])
          .groupby(["zone_fonctionnelle", "complexite_cat"], as_index=False)["volumetrie_an"]
          .sum()
    )

    n_zone       = sb_df["zone_fonctionnelle"].nunique()
    n_complexite = sb_df["complexite_cat"].nunique()

    if n_zone >= 2 and n_complexite >= 2 and not sb_df.empty:
        fig = px.sunburst(
            sb_df,
            path=["zone_fonctionnelle", "complexite_cat"],
            values="volumetrie_an",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Données insuffisantes pour construire un sunburst (au moins 2 zones et 2 complexités nécessaires).")
else:
    st.info("Colonnes nécessaires au sunburst manquantes.")


# ═══ 11. Données ═══
with tab_data:
    st.subheader("Profil")
    st.dataframe(profile(df), use_container_width=True)
    st.subheader("Table filtrée")
    st.dataframe(df, use_container_width=True)

# ─────────────────────────── FOOTER ───────────────────────────
st.caption("© 2025 Generali / Avanade – Samir El Hassani")
