# -*- coding: utf-8 -*-
"""
Streamlit – Inventaire RPA Generali  (v3.6 • 2025-06-05)
Auteur : Samir El Hassani – Avanade
"""

from __future__ import annotations
import csv, io, itertools, re, unicodedata, warnings, pathlib
import numpy as np, pandas as pd, scipy.stats as stats
import altair as alt, plotly.express as px, plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans; from sklearn.preprocessing import StandardScaler

# ── CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config("Inventaire RPA Generali", "🤖", "wide")
pd.options.display.float_format = "{:,.2f}".format
alt.data_transformers.disable_max_rows()

# ── UTILS ─────────────────────────────────────────────────────────────────
def slugify(t:str)->str:
    t=unicodedata.normalize("NFKD",str(t)).encode("ascii","ignore").decode()
    return re.sub(r"_+","_",re.sub(r"[^0-9a-z]+","_",t.lower())).strip("_")

def read_any(buf)->pd.DataFrame:
    ext=pathlib.Path(buf.name).suffix.lower()
    if ext in (".xls",".xlsx",".xlsm"): return pd.read_excel(buf,0,dtype=str,keep_default_na=False)
    buf.seek(0); return pd.read_csv(buf,sep=None,engine="python",dtype=str,keep_default_na=False)

def promote_header(df:pd.DataFrame)->pd.DataFrame:
    for i in range(min(10,len(df))):
        row=df.iloc[i].str.strip().str.lower()
        if {"id","nom"}.issubset(set(row)):
            df=df.iloc[i+1:].reset_index(drop=True); df.columns=row; break
    return df

def to_num(s:pd.Series,pct=False):
    s=(s.astype(str).str.replace("%","").str.replace(",","").str.replace(" ","").str.strip())
    n=pd.to_numeric(s,errors="coerce"); return n/100 if pct else n

def clean(raw:pd.DataFrame)->pd.DataFrame:
    df=promote_header(raw); df.dropna(axis=1,how="all",inplace=True)
    df.columns=[slugify(c) or f"col_{i}" for i,c in enumerate(df.columns)]
    for base,pct in {"volumetrie_an":False,"gains_etp":False,"reussite":True}.items():
        col=next((c for c in df.columns if c.startswith(base)),None)
        if col: df.rename(columns={col:base},inplace=True); df[base]=to_num(df[base],pct)
    if "synch_asynch" in df: df["synch_mode"]=df["synch_asynch"].str.lower().str.strip()
    if "complexite_s_m_l" in df:
        df["complexite_cat"]=df["complexite_s_m_l"].str.upper().str[0].map({"S":"Small","M":"Medium","L":"Large"})
    return df

@st.cache_data(show_spinner=False)
def profile(df): d=df.describe(include="all"); tot=len(df); p=(d.T.assign(dtype=df.dtypes.astype(str)).reset_index()
                 .rename(columns={"index":"col"})); p["missing"]=tot-p["count"].fillna(0); p["missing_pct"]=(p["missing"]/tot*100).round(1); return p

@st.cache_data(show_spinner=False)
def corr_with_p(df):
    num=df.select_dtypes(np.number); out=[]
    for a,b in itertools.combinations(num,2):
        v=num[[a,b]].dropna()
        if len(v)>2: r,p=stats.pearsonr(v[a],v[b]); out.append({"var_x":a,"var_y":b,"corr":r,"pval":p})
    return pd.DataFrame(out)

# ── IMPORT ───────────────────────────────────────────────────────────────
st.sidebar.header("📂 Import")
file=st.sidebar.file_uploader("Feuille 1 Excel / CSV",["csv","xls","xlsx","xlsm"])
if not file: st.sidebar.info("➡️ Déposez un fichier"); st.stop()

df=clean(read_any(file))
if st.sidebar.checkbox("Afficher colonnes"): st.sidebar.write(df.columns.tolist())

# ── FILTRES ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filtres")
    c_map={"zone_fonctionnelle":"Zone","departement":"Département",
           "complexite_cat":"Complexité","synch_mode":"Synch/Asynch"}
    for c,l in c_map.items():
        if c in df and df[c].notna().any():
            vals=sorted(df[c].dropna().unique())
            df=df[df[c].isin(st.multiselect(l,vals,vals))]
    if "volumetrie_an" in df:
        v=df["volumetrie_an"].dropna(); rng=st.slider("Volumétrie/an",float(v.min()),float(v.max()),(float(v.min()),float(v.max())))
        df=df[df["volumetrie_an"].between(*rng)]
    if "reussite" in df:
        pct=st.slider("% réussite",0.0,1.0,(0.0,1.0)); df=df[df["reussite"].between(*pct)]

# ── KPI ──────────────────────────────────────────────────────────────────
st.success(f"{len(df):,} lignes – {df.shape[1]} colonnes")
k1,k2,k3,k4,k5=st.columns(5)
k1.metric("Robots",f"{len(df):,}")
if "volumetrie_an" in df: k2.metric("Volumétrie",f"{df['volumetrie_an'].sum():,.0f}")
if "reussite" in df:
    k3.metric("Médiane %",f"{df['reussite'].median()*100:,.1f}%")
    k4.metric("<60 %",int((df['reussite']<.6).sum()))
if "gains_etp" in df: k5.metric("Gains ETP",f"{df['gains_etp'].sum():,.1f}")

# ── TABS ────────────────────────────────────────────────────────────────
tabs=st.tabs(["Vue générale","Distributions","Relations","Performance","Heat-Zones",
              "Clustering","Complexité/Mode","Tech-Stack","Données"])
(tab_over,tab_dist,tab_rel,tab_perf,tab_heat,tab_clu,tab_comp,tab_tech,tab_data)=tabs

# 1. Vue générale
with tab_over:
    st.subheader("📈 Pareto volumétrie (Top 20)")
    if {"volumetrie_an","nom"}.issubset(df):
        top=df.dropna(subset=["volumetrie_an"]).sort_values("volumetrie_an",ascending=False)
        top["cum"]=top["volumetrie_an"].cumsum()/top["volumetrie_an"].sum()
        st.altair_chart(
            alt.layer(
                alt.Chart(top.head(20)).mark_bar().encode(x=alt.X("nom:N",sort="-y",axis=None),
                                                          y="volumetrie_an:Q"),
                alt.Chart(top.head(20)).mark_line(point=True,color="#E67E22")
                    .encode(x="nom:N",y=alt.Y("cum:Q",axis=alt.Axis(format="%")))
            ).resolve_scale(y="independent"),use_container_width=True)
    if "zone_fonctionnelle" in df:
        st.dataframe(df.groupby("zone_fonctionnelle")
                       .agg(robots=("nom","count"),
                            volumetrie=("volumetrie_an","sum"),
                            reussite_moy=("reussite","mean")).reset_index(),
                     use_container_width=True,hide_index=True)

# 2. Distributions
with tab_dist:
    st.subheader("📊 Histogrammes / Box-plots")
    nums=df.select_dtypes(np.number).columns
    if nums.any():
        var=st.selectbox("Variable",nums)
        c1,c2=st.columns(2)
        c1.altair_chart(alt.Chart(df).transform_filter(f"datum['{var}']!=null")
                         .mark_bar().encode(x=alt.X(f"{var}:Q",bin=True),y='count()'),
                         use_container_width=True)
        c2.altair_chart(alt.Chart(df).transform_filter(f"datum['{var}']!=null")
                         .mark_boxplot(extent="min-max").encode(y=f"{var}:Q"),
                         use_container_width=True)
    else: st.info("Aucune variable numérique.")

# 3. Relations
with tab_rel:
    st.subheader("🔗 Matrice de corrélation")
    heat=corr_with_p(df)
    if not heat.empty:
        st.altair_chart(alt.Chart(heat).mark_rect().encode(
            x="var_x:O",y="var_y:O",color=alt.Color("corr:Q",scale=alt.Scale(scheme="viridis")),
            tooltip=["corr:Q","pval:Q"]),use_container_width=True)

# 4. Performance
with tab_perf:
    st.subheader("📐 Ajustement Gaussien")
    if nums.any():
        v=st.selectbox("Variable",nums,key="gauss")
        ser=df[v].dropna()
        if len(ser)>3:
            mu,sig=ser.mean(),ser.std()
            xs=np.linspace(ser.min(),ser.max(),200)
            fig=go.Figure()
            fig.add_histogram(x=ser,nbinsx=40,histnorm="probability density")
            fig.add_scatter(x=xs,y=stats.norm.pdf(xs,mu,sig),mode="lines")
            st.plotly_chart(fig,use_container_width=True)

# 5. Heat-Zones
with tab_heat:
    st.subheader("🌡️ Zones vs classes de réussite")
    if {"zone_fonctionnelle","reussite"}.issubset(df):
        df["classe"]=pd.cut(df["reussite"],[0,.6,.8,1.01],labels=["<60","60-80","≥80"])
        mode=st.radio("Couleur par",["Nombre","Volumétrie"],horizontal=True)
        if mode=="Nombre":
            pt=df.groupby(["zone_fonctionnelle","classe"]).size().reset_index(name="val")
        else:
            pt=df.groupby(["zone_fonctionnelle","classe"])["volumetrie_an"].sum().reset_index(name="val")
        st.altair_chart(alt.Chart(pt).mark_rect().encode(
            x="classe:N",y="zone_fonctionnelle:N",
            color=alt.Color("val:Q",scale=alt.Scale(scheme="blues"))),
            use_container_width=True)

# 6. Clustering
with tab_clu:
    st.subheader("🎯 K-means")
    feats=[c for c in ("volumetrie_an","reussite","gains_etp") if c in df]
    base=df[feats].dropna()
    if base.shape[0]>=5 and base.shape[1]>=2:
        k=st.slider("k",2,6,3)
        lab=KMeans(k,n_init="auto",random_state=0).fit_predict(StandardScaler().fit_transform(base))
        df.loc[base.index,"cluster"]=lab
        st.plotly_chart(px.scatter(df,x=feats[0],y=feats[1],color="cluster",hover_data=["nom"]),
                        use_container_width=True)
    else: st.info("≥5 lignes et ≥2 variables requises.")

# 7. Complexité / Mode
with tab_comp:
    st.header("🔍 Complexité & Synch")
    if "complexite_cat" in df and df["complexite_cat"].notna().any():
        st.plotly_chart(px.pie(df,names="complexite_cat",hole=.45),use_container_width=True)
    if "synch_mode" in df and df["synch_mode"].notna().any():
        st.plotly_chart(px.pie(df,names="synch_mode",hole=.45),use_container_width=True)

# ═════ 8. Tech-Stack
with tab_tech:
    st.header("🛠️ Tech-Stack")

    # ----- TECHNO -----
    if "techno" in df and df["techno"].notna().any():
        tech = (df["techno"]
                  .value_counts()
                  .reset_index(drop=False)
                  .rename(columns={"index": "Technologie", "techno": "Robots"}))
        if not tech.empty:
            st.plotly_chart(
                px.bar(tech, x="Technologie", y="Robots",
                       title="Nombre de robots par technologie"),
                use_container_width=True)
        else:
            st.info("Aucune donnée technologie après filtres.")
    else:
        st.info("Colonne 'techno' absente ou vide.")

    # ----- VENDOR -----
    if "vendor" in df and df["vendor"].notna().any():
        vend = (df["vendor"]
                  .value_counts()
                  .reset_index(drop=False)
                  .rename(columns={"index": "Vendor", "vendor": "Robots"}))
        if not vend.empty:
            st.plotly_chart(
                px.bar(vend, x="Vendor", y="Robots",
                       title="Nombre de robots par vendor"),
                use_container_width=True)
        else:
            st.info("Aucune donnée vendor après filtres.")
    else:
        st.info("Colonne 'vendor' absente ou vide.")

# 9. Données
with tab_data:
    st.subheader("Profil")
    st.dataframe(profile(df),use_container_width=True)
    st.subheader("Table filtrée")
    st.dataframe(df,use_container_width=True)

# ── FOOTER ───────────────────────────────────────────────────────────────
st.caption(f"© 2025 Generali / Avanade – Samir El Hassani")
