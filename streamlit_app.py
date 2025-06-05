# ───────────── FONCTIONS UTILITAIRES MINIMALES ─────────────
import csv, io, unicodedata, re, warnings
import pandas as pd
import numpy as np
import streamlit as st

def slugify(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", str(txt)).encode("ascii","ignore").decode()
    return re.sub(r"_+","_", re.sub(r"[^0-9a-z]+","_", txt.lower())).strip("_")

def read_csv_robust(buf) -> pd.DataFrame:
    raw = buf.read()
    sample = raw[:2048].decode("utf-8", "replace")
    try:  sep = csv.Sniffer().sniff(sample, [",", ";", "\t"]).delimiter
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
           .str.replace("%", "", regex=False)
           .str.replace(",", "", regex=False)
           .str.replace(" ", "", regex=False)   # espace insécable
           .str.replace(" ", "", regex=False))
    n = pd.to_numeric(s, errors="coerce")
    return n/100 if pct else n

def clean(df0: pd.DataFrame) -> pd.DataFrame:
    df = promote_header(df0); df.dropna(how="all", inplace=True)
    df.columns = [slugify(c) or f"col_{i}" for i, c in enumerate(df.columns)]
    for col, pct in {"volumetrie_an": False, "gains_etp": False, "reussite": True}.items():
        if col in df.columns: df[col] = to_num(df[col], pct)
    return df
# -----------------------------------------------------------

st.sidebar.header("📂 Import")
upl = st.sidebar.file_uploader("CSV UTF-8", ["csv"])
if upl is None:
    st.sidebar.info("➡️ Déposez un fichier pour commencer."); st.stop()

df = clean(read_csv_robust(upl))
df_original = df.copy()          # <-- ta ligne 27 redeviendra valide


# Bibliothèques pour les visualisations avancées
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Titre de l'application
st.title("Dashboard Inventaire RPA - Analyses Avancées")

# Chargement des données (à adapter selon la source réelle des données)
# Ici on suppose un fichier CSV 'inventaire_rpa.csv' disponible, sinon utiliser st.file_uploader
# df = pd.read_csv('inventaire_rpa.csv')
# Pour l'exemple, on part du principe que df est déjà fourni dans l'app v3.0 existante.

# --- Préparation des données ---
# Copie de sauvegarde
df_original = df.copy()
# Optionnel: Renommer certaines colonnes pour simplifier la manipulation (pas obligatoire si on utilise les noms directement)
df = df.rename(columns={
    'Zone fonctionnelle': 'Zone',
    'Département': 'Departement',
    'Techno': 'Technologie',
    'Vendor': 'Vendor',
    'Cloud/On premise': 'Environnement',
    'Volumétrie/an': 'Volumetrie_an',
    '%Réussite': 'Taux_reussite',
    'Gains ETP': 'Gains_ETP'
})

# Encodage de certaines colonnes catégorielles utiles pour analyses (sans modifier le df original)
# (Par exemple pour PCA/KMeans on peut vouloir des colonnes numériques)
encoded_df = df.copy()
# On peut encoder 'Complexité' si elle est catégorielle (faible/moyenne/haute) en valeurs numériques
if 'Complexité' in encoded_df.columns:
    # Mapping manuel possible si Complexité est textual, par ex:
    complexity_map = {'Faible': 1, 'Moyenne': 2, 'Haute': 3}
    encoded_df['Complexite_num'] = encoded_df['Complexité'].map(complexity_map)
# Encodage des autres catégories avec LabelEncoder si besoin
for col in ['Zone', 'Departement', 'Technologie', 'Vendor', 'Environnement', 'Synch/Asynch']:
    if col in encoded_df.columns:
        le = LabelEncoder()
        encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))

# --- Création des onglets pour organiser les visualisations ---
tab1, tab2, tab3, tab4 = st.tabs(["Statistiques Descriptives", "Clustering & Réduction de Dimension", "Modélisation Prédictive", "Visuels de Synthèse"])

# ========== Onglet 1: Statistiques Descriptives ==========
with tab1:
    st.header("Analyse Statistique Descriptive")
    # Sous-section: Statistiques globales
    st.subheader("Statistiques globales")
    st.write("**Nombre total de processus RPA :** ", len(df))
    # Afficher quelques indicateurs clés agrégés (somme des gains, moyenne taux réussite, etc.)
    if 'Gains_ETP' in df.columns:
        total_gains = df['Gains_ETP'].sum()
        st.write(f"**Gains ETP total (somme) :** {total_gains:.2f}")
    if 'Taux_reussite' in df.columns:
        avg_success = df['Taux_reussite'].mean()
        st.write(f"**Taux de réussite moyen :** {avg_success:.1f}%")
    # etc. (on peut ajouter d'autres KPIs globaux si pertinent)

    # Sous-section: Distribution des variables numériques (Histogrammes)
    st.subheader("Distribution des variables numériques")
    numeric_cols = ['Volumetrie_an', 'Taux_reussite', 'Gains_ETP']
    for col in numeric_cols:
        if col in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)  # histogramme avec courbe de densité
            ax.set_title(f"Histogramme de {col}")
            st.pyplot(fig)

    # Sous-section: Boxplots par catégorie (ex: Gains ETP par Complexité)
    if 'Gains_ETP' in df.columns and 'Complexité' in df.columns:
        st.subheader("Répartition des gains ETP par niveau de complexité")
        fig, ax = plt.subplots()
        sns.boxplot(x=df['Complexité'], y=df['Gains_ETP'], ax=ax)
        ax.set_title("Gains ETP par niveau de Complexité")
        st.pyplot(fig)

    # Sous-section: Bar charts des distributions catégorielles
    st.subheader("Nombre de processus par catégorie")
    col1, col2 = st.columns(2)
    # Bar chart par Zone fonctionnelle
    if 'Zone' in df.columns:
        counts = df['Zone'].value_counts()
        fig1 = px.bar(x=counts.index, y=counts.values, labels={'x': 'Zone fonctionnelle', 'y': 'Nombre de processus'}, title="Processus par Zone fonctionnelle")
        col1.plotly_chart(fig1, use_container_width=True)
    # Bar chart par Département
    if 'Departement' in df.columns:
        counts2 = df['Departement'].value_counts()
        fig2 = px.bar(x=counts2.index, y=counts2.values, labels={'x': 'Département', 'y': 'Nombre de processus'}, title="Processus par Département")
        col2.plotly_chart(fig2, use_container_width=True)

    # Sous-section: Matrice de corrélation (heatmap)
    st.subheader("Matrice de corrélation")
    # Calcul de la matrice de corrélation sur les variables numériques principales
    corr_matrix = df[['Volumetrie_an', 'Taux_reussite', 'Gains_ETP']].corr() if all(x in df.columns for x in ['Volumetrie_an','Taux_reussite','Gains_ETP']) else None
    if corr_matrix is not None:
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Heatmap de corrélation")
        st.pyplot(fig)
    else:
        st.write("Pas de corrélation calculable (colonnes manquantes).")

    # Sous-section: Scatter plot Gains vs Volumetrie avec tendance
    if 'Volumetrie_an' in df.columns and 'Gains_ETP' in df.columns:
        st.subheader("Relation Volume vs Gains (avec tendance)")
        fig = px.scatter(df, x='Volumetrie_an', y='Gains_ETP', trendline="ols",
                         labels={'Volumetrie_an': 'Volumétrie/an', 'Gains_ETP': 'Gains ETP'},
                         title="Gains ETP en fonction de la Volumétrie annuelle")
        st.plotly_chart(fig, use_container_width=True)

# ========== Onglet 2: Clustering & Réduction de Dimension ==========
with tab2:
    st.header("Clustering et Réduction de Dimension")
    # Préparation des données numériques pour PCA/Clustering
    features_for_analysis = []
    for col in ['Volumetrie_an','Taux_reussite','Gains_ETP','Complexite_num']:
        if col in encoded_df.columns:
            features_for_analysis.append(col)
    data_for_analysis = encoded_df[features_for_analysis].dropna()
    # Standardisation des données
    data_scaled = StandardScaler().fit_transform(data_for_analysis.values) if len(data_for_analysis)>0 else None

    # Application du PCA à 2 composantes
    if data_scaled is not None and data_scaled.shape[1] >= 2:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(data_scaled)
        df_pca = pd.DataFrame(proj, columns=['PC1','PC2'])
        # Ajout de labels pour la couleur (on choisit une variable catégorielle existante si possible)
        if 'Zone' in df.columns:
            df_pca['Couleur'] = df.loc[data_for_analysis.index, 'Zone']
        elif 'Technologie' in df.columns:
            df_pca['Couleur'] = df.loc[data_for_analysis.index, 'Technologie']
        else:
            df_pca['Couleur'] = None
        st.subheader("Projection PCA (2 composantes principales)")
        fig = px.scatter(df_pca, x='PC1', y='PC2', color='Couleur', title="Projection PCA des processus RPA")
        st.plotly_chart(fig, use_container_width=True)
        # Afficher la variance expliquée
        exp_var = pca.explained_variance_ratio_
        st.caption(f"Variance expliquée par PC1 et PC2 : {exp_var[0]*100:.1f}% + {exp_var[1]*100:.1f}% = {sum(exp_var)*100:.1f}%")

    # Application de t-SNE pour visualisation non-linéaire (2D)
    if data_scaled is not None:
        tsne = TSNE(n_components=2, random_state=42, perplexity=15, n_iter=500)
        proj_tsne = tsne.fit_transform(data_scaled)
        df_tsne = pd.DataFrame(proj_tsne, columns=['TSNE1','TSNE2'])
        if 'Zone' in df.columns:
            df_tsne['Couleur'] = df.loc[data_for_analysis.index, 'Zone']
        elif 'Technologie' in df.columns:
            df_tsne['Couleur'] = df.loc[data_for_analysis.index, 'Technologie']
        else:
            df_tsne['Couleur'] = None
        st.subheader("Projection t-SNE (2 composantes)")
        fig = px.scatter(df_tsne, x='TSNE1', y='TSNE2', color='Couleur', title="t-SNE des processus RPA")
        st.plotly_chart(fig, use_container_width=True)

    # Clustering K-Means sur les mêmes features
    if data_scaled is not None:
        # Choix du nombre de clusters k (exemple k=3)
        k = 3
        kmeans = KMeans(n_clusters=k, random_state=1, n_init='auto')
        labels = kmeans.fit_predict(data_scaled)
        df_clust = df.loc[data_for_analysis.index].copy()
        df_clust['Cluster'] = labels
        # Info sur les clusters
        st.subheader("Clustering K-Means")
        st.write(f"**Nombre de clusters choisi :** {k}")
        # Afficher taille et caractéristiques moyennes de chaque cluster
        cluster_summary = df_clust.groupby('Cluster')[['Volumetrie_an','Taux_reussite','Gains_ETP']].mean()
        cluster_counts = df_clust['Cluster'].value_counts()
        st.write("**Effectifs par cluster :**", cluster_counts.to_dict())
        st.write("**Moyennes des variables par cluster :**")
        st.dataframe(cluster_summary)
        # Visualisation des clusters sur PCA (si calculé)
        if data_scaled.shape[1] >= 2:
            df_pca['Cluster'] = labels
            fig = px.scatter(df_pca, x='PC1', y='PC2', color=df_pca['Cluster'].astype(str), title="Clusters (K-Means) sur projection PCA")
            st.plotly_chart(fig, use_container_width=True)

    # Détection d'outliers simples via Z-score
    st.subheader("Processus RPA atypiques (Outliers)")
    if data_scaled is not None:
        # Calcul du Z-score moyen (somme des valeurs absolues des z-scores sur chaque variable normalisée)
        z_scores = np.abs((data_for_analysis - data_for_analysis.mean())/data_for_analysis.std(ddof=0))
        z_sum = z_scores.sum(axis=1)
        # On considère comme outliers les 5 plus grands scores
        outlier_idx = z_sum.nlargest(5).index
        outliers = df.loc[outlier_idx, ['Id','Nom','Zone','Departement','Volumetrie_an','Taux_reussite','Gains_ETP']]
        st.write("Top 5 des processus les plus atypiques (selon un score d'anomalie simple) :")
        st.dataframe(outliers)
    else:
        st.write("Données insuffisantes pour calculer des outliers.")

# ========== Onglet 3: Modélisation Prédictive ==========
with tab3:
    st.header("Modélisation Prédictive & Importance des Variables")
    # Régression linéaire simple (Volumetrie -> Gains)
    st.subheader("Régression linéaire : Gains ETP ~ Volumétrie/an")
    if 'Volumetrie_an' in df.columns and 'Gains_ETP' in df.columns:
        # On utilise Plotly Express avec trendline pour illustrer la régression
        fig = px.scatter(df, x='Volumetrie_an', y='Gains_ETP', trendline="ols",
                         labels={'Volumetrie_an':'Volumétrie/an', 'Gains_ETP':'Gains ETP'},
                         title="Relation entre volumétrie et gains ETP (régression OLS)")
        st.plotly_chart(fig, use_container_width=True)
        # On peut extraire les paramètres de la régression via les résultats statsmodels si besoin:
        # results = px.get_trendline_results(fig)
        # st.write(results.px_fit_results.iloc[0].summary())
    else:
        st.write("Variables Volumétrie/an ou Gains ETP manquantes, régression non effectuée.")

    # Importance des variables via Random Forest
    st.subheader("Importance des variables (Random Forest)")
    # Exemple: prédire un succès élevé/faible en classification (on définit succès élevé si % réussite > 90% par ex)
    if 'Taux_reussite' in df.columns:
        # Créer une classe binaire succès (1 si taux >= 90, 0 sinon) pour l'exemple
        df_model = encoded_df.dropna(subset=['Taux_reussite']).copy()
        df_model['HighSuccess'] = (df_model['Taux_reussite'] >= 90).astype(int)
        # Features candidates (on évite d'inclure Taux_reussite évidemment)
        feature_candidates = ['Volumetrie_an','Complexite_num','Gains_ETP']
        feature_cols = [col for col in feature_candidates if col in df_model.columns]
        X = df_model[feature_cols]
        y = df_model['HighSuccess']
        if len(X) > 0:
            # Entraînement d'un RandomForestClassifier simple
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
            fig = px.bar(x=feat_imp.values, y=feat_imp.index, orientation='h',
                         labels={'x':'Importance', 'y':'Variable'}, title="Importance des variables - Random Forest")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Les variables les plus importantes pour prédire un haut taux de réussite.")
        else:
            st.write("Pas de données suffisantes pour calculer les importances.")
    else:
        st.write("Taux de réussite non disponible, importance des variables non calculée.")

# ========== Onglet 4: Visuels de Synthèse (PPT) ==========
with tab4:
    st.header("Graphiques de Synthèse (Style PPT)")
    # Radar Chart: Profil par Zone fonctionnelle (moyennes normalisées des indicateurs)
    if 'Zone' in df.columns:
        st.subheader("Profil par Zone (Radar Chart)")
        radar_metrics = ['Volumetrie_an','Taux_reussite','Gains_ETP']
        # Calcul des moyennes par zone
        radar_data = df.groupby('Zone')[radar_metrics].mean().reset_index()
        # Normaliser chaque métrique 0-1 pour comparer les formes (optionnel)
        radar_norm = radar_data.copy()
        for m in radar_metrics:
            max_val = radar_data[m].max()
            min_val = radar_data[m].min()
            if max_val > min_val:
                radar_norm[m] = (radar_data[m] - min_val) / (max_val - min_val)
            else:
                radar_norm[m] = 0.0
        # Préparer les données pour plotly (trace par zone)
        fig = go.Figure()
        categories = radar_metrics
        for _, row in radar_norm.iterrows():
            zone = row['Zone']
            values = [row[m] for m in radar_metrics]
            # boucler pour fermer le polygone
            values += [values[0]]
            fig.add_trace(go.Scatterpolar(r=values, theta=categories+ [categories[0]], fill='toself', name=str(zone)))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), title="Profil comparatif des zones (moyennes normalisées)")
        st.plotly_chart(fig, use_container_width=True)

    # Waterfall Chart: Contribution des départements aux gains totaux
    if 'Gains_ETP' in df.columns and 'Departement' in df.columns:
        st.subheader("Contribution des Départements aux Gains Totaux (Waterfall)")
        dept_sum = df.groupby('Departement')['Gains_ETP'].sum().reset_index()
        dept_sum = dept_sum.sort_values('Gains_ETP', ascending=False)
        # Préparer le waterfall via plotly
        fig = go.Figure(go.Waterfall(
            name="Gains", orientation="v",
            x=dept_sum['Departement'],
            y=dept_sum['Gains_ETP'],
            text=[f"{val:.1f}" for val in dept_sum['Gains_ETP']],
            textposition="outside"
        ))
        fig.update_layout(title="Gains ETP par département - Waterfall", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Word Cloud des descriptions
    st.subheader("Nuage de Mots des Descriptions")
    if 'Description' in df.columns:
        text = " ".join(df['Description'].astype(str).tolist())
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color="white", max_words=100, collocations=False).generate(text)
            fig, ax = plt.subplots(figsize=(6,3))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.write("Pas de descriptions disponibles pour générer un word cloud.")
    else:
        st.write("Champ 'Description' non disponible.")

    # Sunburst Chart: Répartition hiérarchique Zone -> Département
    if 'Zone' in df.columns and 'Departement' in df.columns:
        st.subheader("Répartition hiérarchique Zone / Département (Sunburst)")
        fig = px.sunburst(df, path=['Zone','Departement'], values=None, title="Nombre de processus par Zone et Département")
        st.plotly_chart(fig, use_container_width=True)

    # Sankey Diagram: Flux Zone -> Technologie (ou Vendor)
    if 'Zone' in df.columns and 'Vendor' in df.columns:
        st.subheader("Utilisation des Technologies par Zone (Sankey)")
        # Construire les noeuds et liens
        zones = list(df['Zone'].unique())
        vendors = list(df['Vendor'].unique())
        all_nodes = zones + vendors
        # dictionnaires pour index
        idx = {node: i for i, node in enumerate(all_nodes)}
        # Calculer les liens (nombre de processus par zone-vendor)
        links = df.groupby(['Zone','Vendor'])['Id'].count().reset_index()
        # Créer les listes source, target, value pour le sankey
        sources = links['Zone'].map(idx).tolist()
        targets = links['Vendor'].map(idx).tolist()
        values = links['Id'].tolist()
        # Diagramme Sankey
        fig = go.Figure(data=[go.Sankey(
            node=dict(label=all_nodes, pad=15, thickness=20),
            link=dict(source=sources, target=targets, value=values)
        )])
        fig.update_layout(title_text="Répartition des technologies RPA par zone", font_size=10)
        st.plotly_chart(fig, use_container_width=True)
