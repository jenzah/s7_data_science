"""
Dashboard Marché Immobilier IDF (2024)

"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 0) CONFIG & DATA LOADING

st.set_page_config(page_title="Immobilier IDF", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
 
    df = pd.read_csv(path)
    # Sécuriser la date + colonnes utiles
    df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")
    df = df.dropna(subset=["date_mutation"])
    df["annee"] = df["date_mutation"].dt.year
    df["mois"] = df["date_mutation"].dt.to_period("M").astype(str)  # ex. "2024-04"
    # Harmoniser types
    if df["code_departement"].dtype != "object":
        df["code_departement"] = df["code_departement"].astype(str)
    # Sécurité : enlever lignes sans coord (pour la carte)
    # (on garde quand même pour les autres graphiques)
    return df

df = load_data("dvf_idf_clean.csv")

st.title("🏡 Marché Immobilier – Île-de-France (2024)")

# 1) SIDEBAR — FILTRES

with st.sidebar:
    st.header("Filtres")

    # Départements
    deps = st.multiselect(
        "Départements",
        options=sorted(df["code_departement"].dropna().unique()),
        default=sorted(df["code_departement"].dropna().unique())
    )

    # Communes dépendantes du choix de département
    communes_options = (df[df["code_departement"].isin(deps)]["nom_commune"]
                        .dropna().sort_values().unique())
    communes = st.multiselect(
        "Communes",
        options=communes_options,
        default=[]
    )

    # Types de bien
    type_opts = df["type_local"].dropna().unique()
    types = st.multiselect(
        "Type de bien",
        options=sorted(type_opts),
        default=sorted(type_opts)
    )

    # Mois
    mois_opts = (df["mois"].dropna().sort_values().unique())
    mois_select = st.multiselect(
        "Mois",
        options=mois_opts,
        default=list(mois_opts)  # tous par défaut
    )

    # Pièces
    min_pcs, max_pcs = int(df["nombre_pieces_principales"].min()), int(df["nombre_pieces_principales"].max())
    pieces = st.slider("Nombre de pièces", min_pcs, max_pcs, (min(1, min_pcs), min(10, max_pcs)))

    # Surface (bâti)
    s_min, s_max = int(df["surface_reelle_bati"].min()), int(df["surface_reelle_bati"].max())
    surface = st.slider("Surface habitable (m²)", s_min, s_max, (max(5, s_min), min(500, s_max)))

    # Prix/m² (quantiles pour éviter les extrêmes)
    q1, q99 = df["prix_m2"].quantile([0.01, 0.99]).round(0)
    p_min, p_max = st.slider(
        "Prix au m² (€) (plage basée sur 1er–99e centile)",
        int(q1), int(q99), (int(q1), int(q99))
    )

# 2) APPLICATION DES FILTRES


mask = (
    df["code_departement"].isin(deps) &
    ((df["nom_commune"].isin(communes)) if len(communes) > 0 else True) &
    (df["type_local"].isin(types)) &
    (df["mois"].isin(mois_select)) &
    (df["nombre_pieces_principales"].between(pieces[0], pieces[1], inclusive="both")) &
    (df["surface_reelle_bati"].between(surface[0], surface[1], inclusive="both")) &
    (df["prix_m2"].between(p_min, p_max, inclusive="both"))
)
df_f = df.loc[mask].copy()

# Petite aide visuelle si le filtre vide tout
if df_f.empty:
    st.warning("Aucune donnée avec ces filtres. Assouplis un peu (prix, mois, etc.).")


# 3) KPIs (avec delta)
# delta c pour calculer ecart avce mois précident

# prix médian courant et au mois précédent (si possible) pour delta
def monthly_median(x: pd.DataFrame) -> pd.Series:
    return (x.groupby("mois")["prix_m2"].median().sort_index())

med_series = monthly_median(df_f)
current_med = med_series.iloc[-1] if len(med_series) else np.nan
prev_med = med_series.iloc[-2] if len(med_series) > 1 else np.nan
delta = None if (np.isnan(current_med) or np.isnan(prev_med)) else (current_med - prev_med)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Prix médian /m²", f"{current_med:,.0f} €".replace(",", " "), 
            delta=None if delta is None else f"{delta:+.0f} € vs mois-1")
col2.metric("Prix moyen /m²", f"{df_f['prix_m2'].mean():,.0f} €".replace(",", " "))
col3.metric("Transactions", f"{len(df_f):,}".replace(",", " "))
col4.metric("Surface moyenne", f"{df_f['surface_reelle_bati'].mean():.0f} m²" if not df_f.empty else "N/A")

st.caption("Delta = évolution du prix médian /m² par rapport au mois précédent dans le filtre actuel.")

# 4) ONGLET — VISUALISATIONS


tabs = st.tabs(["📍 Carte", "💶 Prix", "📊 Volumes", "🏷️ Typologie", "🏙️ Communes"])

# TAB 1 : Carte 
with tabs[0]:
    st.subheader("Carte des transactions filtrées")

    # Choix du mode carte
    mode_map = st.radio(
        "Mode carte",
        options=["Points", "Densité"],
        horizontal=True
    )

    # Centre/zoom auto si dispo
    if df_f[["latitude","longitude"]].dropna().empty:
        st.info("Pas de coordonnées pour afficher la carte.")
    else:
        if mode_map == "Points":
            fig_map = px.scatter_mapbox(
                df_f.dropna(subset=["latitude","longitude"]),
                lat="latitude",
                lon="longitude",
                color="prix_m2",
                size="surface_reelle_bati",
                hover_data={
                    "nom_commune": True,
                    "type_local": True,
                    "valeur_fonciere": ":.0f",
                    "prix_m2": ":.0f",
                    "surface_reelle_bati": True,
                    "latitude": False,
                    "longitude": False
                },
                color_continuous_scale="Viridis",
                zoom=9, height=520
            )
        else:
            # Carte de densité
            fig_map = px.density_mapbox(
                df_f.dropna(subset=["latitude","longitude"]),
                lat="latitude",
                lon="longitude",
                z="prix_m2",
                radius=20,
                color_continuous_scale="Viridis",
                zoom=9, height=520
            )

        fig_map.update_layout(mapbox_style="carto-positron", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_map, use_container_width=True)

# TAB 2 : Prix 
with tabs[1]:
    left, right = st.columns([2, 1])

    with left:
        st.markdown("**Distribution du prix au m² (par type)**")
        fig_hist = px.histogram(
            df_f,
            x="prix_m2",
            color="type_local",
            nbins=50,
            barmode="overlay"
        )
        fig_hist.update_layout(template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("**Tendance mensuelle du prix au m² (médiane & moyenne)**")
        price_month = (df_f.groupby("mois")
                           .agg(prix_m2_med=("prix_m2","median"),
                                prix_m2_mean=("prix_m2","mean"))
                           .reset_index())
        fig_line = px.line(price_month, x="mois", y=["prix_m2_med","prix_m2_mean"],
                           markers=True)
        fig_line.update_layout(template="plotly_white", legend_title="")
        st.plotly_chart(fig_line, use_container_width=True)

# TAB 3 : Volumes 
with tabs[2]:
    st.markdown("**Transactions par mois**")
    vol_mois = (df_f.groupby("mois").size().reset_index(name="nb"))
    fig_bar = px.bar(vol_mois, x="mois", y="nb")
    fig_bar.update_layout(template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

# TAB 4 : Typologie 
with tabs[3]:
    st.markdown("**Répartition par type de bien**")
    type_count = (df_f.groupby("type_local").size().reset_index(name="nb"))
    fig_pie = px.pie(type_count, names="type_local", values="nb", hole=0.35)
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

# TAB 5 : Communes 
with tabs[4]:
    st.markdown("**Top communes (transactions et prix médian)**")
    agg_commune = (df_f.groupby(["code_commune","nom_commune"], as_index=False)
                     .agg(
                         transactions=("prix_m2","size"),
                         prix_m2_med=("prix_m2","median"),
                         surface_med=("surface_reelle_bati","median")
                     )
                   .sort_values(["transactions","prix_m2_med"], ascending=[False, False]))

    top_n = st.slider("Top N communes (par transactions)", 5, 50, 20)
    colA, colB = st.columns(2)

    with colA:
        st.write("**Top N par transactions**")
        st.dataframe(agg_commune.head(top_n), use_container_width=True)

    with colB:
        st.write("**Top N par prix médian/m²**")
        top_price = agg_commune.sort_values("prix_m2_med", ascending=False).head(top_n)
        st.dataframe(top_price, use_container_width=True)


