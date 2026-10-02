# -*- coding: utf-8 -*-
"""
Application Streamlit - Classification ABC multi-critères (TOPSIS + Machine Learning)
EMI Rabat - Analyse décisionnelle pour la gestion des stocks multi-attributs

Lancement :  streamlit run app_stock_abc.py
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import abc_core as core

st.set_page_config(page_title="Classification ABC multi-critères", layout="wide")

NAVY, ROUGE, ORANGE, VERT = "#1f3a5f", "#b03a2e", "#d68910", "#1e8449"
COULEURS = alt.Scale(domain=["A", "B", "C"], range=[ROUGE, ORANGE, VERT])

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2rem; max-width: 1300px; }}
    h1, h2, h3 {{ color: {NAVY}; font-weight: 600; letter-spacing: 0.2px; }}
    [data-testid="stMetricValue"] {{ color: {NAVY}; }}
    .bandeau {{ border-left: 4px solid {NAVY}; padding: 0.4rem 1rem; margin-bottom: 1.2rem;
               background: rgba(31,58,95,0.06); }}
    .note {{ font-size: 0.9rem; color: #566573; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Barre latérale : données et paramètres
# ---------------------------------------------------------------------
st.sidebar.title("Paramètres")
fichier = st.sidebar.file_uploader("Base de données (CSV)", type="csv")

défaut = Path(__file__).parent / "inventory_data.csv"
if fichier is not None:
    brut = pd.read_csv(fichier)
elif défaut.exists():
    brut = pd.read_csv(défaut)
    st.sidebar.caption("Fichier par défaut : inventory_data.csv")
else:
    st.title("Classification ABC multi-critères")
    st.info("Importez le fichier inventory_data.csv dans la barre latérale pour commencer.")
    st.stop()

try:
    df = core.verifier(brut)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.sidebar.subheader("Critères agrégés")
normaliser = st.sidebar.checkbox(
    "Normaliser les attributs quantitatifs (min-max) avant agrégation", value=True,
    help="Ramène Average stock, Daily usage et Lead time sur [0, 1], "
         "comme les scores qualitatifs de la Table 1.")

st.sidebar.subheader("Poids TOPSIS")
st.sidebar.caption("Les poids sont renormalisés pour que leur somme soit égale à 1.")
poids_bruts = [st.sidebar.slider(c, 0.0, 1.0, 0.2, 0.05) for c in core.CRITERES]
if sum(poids_bruts) == 0:
    st.sidebar.error("Au moins un poids doit être non nul.")
    st.stop()

st.sidebar.subheader("Seuils ABC")
pct_a = st.sidebar.slider("Classe A (% des articles)", 5, 40, 20, 1)
pct_b = st.sidebar.slider("Classe B (% des articles)", 5, 60, 30, 1)
if pct_a + pct_b >= 100:
    st.sidebar.error("A + B doit rester inférieur à 100 %.")
    st.stop()
st.sidebar.caption(f"Classe C : {100 - pct_a - pct_b} % des articles")

# ---------------------------------------------------------------------
# Calculs
# ---------------------------------------------------------------------
S = core.convertir(df)
C = core.critères_agrégés(df, S, normaliser)
T = core.topsis(C, poids_bruts)
rang, classe = core.classer(T["cc"], pct_a, pct_b)

res = df.copy()
res.insert(0, "Article", np.arange(1, len(res) + 1))
res["TOPSIS"] = T["cc"]
res["Rang"] = rang
res["Classe"] = classe
res = res.sort_values("Rang").reset_index(drop=True)

# ---------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------
st.title("Classification ABC multi-critères des stocks")
st.markdown(
    '<div class="bandeau">Méthode TOPSIS appliquée à 8 attributs d\'inventaire, '
    "puis classification automatique par apprentissage supervisé.</div>",
    unsafe_allow_html=True)

onglets = st.tabs(["Données", "Conversion des scores", "Critères agrégés", "TOPSIS",
                   "Analyse ABC", "Machine learning", "Export"])

# ---------------------------------------------------------------------
# Onglet 1 : données (Q1)
# ---------------------------------------------------------------------
with onglets[0]:
    st.header("Base de données et nature des variables")
    c1, c2, c3 = st.columns(3)
    c1.metric("Articles", len(df))
    c2.metric("Variables", df.shape[1])
    c3.metric("Valeurs manquantes", int(brut[core.COLONNES].isna().sum().sum()))

    g, d = st.columns([1, 1.4])
    with g:
        st.subheader("Nature des variables")
        st.dataframe(core.NATURE, hide_index=True, use_container_width=True)
    with d:
        st.subheader("Statistiques des variables quantitatives")
        quant = ["Average stock", "Daily usage", "Unit cost", "Lead time"]
        st.dataframe(df[quant].describe().T.round(3), use_container_width=True)

    st.subheader("Aperçu")
    st.dataframe(df.head(15), use_container_width=True, hide_index=True)

    st.subheader("Répartition des variables qualitatives")
    cols = st.columns(4)
    for col, nom in zip(cols, ["Risk", "Demand fluctuation", "Consignment stock", "Unit size"]):
        eff = df[nom].value_counts().rename_axis(nom).reset_index(name="Effectif")
        graphe = alt.Chart(eff).mark_bar(color=NAVY).encode(
            x=alt.X(f"{nom}:N", sort="-y", title=None), y=alt.Y("Effectif:Q", title=None),
            tooltip=[nom, "Effectif"]).properties(height=220, title=nom)
        col.altair_chart(graphe, use_container_width=True)

# ---------------------------------------------------------------------
# Onglet 2 : conversion (Q2)
# ---------------------------------------------------------------------
with onglets[1]:
    st.header("Conversion des variables qualitatives (Table 1)")
    st.write(
        "Chaque modalité qualitative est remplacée par son score normalisé. Cette transformation "
        "permet de calculer des distances (TOPSIS) et de combiner des critères de nature "
        "différente sur une même échelle, sans qu'aucun ne domine à cause de son unité.")
    g, d = st.columns([1, 1.6])
    with g:
        st.dataframe(core.table1_df(), hide_index=True, use_container_width=True, height=520)
    with d:
        aperçu = pd.concat([df[["Risk", "Demand fluctuation", "Consignment stock", "Unit size"]],
                            S], axis=1).head(20)
        st.dataframe(aperçu, hide_index=True, use_container_width=True, height=520)

# ---------------------------------------------------------------------
# Onglet 3 : critères agrégés (Q3)
# ---------------------------------------------------------------------
with onglets[2]:
    st.header("Critères agrégés (Table 2)")
    g, d = st.columns([1, 1.6])
    with g:
        st.dataframe(core.table2_df(), hide_index=True, use_container_width=True)
        st.markdown(
            "- **Criticality** : importance opérationnelle de l'article (risque, évolution de la demande).\n"
            "- **Demand** : intensité de la demande (consommation et stock moyen).\n"
            "- **Supply** : difficulté d'approvisionnement (délai, consignation).\n"
            "- **Unit cost** et **Unit size** : contraintes économiques et physiques.")
    with d:
        st.dataframe(C.head(20).round(4), use_container_width=True, hide_index=True)
    if not normaliser:
        st.warning("Sans normalisation, les attributs quantitatifs (échelles plus grandes) "
                   "dominent les scores qualitatifs dans les critères Demand et Supply.")

# ---------------------------------------------------------------------
# Onglet 4 : TOPSIS (Q4-Q5)
# ---------------------------------------------------------------------
with onglets[3]:
    st.header("Méthode TOPSIS")
    st.markdown(
        "1. Matrice de décision normalisée (normalisation vectorielle).\n"
        "2. Matrice pondérée (poids de la barre latérale).\n"
        "3. Solutions idéale (maximum) et anti-idéale (minimum) de chaque critère.\n"
        "4. Distances de chaque article à ces deux solutions.\n"
        "5. Coefficient de proximité : distance anti-idéale / (distance idéale + distance anti-idéale).")
    ref = pd.DataFrame({"Poids": T["poids"], "Solution idéale": T["ideale"].values,
                        "Solution anti-idéale": T["anti"].values}, index=core.CRITERES)
    st.subheader("Poids et solutions de référence")
    st.dataframe(ref.round(5), use_container_width=True)

    g, d = st.columns(2)
    with g:
        st.subheader("Matrice normalisée")
        st.dataframe(T["R"].head(10).round(4), use_container_width=True, hide_index=True)
    with d:
        st.subheader("Matrice pondérée")
        st.dataframe(T["V"].head(10).round(5), use_container_width=True, hide_index=True)

    st.subheader("Classement par coefficient de proximité décroissant")
    top = st.slider("Nombre d'articles affichés", 10, 100, 20, 5)
    st.dataframe(res[["Rang", "Article", "TOPSIS", "Classe"] + core.COLONNES].head(top)
                 .round(4), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# Onglet 5 : ABC (Q6-Q7)
# ---------------------------------------------------------------------
with onglets[4]:
    st.header("Analyse ABC")
    rés = core.résumé_classes(res)
    c1, c2, c3 = st.columns(3)
    for col, k in zip([c1, c2, c3], ["A", "B", "C"]):
        col.metric(f"Classe {k}", f"{int(rés.loc[k, 'Articles'])} articles",
                   f"{rés.loc[k, 'Part du score total (%)']:.1f} % du score total",
                   delta_color="off")
    st.dataframe(rés.round(3), use_container_width=True)
    st.markdown(
        f"<p class='note'>Seuils retenus : A = {pct_a} % des articles, B = {pct_b} %, "
        f"C = {100 - pct_a - pct_b} %. Les articles sont triés par score TOPSIS décroissant, "
        "les meilleurs étant affectés à la classe A (principe de Pareto de l'analyse ABC).</p>",
        unsafe_allow_html=True)

    g, d = st.columns(2)
    with g:
        hist = alt.Chart(res).mark_bar(opacity=0.85).encode(
            x=alt.X("TOPSIS:Q", bin=alt.Bin(maxbins=30), title="Coefficient de proximité"),
            y=alt.Y("count():Q", title="Articles"),
            color=alt.Color("Classe:N", scale=COULEURS)).properties(
            height=320, title="Distribution du score par classe")
        st.altair_chart(hist, use_container_width=True)
    with d:
        pareto = pd.DataFrame({
            "Part des articles (%)": np.arange(1, len(res) + 1) / len(res) * 100,
            "Part cumulée du score (%)": res["TOPSIS"].cumsum() / res["TOPSIS"].sum() * 100})
        ligne = alt.Chart(pareto).mark_line(color=NAVY).encode(
            x="Part des articles (%):Q", y="Part cumulée du score (%):Q")
        seuils = alt.Chart(pd.DataFrame({"x": [pct_a, pct_a + pct_b]})).mark_rule(
            strokeDash=[5, 4], color="gray").encode(x="x:Q")
        st.altair_chart((ligne + seuils).properties(height=320, title="Courbe ABC"),
                        use_container_width=True)

    st.subheader("Base de données avec la variable Classe")
    filtre = st.multiselect("Filtrer par classe", ["A", "B", "C"], default=["A", "B", "C"])
    st.dataframe(res[res["Classe"].isin(filtre)].round(4), use_container_width=True,
                 hide_index=True, height=400)

# ---------------------------------------------------------------------
# Onglet 6 : machine learning (Q8-Q9)
# ---------------------------------------------------------------------
with onglets[5]:
    st.header("Classification automatique par apprentissage supervisé")
    st.write("Les classes ABC issues de TOPSIS servent de labels. Les variables explicatives sont "
             "les 8 attributs d'origine (qualitatifs convertis par la Table 1). Le score TOPSIS "
             "et les critères agrégés sont exclus pour ne pas fournir la réponse aux modèles.")

    c1, c2 = st.columns([2, 1])
    noms = c1.multiselect("Modèles à comparer", list(core.modèles_disponibles()),
                          default=list(core.modèles_disponibles()))
    taille_test = c2.slider("Part du jeu de test", 0.10, 0.40, 0.25, 0.05)

    if st.button("Entraîner et évaluer", type="primary", disabled=not noms):
        with st.spinner("Entraînement en cours"):
            ordre = res.sort_values("Article")
            S_ord = core.convertir(df)
            classes_ord = pd.Series(ordre["Classe"].values, index=df.index)
            st.session_state["ml"] = core.entraîner(df, S_ord, classes_ord, noms, taille_test)

    ml = st.session_state.get("ml")
    if ml is not None:
        st.success(f"Entraînement : {ml['n_train']} articles, test : {ml['n_test']} articles.")
        tab = ml["tableau"]
        st.subheader("Comparaison des performances")
        st.dataframe(tab.style.format({k: "{:.3f}" for k in tab.columns[1:]})
                     .highlight_max(subset=tab.columns[1:], color="#d6eaf8"),
                     hide_index=True, use_container_width=True)

        long = tab.melt("Modèle", var_name="Indicateur", value_name="Valeur")
        barres = alt.Chart(long).mark_bar().encode(
            y=alt.Y("Modèle:N", sort="-x", title=None),
            x=alt.X("Valeur:Q", scale=alt.Scale(domain=[0, 1])),
            yOffset="Indicateur:N", color=alt.Color("Indicateur:N"),
            tooltip=["Modèle", "Indicateur", alt.Tooltip("Valeur:Q", format=".3f")]
        ).properties(height=40 * len(tab) + 80)
        st.altair_chart(barres, use_container_width=True)

        st.subheader("Analyse détaillée d'un modèle")
        choix = st.selectbox("Modèle", tab["Modèle"].tolist())
        g, d = st.columns(2)
        with g:
            mc = core.matrice_confusion(ml["y_test"], ml["preds"][choix])
            long_mc = mc.reset_index().melt("index", var_name="Prédit", value_name="Effectif")
            long_mc = long_mc.rename(columns={"index": "Réel"})
            base = alt.Chart(long_mc).encode(x=alt.X("Prédit:N"), y=alt.Y("Réel:N"))
            heat = base.mark_rect().encode(color=alt.Color("Effectif:Q",
                                                           scale=alt.Scale(scheme="blues")))
            txt = base.mark_text(fontSize=16).encode(text="Effectif:Q")
            st.altair_chart((heat + txt).properties(height=300, title="Matrice de confusion"),
                            use_container_width=True)
        with d:
            st.markdown("**Rapport par classe**")
            st.dataframe(core.rapport(ml["y_test"], ml["preds"][choix]),
                         use_container_width=True)

        st.subheader("Classer un nouvel article")
        with st.form("nouvel_article"):
            a, b, c, d2 = st.columns(4)
            risk = a.selectbox("Risk", list(core.TABLE1["Risk"]))
            dem = b.selectbox("Demand fluctuation", list(core.TABLE1["Demand fluctuation"]))
            cons = c.selectbox("Consignment stock", list(core.TABLE1["Consignment stock"]))
            size = d2.selectbox("Unit size", list(core.TABLE1["Unit size"]))
            e, f, g2, h = st.columns(4)
            stock = e.number_input("Average stock", 0.0, 1000.0, 100.0)
            usage = f.number_input("Daily usage", 0.0, 100.0, 2.5)
            cout = g2.number_input("Unit cost", 0.0, 1000.0, 5.0)
            delai = h.number_input("Lead time", 0, 365, 15)
            envoyer = st.form_submit_button("Prédire la classe")
        if envoyer:
            x_new = core.encoder_article(risk, dem, stock, usage, cout, delai, cons, size)
            pred = ml["modèles"][choix].predict(x_new)[0]
            st.metric(f"Classe prédite ({choix})", pred)
    else:
        st.info("Choisissez les modèles puis lancez l'entraînement.")

# ---------------------------------------------------------------------
# Onglet 7 : export
# ---------------------------------------------------------------------
with onglets[6]:
    st.header("Export des résultats")
    st.write("Base de données complète avec le score TOPSIS, le rang et la variable Classe.")
    st.download_button("Télécharger la base classée (CSV)",
                       res.to_csv(index=False).encode("utf-8"),
                       file_name="inventory_data_classe.csv", mime="text/csv")
    if st.session_state.get("ml") is not None:
        st.download_button("Télécharger la comparaison des modèles (CSV)",
                           st.session_state["ml"]["tableau"].to_csv(index=False).encode("utf-8"),
                           file_name="comparaison_modeles.csv", mime="text/csv")
