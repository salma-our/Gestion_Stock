# -*- coding: utf-8 -*-
"""Noyau de calcul : conversion Table 1, critères Table 2, TOPSIS, ABC, ML."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

COLONNES = ["Risk", "Demand fluctuation", "Average stock", "Daily usage",
            "Unit cost", "Lead time", "Consignment stock", "Unit size"]

NATURE = pd.DataFrame({
    "Variable": COLONNES,
    "Nature": ["Qualitative", "Qualitative", "Quantitative", "Quantitative",
               "Quantitative", "Quantitative", "Qualitative", "Qualitative"],
    "Type": ["Ordinale", "Ordinale", "Continue", "Continue",
             "Continue", "Discrète", "Binaire", "Ordinale"],
})

# Table 1 : valeur assignée et valeur normalisée
TABLE1 = {
    "Risk": {"High": (8, 0.47), "Normal": (6, 0.35), "Low": (3, 0.18)},
    "Demand fluctuation": {"Increasing": (90, 0.36), "Stable": (70, 0.28),
                           "Unknown": (50, 0.20), "Decreasing": (40, 0.16),
                           "Ending": (0, 0.00)},
    "Consignment stock": {"No": (4, 0.80), "Yes": (1, 0.20)},
    "Unit size": {"Large": (8, 0.53), "Medium": (5, 0.31), "Small": (2, 0.13)},
}

# Table 2 : critères globaux
TABLE2 = [
    ("Criticality", "Risk", 0.78),
    ("Criticality", "Demand fluctuation", 0.22),
    ("Demand", "Daily usage", 0.71),
    ("Demand", "Average stock", 0.29),
    ("Supply", "Lead time", 0.75),
    ("Supply", "Consignment stock", 0.25),
]
CRITERES = ["Criticality", "Demand", "Supply", "Unit cost", "Unit size"]


def table1_df():
    lignes = []
    for attr, mod in TABLE1.items():
        for m, (a, n) in mod.items():
            lignes.append({"Attribut": attr, "Modalité": m,
                           "Valeur assignée": a, "Valeur normalisée": n})
    return pd.DataFrame(lignes)


def table2_df():
    return pd.DataFrame(TABLE2, columns=["Critère global", "Attribut combiné", "Poids"])


def verifier(df):
    manquantes = [c for c in COLONNES if c not in df.columns]
    if manquantes:
        raise ValueError("Colonnes manquantes : " + ", ".join(manquantes))
    for attr, mod in TABLE1.items():
        inconnues = set(df[attr].dropna().unique()) - set(mod)
        if inconnues:
            raise ValueError(f"Modalités inconnues pour {attr} : {sorted(inconnues)}")
    return df[COLONNES].dropna().reset_index(drop=True)


def convertir(df):
    """Q2 : scores normalisés (Table 1) pour les variables qualitatives."""
    S = pd.DataFrame(index=df.index)
    S["Risk_n"] = df["Risk"].map({k: v[1] for k, v in TABLE1["Risk"].items()})
    S["Demand_fluct_n"] = df["Demand fluctuation"].map(
        {k: v[1] for k, v in TABLE1["Demand fluctuation"].items()})
    S["Consignment_n"] = df["Consignment stock"].map(
        {k: v[1] for k, v in TABLE1["Consignment stock"].items()})
    S["Unit_size_n"] = df["Unit size"].map({k: v[1] for k, v in TABLE1["Unit size"].items()})
    return S


def critères_agrégés(df, S, normaliser=True):
    """Q3 : critères globaux selon les poids additifs de la Table 2."""
    Q = pd.DataFrame(index=df.index)
    for c in ["Average stock", "Daily usage", "Lead time"]:
        x = df[c].astype(float)
        Q[c] = (x - x.min()) / (x.max() - x.min()) if normaliser else x
    C = pd.DataFrame(index=df.index)
    C["Criticality"] = 0.78 * S["Risk_n"] + 0.22 * S["Demand_fluct_n"]
    C["Demand"] = 0.71 * Q["Daily usage"] + 0.29 * Q["Average stock"]
    C["Supply"] = 0.75 * Q["Lead time"] + 0.25 * S["Consignment_n"]
    C["Unit cost"] = df["Unit cost"].astype(float)
    C["Unit size"] = S["Unit_size_n"]
    return C


def topsis(C, poids):
    """Q4 : matrices normalisée et pondérée, solutions idéales, coefficient de proximité."""
    w = np.asarray(poids, float)
    w = w / w.sum()
    X = C.values.astype(float)
    R = pd.DataFrame(X / np.sqrt((X ** 2).sum(axis=0)), columns=C.columns, index=C.index)
    V = R * w
    ideale, anti = V.max(), V.min()
    d_plus = np.sqrt(((V - ideale) ** 2).sum(axis=1))
    d_moins = np.sqrt(((V - anti) ** 2).sum(axis=1))
    cc = d_moins / (d_plus + d_moins)
    return {"R": R, "V": V, "ideale": ideale, "anti": anti,
            "d_plus": d_plus, "d_moins": d_moins, "cc": cc, "poids": w}


def classer(cc, pct_a, pct_b):
    """Q5-Q7 : rang et classe ABC selon les parts d'articles A et B."""
    n = len(cc)
    rang = cc.rank(ascending=False, method="first").astype(int)
    nA, nB = int(round(pct_a / 100 * n)), int(round(pct_b / 100 * n))
    classe = np.where(rang <= nA, "A", np.where(rang <= nA + nB, "B", "C"))
    return rang, pd.Series(classe, index=cc.index)


def résumé_classes(res):
    n = len(res)
    g = res.groupby("Classe")["TOPSIS"].agg(["count", "min", "max", "mean"])
    g.columns = ["Articles", "Score min", "Score max", "Score moyen"]
    g["Part des articles (%)"] = g["Articles"] / n * 100
    g["Part du score total (%)"] = res.groupby("Classe")["TOPSIS"].sum() / res["TOPSIS"].sum() * 100
    return g.reindex(["A", "B", "C"])


FEATURES = ["Risk_n", "Demand_fluct_n", "Consignment_n", "Unit_size_n",
            "Average stock", "Daily usage", "Unit cost", "Lead time"]


def modèles_disponibles(seed=42):
    return {
        "Régression logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "SVM": make_pipeline(StandardScaler(), SVC(C=10)),
        "Naive Bayes": GaussianNB(),
        "Arbre de décision": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=seed),
        "ANN (réseau de neurones)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                          max_iter=2000, random_state=seed)),
    }


def entraîner(df, S, classes, noms, test_size=0.25, seed=42):
    """Q8-Q9 : entraînement, test et indicateurs de performance."""
    X = pd.concat([S, df[["Average stock", "Daily usage", "Unit cost", "Lead time"]]],
                  axis=1)[FEATURES]
    y = classes
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          random_state=seed, stratify=y)
    tous = modèles_disponibles(seed)
    modèles, preds, lignes = {}, {}, []
    for nom in noms:
        m = tous[nom].fit(Xtr, ytr)
        p = m.predict(Xte)
        modèles[nom], preds[nom] = m, p
        lignes.append({"Modèle": nom,
                       "Accuracy": accuracy_score(yte, p),
                       "Precision": precision_score(yte, p, average="macro", zero_division=0),
                       "Recall": recall_score(yte, p, average="macro", zero_division=0),
                       "F1-score": f1_score(yte, p, average="macro", zero_division=0)})
    tableau = pd.DataFrame(lignes).sort_values("F1-score", ascending=False).reset_index(drop=True)
    return {"tableau": tableau, "modèles": modèles, "preds": preds,
            "y_test": yte, "n_train": len(Xtr), "n_test": len(Xte)}


def matrice_confusion(y_test, pred):
    m = confusion_matrix(y_test, pred, labels=["A", "B", "C"])
    return pd.DataFrame(m, index=["A", "B", "C"], columns=["A", "B", "C"])


def rapport(y_test, pred):
    return pd.DataFrame(classification_report(y_test, pred, output_dict=True,
                                              zero_division=0)).T.round(3)


def encoder_article(risk, demand, stock, usage, cost, lead, consign, size):
    """Ligne de variables pour prédire la classe d'un nouvel article."""
    return pd.DataFrame([{
        "Risk_n": TABLE1["Risk"][risk][1],
        "Demand_fluct_n": TABLE1["Demand fluctuation"][demand][1],
        "Consignment_n": TABLE1["Consignment stock"][consign][1],
        "Unit_size_n": TABLE1["Unit size"][size][1],
        "Average stock": stock, "Daily usage": usage,
        "Unit cost": cost, "Lead time": lead}])[FEATURES]