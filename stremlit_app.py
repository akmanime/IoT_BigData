# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 1️⃣ Charger le modèle et le scaler
# -----------------------------
clf = joblib.load("rf_smart_rental_guardian.pkl")
scaler = joblib.load("scaler_smart_rental_guardian.pkl")

# Liste des colonnes dans le même ordre que l'entraînement
columns_train = ['debit_eau', 'pression_eau', 'gaz_ppm', 'fumee', 'flamme',
                 'tension', 'courant', 'temperature', 'humidite', 'puissance']

# -----------------------------
# 2️⃣ Fonction de prédiction ajustée
# -----------------------------
def predict_adjusted(clf, scaler, values, columns_train, seuils=None):
    """
    clf : modèle RandomForest
    scaler : scaler utilisé
    values : liste des valeurs capteurs
    columns_train : noms des colonnes
    seuils : dictionnaire des seuils critiques par capteur pour alertes
    """
    df = pd.DataFrame([values], columns=columns_train)
    df_scaled = scaler.transform(df)
    
    proba = clf.predict_proba(df_scaled)[0]
    proba_dict = dict(zip(clf.classes_, proba))
    
    # Si seuil critique externe défini, appliquer logique "if else"
    if seuils:
        for capteur, seuil in seuils.items():
            idx = columns_train.index(capteur)
            if values[idx] > seuil:
                # Remplacer la prédiction par l'anomalie correspondante
                # Ici on peut décider laquelle : simple exemple
                if capteur == "gaz_ppm":
                    return "fuite_gaz", proba_dict
                elif capteur == "debit_eau":
                    return "fuite_eau", proba_dict
                elif capteur == "puissance":
                    return "surcharge", proba_dict
                elif capteur in ["fumee", "flamme", "temperature"]:
                    return "incendie", proba_dict
    
    # Sinon, seuil standard sur probabilité du modèle
    anomalies = {k:v for k,v in proba_dict.items() if k != "normal" and v > 0.5}
    if anomalies:
        pred = max(anomalies, key=lambda k: anomalies[k])
    else:
        pred = "normal"
    
    return pred, proba_dict

# -----------------------------
# 3️⃣ Interface Streamlit
# -----------------------------
st.title("🛡️ Smart Rental Guardian - Détection d'anomalies")

st.write("Entrez les valeurs de vos capteurs pour prédire l'état de votre logement.")

# Entrée utilisateur
debit_eau = st.number_input("Débit d'eau (L/min)", value=1.0)
pression_eau = st.number_input("Pression d'eau (bar)", value=2.0)
gaz_ppm = st.number_input("Concentration gaz (ppm)", value=0.0)
fumee = st.number_input("Fumée (0 ou 1)", value=0)
flamme = st.number_input("Flamme (0 ou 1)", value=0)
tension = st.number_input("Tension (V)", value=230.0)
courant = st.number_input("Courant (A)", value=5.0)
temperature = st.number_input("Température (°C)", value=25.0)
humidite = st.number_input("Humidité (%)", value=40.0)
puissance = st.number_input("Puissance (W)", value=500.0)

valeurs = [debit_eau, pression_eau, gaz_ppm, fumee, flamme,
           tension, courant, temperature, humidite, puissance]

# Seuils critiques (logiques)
seuils_critique = {
    "debit_eau": 5.0,   # si débit > 5 L/min → fuite d'eau
    "gaz_ppm": 600,     # si gaz_ppm > 600 → fuite de gaz
    "puissance": 2000,  # si puissance > 2000 → surcharge
    "fumee": 0.5,       # si fumée détectée → incendie
    "flamme": 0.5,
    "temperature": 60
}

# Bouton pour prédire
if st.button("✅ Prédire l'état"):
    pred, proba = predict_adjusted(clf, scaler, valeurs, columns_train, seuils=seuils_critique)
    
    st.subheader("Résultat :")
    st.write(f"**État prédit : {pred}**")
    
    st.subheader("Probabilités par classe :")
    for k, v in proba.items():
        st.write(f"{k} : {v:.3f}")
