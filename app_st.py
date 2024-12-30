import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score


# *************************************************************** #
@st.cache_data
def load_data():
    """Charger le dataset Boston"""
    # Charger les données depuis l'URL originale
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)

    # Transformation des données
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # Création d'un DataFrame avec les caractéristiques et la cible
    columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
        "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
    ]
    df = pd.DataFrame(data, columns=columns)
    df["MEDV"] = target  # Ajout de la cible
    return df, columns

@st.cache_resource
def train_model(X_train, y_train):
    model = LinearRegression().fit(X_train, y_train)
    return model

# *************************************************************** #

# Chargement des données
df, feature_names = load_data()

# Titre de l'application
st.title("Analyse du Boston Housing Dataset 🏠")


# Section 1 : Aperçu des données ************************************ #
st.header("1. Aperçu des données")

st.write("Voici un aperçu des 5 premières lignes du dataset :")
st.dataframe(df.head())

if st.checkbox("Afficher les statistiques descriptives"):
    st.write(df.describe())
    

# Section 2 : Visualisation ***************************************** #
st.header("2. Visualisation des relations entre les variables")

# Corrélation entre les variables explicatives et la variable cible
st.write("**Matrice de corrélation**")
st.write(df.corr())

x_axis = st.selectbox("Sélectionnez une variable pour l'axe X :", options=feature_names)
y_axis = "MEDV"

fig = plt.figure()
plt.scatter(x=df[x_axis], y=df[y_axis])
plt.title(f"Relation entre {x_axis} et MEDV")
st.pyplot(fig)


# Section 3 : Modèle de régression ********************************** #
st.header("3. Modèle de régression linéaire")

features = st.multiselect(
    "Sélectionnez les variables explicatives :", 
    options=feature_names, 
    default=["CRIM", "DIS", "NOX", "RM", "LSTAT", "PTRATIO"]
)

if features:
    # Préparation des données
    X = df[features]
    y = df["MEDV"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner le modèle
    model = train_model(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Évaluer le modèle
    st.subheader("Résultats du modèle sur les données d'entraînement")
    st.write(f"**RMSE :** {root_mean_squared_error(y_train, model.predict(X_train)):.2f}")
    st.write(f"**R² :** {r2_score(y_train, model.predict(X_train)):.2f}")
    
    st.subheader("Résultats du modèle sur les données de test")
    st.write(f"**RMSE :** {root_mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"**R² :** {r2_score(y_test, y_pred):.2f}")

    # Afficher les coefficients
    coef_df = pd.DataFrame({
        "Variable": features,
        "Coefficient": model.coef_
    })
    st.write("**Coefficients des variables :**")
    st.dataframe(coef_df)

    # Comparaison des valeurs réelles et prédites
    fig = plt.figure()
    plt.scatter(x=y_test, y=y_pred)
    max_val = max(np.max(y_test), np.max(y_pred))
    
    min_val = min(np.min(y_test), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], c='r', lw=2)
    plt.grid()
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Valeurs prédites")
    plt.title("Comparaison des valeurs réelles et prédites")
    
    st.pyplot(fig)
else:
    st.warning("Veuillez sélectionner au moins une variable explicative.")


# Section 4 : Inférence *********************************************** #

st.header("4. Prédictions sur de nouvelles données")
st.write("Saisissez les valeurs des caractéristiques pour prédire la valeur médiane des logements (MEDV).")

if features:
    # Formulaire pour la saisie des données
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"Valeur pour {feature} :", value=float(df[feature].mean()))

    # Bouton pour effectuer la prédiction
    if st.button("Prédire"):
        # Préparer les données pour la prédiction
        input_df = pd.DataFrame([input_data])  # Convertir les valeurs saisies en DataFrame
        prediction = model.predict(input_df)[0]  # Effectuer la prédiction

        # Afficher le résultat de la prédiction
        st.write(f"### Prédiction : ${prediction * 1000:.0f} (valeur médiane)")
else:
    st.warning("Veuillez sélectionner des variables explicatives dans la section précédente pour activer l'inférence.")

