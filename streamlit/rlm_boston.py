import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the California Housing dataset
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['MedHouseVal'] = housing.target  # Target variable is median house value

# Title of the app
st.title("Predicción de Valores de Vivienda en California usando Regresión Lineal Múltiple")

# Display data preview
st.write("Vista previa de los datos:", data.head())

# Display summary statistics
st.subheader("Estadísticas Descriptivas")
st.write(data.describe())

# Correlation heatmap
st.subheader("Matriz de Correlación")
corr_matrix = data.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# Selecting independent and dependent variables
X = data.drop("MedHouseVal", axis=1)
y = data["MedHouseVal"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Evaluación del Modelo")
st.write(f"**Error Cuadrático Medio (MSE):** {mse:.2f}")
st.write(f"**R² (Coeficiente de Determinación):** {r2:.2f}")

# Display coefficients
st.subheader("Coeficientes del Modelo")
coef_df = pd.DataFrame({"Variable": X.columns, "Coeficiente": model.coef_})
st.write(coef_df)

# Prediction section
st.subheader("Hacer una Predicción")
prediction_input = {}
for col in X.columns:
    prediction_input[col] = st.number_input(f"Valor para {col}", value=float(X[col].mean()))

# Make prediction
if st.button("Predecir Precio"):
    input_data = np.array([list(prediction_input.values())])
    prediction = model.predict(input_data)
    st.write(f"El valor estimado de la vivienda es: ${prediction[0]:.2f} (en cientos de miles de dólares)")
