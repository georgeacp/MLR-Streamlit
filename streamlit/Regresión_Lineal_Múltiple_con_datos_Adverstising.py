import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Title of the app
st.title("Predicción de Ventas usando Regresión Lineal Múltiple")

# Load data from URL
data_url = "https://raw.githubusercontent.com/rpizarrog/Analisis-Inteligente-de-datos/main/datos/Advertising_Web.csv"
datos = pd.read_csv(data_url)

# Display the data preview
st.write("Vista previa de los datos:", datos.head())

# Descriptive statistics
st.subheader("Estadísticas Descriptivas")
st.write(datos.describe())

# Pairplot visualization
st.subheader("Visualización de Relaciones entre Variables")
# Create a pairplot figure
pairplot_fig = sns.pairplot(datos, x_vars=['TV', 'Radio', 'Newspaper', 'Web'], y_vars='Sales', kind='reg')
st.pyplot(pairplot_fig)  # Display the pairplot

# Select independent and dependent variables
st.subheader("Selección de Variables para el Modelo")
independent_vars = st.multiselect("Selecciona variables independientes", options=datos.columns[:-1], default=['TV', 'Radio', 'Newspaper', 'Web'])
dependent_var = st.selectbox("Selecciona la variable dependiente", options=[datos.columns[-1]], index=0)

if independent_vars and dependent_var:
    X = datos[independent_vars]
    y = datos[dependent_var]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write("**Errores del Modelo:**")
    st.write(f"MSE (Error Cuadrático Medio): {mse}")
    st.write(f"R² (Coeficiente de Determinación): {r2}")
    
    # Display coefficients
    st.write("**Coeficientes del Modelo:**")
    coef_df = pd.DataFrame({"Variable": independent_vars, "Coeficiente": model.coef_})
    st.write(coef_df)
    
    # Prediction input
    st.subheader("Hacer una Predicción")
    prediction_inputs = {var: st.number_input(f"Valor para {var}", min_value=0.0) for var in independent_vars}
    
    # Make prediction
    if st.button("Predecir"):
        new_data = pd.DataFrame([prediction_inputs])
        prediction = model.predict(new_data)
        st.write(f"La predicción de ventas es: {prediction[0]:.2f}")
