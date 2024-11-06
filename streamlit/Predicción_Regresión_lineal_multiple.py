import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up the app title
st.title("Predicción de Productos Terminados usando Regresión Lineal Múltiple")

# File upload for the dataset
uploaded_file = st.file_uploader("Sube tu archivo de datos en formato Excel", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Display the dataset
    st.write("Vista previa de los datos:", df.head())

    # Variable selection
    x1 = "Horas Trabajadas"
    x2 = "Horas Descanso"
    y = "Productos Terminados"

    # Train the model
    variables_x = [x1, x2]
    variable_y = y
    model = LinearRegression()
    model.fit(df[variables_x], df[variable_y])

    # Model coefficients
    st.write("**Coeficientes del modelo:**", model.coef_)
    st.write("**Intercepción:**", model.intercept_)
    st.write(f"**Ecuación del plano:** y = {round(model.coef_[0], 3)} * {x1} + {round(model.coef_[1], 3)} * {x2} + {round(model.intercept_, 3)}")

    # R2 Score
    r2 = r2_score(df[variable_y], model.predict(df[variables_x]))
    st.write(f"**Coeficiente de determinación (R²):** {round(r2, 3)}")

    # 3D Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[x1], df[x2], df[y], color='blue')

    # Create meshgrid for the plane
    x = np.linspace(df[x1].min(), df[x1].max(), num=10)
    y = np.linspace(df[x2].min(), df[x2].max(), num=10)
    x, y = np.meshgrid(x, y)
    z = model.intercept_ + model.coef_[0] * x + model.coef_[1] * y
    ax.plot_surface(x, y, z, alpha=0.5)

    ax.set_xlabel(x1)
    ax.set_ylabel(x2)
    ax.set_zlabel("Productos Terminados")
    st.pyplot(fig)

    # Prediction input
    st.subheader("Hacer una predicción")
    horas_trabajadas_nuevas = st.number_input("Horas trabajadas (nuevas)", min_value=0)
    horas_descanso_nuevas = st.number_input("Horas de descanso (nuevas)", min_value=0)

    # Prediction calculation
    if st.button("Predecir"):
        prediccion_nueva = pd.DataFrame({x1: [horas_trabajadas_nuevas], x2: [horas_descanso_nuevas]})
        prediccion = model.predict(prediccion_nueva)
        st.write(f"La predicción de productos terminados para {horas_trabajadas_nuevas} horas trabajadas y {horas_descanso_nuevas} horas de descanso es {round(prediccion[0], 3)}")
