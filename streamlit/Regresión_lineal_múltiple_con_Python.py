import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load data from URL
data_url = "https://raw.githubusercontent.com/JoaquinAmatRodrigo/Estadistica-machine-learning-python/master/data/state_x77.csv"
datos = pd.read_csv(data_url)

# Title of the app
st.title("Regresión Lineal Múltiple con Selección de Variables")

# Display the data preview
st.write("Vista previa de los datos:", datos.head())

# Display data information (alternative to `datos.info()`)
st.write("Información de los datos:")
data_info = pd.DataFrame({
    "Column": datos.columns,
    "Non-Null Count": datos.notnull().sum(),
    "Dtype": datos.dtypes
})
st.write(data_info)

# Split data into independent and dependent variables
X = datos.drop(columns="esp_vida")
y = datos["esp_vida"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1234, shuffle=True)

# Correlation matrix and heatmap
st.subheader("Matriz de Correlación")
corr_matrix = datos.corr(method="pearson")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="viridis", vmin=-1, vmax=1, center=0, square=True, ax=ax)
st.pyplot(fig)

# Histograms of numerical variables
st.subheader("Distribución de Variables Numéricas")
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 8))
axes = axes.flat
columnas_numeric = datos.select_dtypes(include=np.number).columns

for i, colum in enumerate(columnas_numeric):
    sns.histplot(datos[colum], kde=True, color=(list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"], ax=axes[i])
    axes[i].set_title(colum, fontsize=10)
    axes[i].tick_params(labelsize=8)

fig.tight_layout()
st.pyplot(fig)

# Model setup and fitting
st.subheader("Entrenamiento del Modelo de Regresión Lineal")
X_train = sm.add_constant(X_train, prepend=True).rename(columns={"const": "intercept"})
modelo = sm.OLS(y_train, X_train)
modelo_res = modelo.fit()

st.write("Resumen del modelo de regresión lineal:")
st.write(modelo_res.summary())

# Variable selection functions
def forward_selection(X, y, criterio="aic", verbose=False):
    if "intercept" not in X.columns:
        X = sm.add_constant(X)
    seleccion = []
    while len(seleccion) < X.shape[1]:
        remaining = [col for col in X.columns if col not in seleccion]
        new_pval = pd.Series(index=remaining)
        for col in remaining:
            model = sm.OLS(y, X[seleccion + [col]]).fit()
            new_pval[col] = getattr(model, criterio)
        best_feature = new_pval.idxmin()
        seleccion.append(best_feature)
        if verbose:
            st.write(f"Selected variables: {seleccion}")
    return seleccion

def backward_selection(X, y, criterio="aic", verbose=False):
    X = sm.add_constant(X)
    seleccion = list(X.columns)
    while len(seleccion) > 1:
        model = sm.OLS(y, X[seleccion]).fit()
        worst_pval = model.pvalues.idxmax()
        if model.pvalues[worst_pval] > 0.05:
            seleccion.remove(worst_pval)
        else:
            break
        if verbose:
            st.write(f"Remaining variables: {seleccion}")
    return seleccion

# Run forward and backward selection
st.subheader("Selección de Variables")
st.write("Seleccione el criterio de selección de variables y el método.")

criterio = st.selectbox("Criterio", ["aic", "bic", "rsquared_adj"], index=0)
metodo = st.selectbox("Método de selección", ["Forward", "Backward"], index=0)

if metodo == "Forward":
    selected_vars = forward_selection(X_train, y_train, criterio=criterio, verbose=True)
else:
    selected_vars = backward_selection(X_train, y_train, criterio=criterio, verbose=True)

st.write(f"Variables seleccionadas con {metodo} selection y criterio {criterio}: {selected_vars}")

# Refit model with selected variables
X_train_selected = X_train[selected_vars]
model_selected = sm.OLS(y_train, X_train_selected).fit()
st.write("Resumen del modelo con variables seleccionadas:")
st.write(model_selected.summary())
