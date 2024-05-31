import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Funciones necesarias para el modelo
def compute_cost(X, y, theta):
    m = len(y)
    J = (1/(2*m)) * np.sum(np.square(np.dot(X, theta) - y))
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        theta = theta - (alpha/m) * np.dot(X.T, (np.dot(X, theta) - y))
        J_history[i] = compute_cost(X, y, theta)
    return theta, J_history

# Cargar datos
data = pd.read_csv('AmesHousing.csv')

# Seleccionar características y variable objetivo
features = [
    'Overall Qual', 'Gr Liv Area', 'Total Bsmt SF', 'Year Built',
    'Year Remod/Add', 'Garage Cars', 'Garage Area', '1st Flr SF',
    'Full Bath', 'TotRms AbvGrd'
]
target = 'SalePrice'

data = data[features + [target]].dropna()
X = data[features].values
y = data[target].values

# Normalizar las características
mean_X = X.mean(axis=0)
std_X = X.std(axis=0)
X = (X - mean_X) / std_X

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Añadir el término de sesgo (intercepto) a los datos de entrenamiento y prueba
X_train = np.concatenate([np.ones((X_train.shape[0], 1)), X_train], axis=1)
X_test = np.concatenate([np.ones((X_test.shape[0], 1)), X_test], axis=1)

# Inicializar parámetros
theta = np.zeros(X_train.shape[1])
alpha = 0.01
num_iters = 1000

# Ejecutar descenso de gradiente
theta, J_history = gradient_descent(X_train, y_train, theta, alpha, num_iters)

# Función para realizar la predicción
def predict_price(new_house, theta, mean_X, std_X):
    # Normalizar la nueva entrada usando la media y la desviación estándar del conjunto de entrenamiento
    new_house_normalized = (new_house - mean_X) / std_X
    # Añadir término de sesgo
    new_house_normalized_with_bias = np.concatenate([[1], new_house_normalized])
    # Predicción
    predicted_price = np.dot(new_house_normalized_with_bias, theta)
    return predicted_price

# Interfaz de usuario en Streamlit
st.title('Predicción de Precios de Casas')

st.write("""
### Introduce las características de la casa:
""")

# Características de la nueva casa utilizando sliders
overall_qual = st.slider('Calidad general (Overall Qual)', min_value=1, max_value=10, value=5)
gr_liv_area = st.slider('Área habitable (Gr Liv Area)', min_value=300, max_value=6000, value=1500)  # valores aproximados basados en el dataset
total_bsmt_sf = st.slider('Área total del sótano (Total Bsmt SF)', min_value=0, max_value=6000, value=1000)  # valores aproximados basados en el dataset
year_built = st.slider('Año de construcción (Year Built)', min_value=1800, max_value=2024, value=2000)
year_remod_add = st.slider('Año de remodelación/adición (Year Remod/Add)', min_value=1800, max_value=2024, value=2000)
garage_cars = st.slider('Capacidad del garaje (Garage Cars)', min_value=0, max_value=5, value=2)
garage_area = st.slider('Área del garaje (Garage Area)', min_value=0, max_value=1500, value=500)
first_flr_sf = st.slider('Área del primer piso (1st Flr SF)', min_value=300, max_value=4000, value=1000)
full_bath = st.slider('Número de baños completos (Full Bath)', min_value=0, max_value=5, value=2)
tot_rms_abv_grd = st.slider('Número total de habitaciones (TotRms AbvGrd)', min_value=1, max_value=15, value=6)

# Predicción del precio
new_house = np.array([
    overall_qual, gr_liv_area, total_bsmt_sf, year_built, year_remod_add, garage_cars, garage_area, first_flr_sf, full_bath, tot_rms_abv_grd
])

if st.button('Predecir Precio'):
    predicted_price = predict_price(new_house, theta, mean_X, std_X)
    st.write(f'El precio predicho para la casa es: ${predicted_price:.2f}')

# Mostrar la función de costo
st.write("""
### Gráfica de la función de costo durante el entrenamiento:
""")
plt.figure(figsize=(10, 6))
plt.plot(range(num_iters), J_history, 'r')
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.title('Convergencia del descenso de gradiente')
st.pyplot(plt)

# Nuevas casas para predecir
new_houses = [
    np.array([8, 2000, 1500, 2010, 2012, 2, 600, 1300, 2, 9]),  # Casa 1
    np.array([6, 1800, 1200, 2000, 2005, 2, 550, 1250, 2, 7]),  # Casa 2
    np.array([7, 1600, 1100, 2008, 2010, 2, 520, 1220, 2, 8]),  # Casa 3
    np.array([9, 2200, 1700, 2015, 2017, 3, 700, 1400, 3, 10])  # Casa 4
]

# Inicializar una lista para almacenar las predicciones
predictions = []

# Realizar las predicciones para cada nueva casa
for new_house in new_houses:
    predicted_price = predict_price(new_house, theta, mean_X, std_X)
    predictions.append(predicted_price)

# Imprimir las predicciones
for i, predicted_price in enumerate(predictions):
    st.write(f'El precio predicho para la casa {i+1} es: ${predicted_price:.2f}')
